"""FastAPI app: POST /session, POST /chat, POST /solve-image, GET /health, /conversations (persistent chat history)."""
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from openai import RateLimitError

from app.config import get_settings
from app.rag import RagPipeline
from app.schemas import (
    ChatRequest, ChatResponse, ConversationDetail, ConversationSummary, ImageChatResponse, SessionResponse,
)
from app.sessions import SessionStore
from app.vision import MAX_IMAGE_BYTES, ImageError, QuestionExtractor, sniff_mime

QUOTA_MSG = "The LLM provider's quota is exhausted (OpenAI). Please try again later or use another API key/model."
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["rag"] = RagPipeline.from_settings()
    cfg = get_settings()
    state["sessions"] = SessionStore(cfg.db_path, cfg.upload_dir)
    state["extractor"] = QuestionExtractor(state["rag"].llm)
    yield
    state["sessions"].close()
    state.clear()


app = FastAPI(title="NCERT Class 10 Science Chatbot", lifespan=lifespan)


@app.get("/health")
def health():
    meta = state["rag"].meta
    return {"status": "ok", "source": meta["source"], "pages": meta["pages"], "chunks": meta["chunks"]}


@app.post("/session", response_model=SessionResponse)
def create_session():
    return SessionResponse(session_id=state["sessions"].create())


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    sessions: SessionStore = state["sessions"]
    if not sessions.exists(req.session_id):
        raise HTTPException(status_code=404, detail="Unknown session_id; create one with POST /session")
    started = time.perf_counter()
    try:
        result = state["rag"].ask(req.message, sessions.history(req.session_id))
    except RateLimitError as exc:  # LLM provider quota exhausted: not a gateway fault, and retrying later can succeed
        raise HTTPException(
            status_code=429,
            detail="The LLM provider's quota is exhausted (OpenAI). Please try again later or use another API key/model.",
        ) from exc
    except Exception as exc:  # LLM / network failure
        raise HTTPException(status_code=502, detail=f"Answer generation failed: {exc}") from exc
    latency = int((time.perf_counter() - started) * 1000)
    sessions.add_turn(req.session_id, req.message, result["answer"], result={**result, "latency_ms": latency})
    return ChatResponse(**result, latency_ms=latency)


@app.post("/solve-image", response_model=ImageChatResponse)
async def solve_image(session_id: str = Form(...), image: UploadFile = File(...)):
    """Photo of a question -> extract the question text (no answering) -> the same RAG pipeline in solution mode."""
    sessions: SessionStore = state["sessions"]
    if not sessions.exists(session_id):
        raise HTTPException(status_code=404, detail="Unknown session_id; create one with POST /session")
    data = await image.read(MAX_IMAGE_BYTES + 1)  # bounded read; never buffers an arbitrarily large upload
    started = time.perf_counter()

    def elapsed() -> int:
        return int((time.perf_counter() - started) * 1000)

    try:
        extraction = await run_in_threadpool(state["extractor"].extract, data)
    except ImageError as exc:
        raise HTTPException(status_code=exc.status, detail=exc.message) from exc
    except RateLimitError as exc:
        raise HTTPException(status_code=429, detail=QUOTA_MSG) from exc
    except Exception as exc:  # vision model / network failure; details stay out of the response
        raise HTTPException(status_code=502, detail="Reading the image failed. Please try again.") from exc

    if extraction.error:  # unreadable, or diagram can't be interpreted: friendly message, nothing cached/recorded
        return ImageChatResponse(
            question=extraction.question, answer=extraction.error, in_scope=False, extraction_failed=True,
            latency_ms=elapsed(),
        )
    try:
        result = await run_in_threadpool(
            state["rag"].ask, extraction.question, sessions.history(session_id), "solution"
        )
    except RateLimitError as exc:
        raise HTTPException(status_code=429, detail=QUOTA_MSG) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Answer generation failed. Please try again.") from exc
    sessions.add_turn(
        session_id, extraction.question, result["answer"], result={**result, "latency_ms": elapsed()},
        image=data, mime=sniff_mime(data),
    )
    return ImageChatResponse(question=extraction.question, **result, latency_ms=elapsed())


# ---------------------------------------------------------------- chat history
@app.post("/conversations", response_model=SessionResponse)
def create_conversation():
    """Start a new chat. The returned id doubles as the session_id for /chat and /solve-image."""
    return SessionResponse(session_id=state["sessions"].create())


@app.get("/conversations", response_model=list[ConversationSummary])
def list_conversations(q: str = "", limit: int = 50):
    return state["sessions"].list_all(q, limit)


@app.get("/conversations/{conversation_id}", response_model=ConversationDetail)
def get_conversation(conversation_id: str):
    conv = state["sessions"].get(conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Unknown conversation")
    return conv


@app.get("/conversations/{conversation_id}/messages/{message_id}/image")
def get_message_image(conversation_id: str, message_id: int):
    found = state["sessions"].image(conversation_id, message_id)
    if found is None:
        raise HTTPException(status_code=404, detail="No image for this message")
    data, mime = found
    return Response(content=data, media_type=mime, headers={"Cache-Control": "private, max-age=3600"})


@app.delete("/conversations/{conversation_id}", status_code=204)
def delete_conversation(conversation_id: str):
    if not state["sessions"].delete(conversation_id):
        raise HTTPException(status_code=404, detail="Unknown conversation")
    return Response(status_code=204)
