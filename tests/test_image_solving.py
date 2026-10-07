"""Photo-question feature: extraction -> existing RAG pipeline (solution mode) -> cited answer.
Uses the real FAISS index with a scripted LLM, so no API key is needed. The scripted model plays three roles:
vision (reads a scenario marker embedded in the fake image bytes), query rewriting, and grounded answering."""
import base64
import json

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from app import main
from app.ingestion import load_index
from app.rag import REFUSAL, RagPipeline
from app.sessions import SessionStore
from app.vision import DIAGRAM_UNCLEAR, MAX_IMAGE_BYTES, UNREADABLE, QuestionExtractor

PNG = b"\x89PNG\r\n\x1a\n"

# scenario marker (inside the fake image) -> what the vision step "reads"
SCENARIOS = {
    "focal": {"question": "R = 20 cm, find focal length", "confidence": 0.96},
    "chem": {"question": "Write the balanced chemical equation for the reaction of iron with steam.", "confidence": 0.93},
    "bio": {"question": "Explain how photosynthesis occurs in plants.", "confidence": 0.91},
    "oos": {"question": "Who is the Prime Minister of India?", "confidence": 0.97},
    "blurry": {"question": None, "confidence": 0.1},
    "lowconf": {"question": "R = ?? cm find f", "confidence": 0.3},
    "diagram": {"question": "Find the image distance for the ray diagram shown.", "confidence": 0.9,
                "needs_diagram": True, "diagram_readable": False},
}


def image(scenario: str) -> bytes:
    return PNG + scenario.encode()


class ScriptedLLM:
    def __init__(self):
        self.vision_calls, self.answer_systems, self.contexts = 0, [], []

    def invoke(self, messages):
        sys, human = messages[0].content, messages[1].content
        if "photos and screenshots" in sys:  # vision / extraction
            self.vision_calls += 1
            url = next(b["image_url"]["url"] for b in human if b["type"] == "image_url")
            raw = base64.b64decode(url.split(",", 1)[1])
            return AIMessage(content="```json\n" + json.dumps(SCENARIOS[raw[len(PNG):].decode()]) + "\n```")
        if "retrieval index" in sys:  # rewrite
            q = human.split("Latest message:")[-1].strip()
            low = q.lower()
            if "prime minister" in low:
                return AIMessage(content="NOT_TEXTBOOK")
            if "focal length" in low:
                return AIMessage(content=f"QUESTION: {q}\nSEARCH: focal length and radius of curvature relationship for spherical mirrors")
            return AIMessage(content=f"QUESTION: {q}\nSEARCH: {q}")
        self.answer_systems.append(sys)
        self.contexts.append(human.split("</context>")[0])
        return AIMessage(content="### Given\nR = 20 cm\n\n### Formula\nR = 2f\n\n### Answer\n**10 cm**")


@pytest.fixture(scope="module")
def llm():
    return ScriptedLLM()


@pytest.fixture(scope="module")
def rag(llm):
    store, meta = load_index()
    return RagPipeline(store, meta, llm)


@pytest.fixture(scope="module")
def client(rag, llm):
    main.state["rag"], main.state["sessions"] = rag, SessionStore()
    main.state["extractor"] = QuestionExtractor(llm)
    return TestClient(main.app)


def sid(client) -> str:
    return client.post("/session").json()["session_id"]


def solve(client, session, data, name="q.png", ctype="image/png"):
    return client.post("/solve-image", data={"session_id": session}, files={"image": (name, data, ctype)})


# -- A/B/C/D: valid images go through the same RAG pipeline ----------------------------------------------
def test_A_B_physics_numerical_image(client, llm):
    n = len(llm.contexts)
    r = solve(client, sid(client), image("focal"))
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["question"] == "R = 20 cm, find focal length" and not d["extraction_failed"]
    assert d["in_scope"] and d["source"].startswith("Chapter 9") and d["sources"]
    assert all(isinstance(s["page"], int) for s in d["sources"])
    assert "10 cm" in d["answer"] and d["latency_ms"] >= 0
    assert len(llm.contexts) == n + 1
    assert "R = 2f" in llm.contexts[-1]  # grounded in the retrieved NCERT excerpt, not model knowledge
    assert "SOLUTION MODE" in llm.answer_systems[-1] and "### Given" in llm.answer_systems[-1]


def test_C_chemistry_image(client):
    d = solve(client, sid(client), image("chem")).json()
    assert d["in_scope"] and d["source"].startswith("Chapter 1") and d["sources"]


def test_D_biology_image(client):
    d = solve(client, sid(client), image("bio")).json()
    assert d["in_scope"] and d["sources"] and "Life Processes" in d["source"]


# -- E: out of scope image is refused without reaching answer generation ----------------------------------
def test_E_out_of_scope_image_refused(client, llm):
    n = len(llm.contexts)
    d = solve(client, sid(client), image("oos")).json()
    assert d["answer"] == REFUSAL and not d["in_scope"] and d["source"] is None and not d["extraction_failed"]
    assert d["question"] == "Who is the Prime Minister of India?"
    assert len(llm.contexts) == n


# -- F: unreadable / low-confidence / unreadable diagram ----------------------------------------------------
@pytest.mark.parametrize("scenario,message", [("blurry", UNREADABLE), ("lowconf", UNREADABLE), ("diagram", DIAGRAM_UNCLEAR)])
def test_F_unreadable_image_gets_friendly_message(client, llm, scenario, message):
    n, hist = len(llm.contexts), main.state["sessions"]
    session = sid(client)
    d = solve(client, session, image(scenario)).json()
    assert d["answer"] == message and d["extraction_failed"] and not d["in_scope"] and not d["cache_hit"]
    assert len(llm.contexts) == n  # never reached answer generation
    assert hist.history(session) == []  # failed extraction is not recorded in the conversation


def test_extraction_garbage_output_is_unreadable():
    class Junk:
        def invoke(self, m):
            return AIMessage(content="Sure! The answer is 42.")
    assert QuestionExtractor(Junk()).extract(PNG + b"x").error == UNREADABLE


# -- G/H: upload validation ----------------------------------------------------------------------------------
def test_G_unsupported_file_type(client, llm):
    calls = llm.vision_calls
    r = solve(client, sid(client), b"%PDF-1.4 not an image", name="q.pdf", ctype="application/pdf")
    assert r.status_code == 415 and "JPG, PNG or WEBP" in r.json()["detail"]
    r = solve(client, sid(client), b"<script>alert(1)</script>", name="evil.png", ctype="image/png")  # lying filename/type
    assert r.status_code == 415
    assert llm.vision_calls == calls  # nothing sent to the model


def test_H_empty_upload(client):
    r = solve(client, sid(client), b"")
    assert r.status_code == 400 and "empty" in r.json()["detail"]


def test_oversized_upload(client):
    r = solve(client, sid(client), PNG + b"\0" * MAX_IMAGE_BYTES)
    assert r.status_code == 413


def test_webp_and_jpeg_accepted_by_sniffing(client):
    from app.vision import sniff_mime
    assert sniff_mime(b"RIFF\0\0\0\0WEBPVP8 ") == "image/webp" and sniff_mime(b"\xff\xd8\xff\xe0abc") == "image/jpeg"


def test_unknown_session(client):
    assert solve(client, "nope", image("bio")).status_code == 404


def test_backend_errors_do_not_leak_details(client, llm, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("secret sk-12345 /internal/path")
    monkeypatch.setattr(llm, "invoke", boom)
    r = solve(client, sid(client), image("bio"))
    assert r.status_code == 502 and "sk-12345" not in r.text and "internal" not in r.text


# -- I: cache -------------------------------------------------------------------------------------------------
def test_I_repeated_image_question_uses_existing_cache(client, llm):
    solve(client, sid(client), image("bio"))  # may already be cached by the earlier test: either way the next is a hit
    n = len(llm.contexts)
    d = solve(client, sid(client), image("bio")).json()
    assert d["cache_hit"] and len(llm.contexts) == n
    assert d["source"] and d["sources"]


def test_I_numeric_image_answers_and_refusals_not_cached(client, llm):
    n = len(llm.contexts)
    solve(client, sid(client), image("focal"))
    assert not solve(client, sid(client), image("focal")).json()["cache_hit"]  # existing rule: numeric answers never cached
    assert len(llm.contexts) == n + 2
    solve(client, sid(client), image("oos"))
    assert not solve(client, sid(client), image("oos")).json()["cache_hit"]
    solve(client, sid(client), image("blurry"))
    assert not solve(client, sid(client), image("blurry")).json()["cache_hit"]


def test_solution_answers_do_not_mix_with_chat_cache(rag, llm):
    rag.ask("What is respiration in living organisms", [])  # chat format cached
    n = len(llm.answer_systems)
    r = rag.ask("What is respiration in living organisms", [], "solution")
    assert not r["cache_hit"] and "SOLUTION MODE" in llm.answer_systems[n]  # generated in solution format, not served from chat
    assert rag.ask("What is respiration in living organisms", [], "solution")["cache_hit"]
    assert "SOLUTION MODE" not in (rag.ask("What is respiration in living organisms", [])["answer"])


# -- J/K/L/M: typed behaviour unchanged (chat mode is the default) ---------------------------------------------
def test_J_typed_questions_still_work(client, llm):
    s = sid(client)
    r = client.post("/chat", json={"session_id": s, "message": "What is a chemical reaction?"}).json()
    assert r["in_scope"] and r["source"].startswith("Chapter 1")
    assert "SOLUTION MODE" not in llm.answer_systems[-1]
    r = client.post("/chat", json={"session_id": s, "message": "R = 20 cm, find f"}).json()
    assert r["in_scope"]


def test_K_followup_after_image_question_sees_history(client, rag):
    s = sid(client)
    solve(client, s, image("bio"))
    hist = main.state["sessions"].history(s)
    assert hist and hist[0]["question"] == SCENARIOS["bio"]["question"]


def test_L_chapter_lookup_still_works(client):
    r = client.post("/chat", json={"session_id": sid(client), "message": "Chapter 13 name"}).json()
    assert r["in_scope"] and r["source"].startswith("Chapter 13")


def test_M_typed_refusal_still_works(client):
    r = client.post("/chat", json={"session_id": sid(client), "message": "Who is the Prime Minister of India?"}).json()
    assert r["answer"] == REFUSAL and not r["in_scope"]
