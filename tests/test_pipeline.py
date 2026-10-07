"""End-to-end checks on the real FAISS index with a scripted fake LLM (no API key needed).
Requires `python scripts/build_index.py` to have been run."""
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from app import main
from app.ingestion import detect_chapters, load_index
from app.rag import REFUSAL, RagPipeline
from app.sessions import SessionStore


class FakeLLM:
    """Condense: resolves 'its examples' to the previous topic. Answer: echoes context head."""
    def __init__(self):
        self.calls = []

    def invoke(self, messages):
        sys, human = messages[0].content, messages[1].content
        self.calls.append("condense" if "standalone" in sys else "answer")
        self.answers = self.calls.count("answer")
        if "standalone" in sys:
            return AIMessage(content="What are examples of chemical reactions?" if "chemical reaction" in human.lower() else human.split("Latest message:")[-1].strip())
        if "who is the prime minister" in human.lower():
            return AIMessage(content="OUT_OF_SCOPE")
        return AIMessage(content="Grounded answer. Context: " + human[:80])


@pytest.fixture(scope="module")
def rag():
    store, meta = load_index()
    return RagPipeline(store, meta, FakeLLM())


@pytest.fixture(scope="module")
def client(rag):
    main.state["rag"], main.state["sessions"] = rag, SessionStore()
    return TestClient(main.app)  # no `with`: skip lifespan (needs a real key)


def chat(client, sid, msg):
    r = client.post("/chat", json={"session_id": sid, "message": msg})
    assert r.status_code == 200, r.text
    return r.json()


def test_every_page_and_chapter_metadata(rag):
    m = rag.meta
    assert m["pages"] == 217 and len(m["chapters"]) == 13
    d = rag.store.similarity_search("chemical reaction", k=1)[0]
    assert set(d.metadata) >= {"source", "page", "chapter"} and d.metadata["source"] == "ncert_science.pdf"


def test_chapter_detection_boundaries():
    pages = [""] * 3  # synthetic: title page then plain pages
    pages[0] = "Heredity\n8CHAPTER\nBody"
    names = detect_chapters(pages)
    assert names[0] == (8, "Heredity")


def test_A_E_textbook_question_and_chapter(client):
    sid = client.post("/session").json()["session_id"]
    r = chat(client, sid, "What is a chemical reaction?")
    assert r["in_scope"] and r["source"] == "Chapter 1 – Chemical Reactions and Equations" and r["sources"]


def test_B_followup_uses_history(client, rag):
    sid = client.post("/session").json()["session_id"]
    chat(client, sid, "What is a chemical reaction?")
    r = chat(client, sid, "What are its examples?")
    assert r["in_scope"] and "Chapter 1" in r["source"]


def test_C_out_of_book_refused(client):
    sid = client.post("/session").json()["session_id"]
    r = chat(client, sid, "Who is the Prime Minister of India?")
    assert r["answer"] == REFUSAL and not r["in_scope"] and r["source"] is None


def test_D_cache_hit_skips_llm(client, rag):
    sid = client.post("/session").json()["session_id"]
    chat(client, sid, "Explain Ohm's law")
    n = rag.llm.calls.count("answer")
    r = chat(client, sid, "Explain Ohm's law")
    assert r["cache_hit"] and rag.llm.calls.count("answer") == n and r["source"].startswith("Chapter 11")


def test_refusals_not_cached(client, rag):
    sid = client.post("/session").json()["session_id"]
    chat(client, sid, "What is the capital of France?")
    assert not chat(client, sid, "What is the capital of France?")["cache_hit"]


def _chapter_name(rag, number):
    return dict((int(n), name) for n, name in rag.meta["chapters"])[number]


@pytest.mark.parametrize("question,number", [
    ("chapter 13 name", 13),
    ("what is chapter 1?", 1),
    ("what is the name of chapter 5?", 5),
    ("Tell me the name of Chapter 1", 1),
])
def test_chapter_name_lookup_uses_metadata(client, rag, question, number):
    sid = client.post("/session").json()["session_id"]
    n = rag.llm.calls.count("answer")
    r = chat(client, sid, question)
    name = _chapter_name(rag, number)
    assert r["in_scope"] and r["answer"] != REFUSAL and name in r["answer"] and f"Chapter {number}" in r["answer"]
    assert r["source"] == f"Chapter {number} – {name}"
    assert rag.llm.calls.count("answer") == n  # no LLM answer from general knowledge


def test_which_chapter_is_name(client, rag):
    sid = client.post("/session").json()["session_id"]
    r = chat(client, sid, f"Which chapter is {_chapter_name(rag, 6)}?")
    assert r["in_scope"] and r["source"].startswith("Chapter 6")


def test_nonexistent_chapter_and_safety_unchanged(client):
    sid = client.post("/session").json()["session_id"]
    assert "no Chapter 99" in chat(client, sid, "what is chapter 99?")["answer"]
    assert chat(client, sid, "Which chapter is the Prime Minister of India?")["answer"] == REFUSAL


def test_llm_quota_error_is_429_not_502(client, rag, monkeypatch):
    import httpx
    from openai import RateLimitError

    def boom(*a, **k):
        req = httpx.Request("POST", "http://x")
        raise RateLimitError("quota", response=httpx.Response(429, request=req), body=None)
    monkeypatch.setattr(rag.llm, "invoke", boom)
    sid = client.post("/session").json()["session_id"]
    r = client.post("/chat", json={"session_id": sid, "message": "Image by a convex mirror"})
    assert r.status_code == 429 and "quota" in r.json()["detail"]
