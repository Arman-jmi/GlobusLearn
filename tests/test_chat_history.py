"""Persistent chat history: SQLite store + /conversations API. The RAG pipeline is a stub here (the real pipeline is
covered by the other test modules), so these run fast and never touch data/globuslearn.db."""
import pytest
from fastapi.testclient import TestClient

from app import main
from app.conversations import make_title
from app.sessions import SessionStore
from app.vision import Extraction

PNG = b"\x89PNG\r\n\x1a\n" + b"fake-image"
SRC = {"source": "Chapter 9 – Light – Reflection and Refraction", "sources": [{"chapter": "Light", "page": 136}, {"chapter": "Light", "page": 158}]}


class StubRag:
    def __init__(self):
        self.histories = []
        self.llm = None
        self.meta = {"source": "x", "pages": 1, "chunks": 1}

    def ask(self, question, history, mode="chat"):
        self.histories.append(list(history))
        return {"answer": f"Answer to: {question}", **SRC, "in_scope": True, "cache_hit": False}


class StubExtractor:
    def extract(self, data):
        return Extraction(question="Find the focal length if R = 20 cm.", confidence=0.95)


def make_client(db, uploads):
    rag = StubRag()
    main.state["rag"], main.state["sessions"] = rag, SessionStore(db, uploads)
    main.state["extractor"] = StubExtractor()
    return TestClient(main.app), rag


@pytest.fixture
def env(tmp_path):
    client, rag = make_client(tmp_path / "t.db", tmp_path / "uploads")
    yield client, rag, tmp_path
    main.state["sessions"].close()


def new_id(client):
    return client.post("/conversations").json()["session_id"]


def test_create_conversation_and_new_chat_gets_new_id(env):
    client, _, _ = env
    a, b = new_id(client), new_id(client)
    assert a != b
    assert client.get(f"/conversations/{a}").json()["messages"] == []


def test_session_endpoint_still_works(env):
    client, _, _ = env
    sid = client.post("/session").json()["session_id"]
    assert client.post("/chat", json={"session_id": sid, "message": "hi"}).status_code == 200
    assert client.post("/chat", json={"session_id": "nope", "message": "hi"}).status_code == 404


def test_store_and_retrieve_messages_with_source_metadata(env):
    client, _, _ = env
    cid = new_id(client)
    client.post("/chat", json={"session_id": cid, "message": "What is reflection of light?"})
    conv = client.get(f"/conversations/{cid}").json()
    user, ai = conv["messages"]
    assert (user["role"], user["content"]) == ("user", "What is reflection of light?")
    assert ai["source"] == SRC["source"] and ai["sources"] == SRC["sources"]
    assert ai["cache_hit"] is False and ai["in_scope"] is True and ai["latency_ms"] is not None
    assert conv["title"] == "Reflection Light"


def test_continue_conversation_passes_stored_history_to_rag(env):
    client, rag, _ = env
    cid = new_id(client)
    for q in ("What is a chemical reaction?", "Explain it simply.", "Give me an example."):
        client.post("/chat", json={"session_id": cid, "message": q})
    client.post("/chat", json={"session_id": cid, "message": "What happens when it occurs?"})
    last = rag.histories[-1]
    assert [t["question"] for t in last] == ["What is a chemical reaction?", "Explain it simply.", "Give me an example."]
    assert last[0]["answer"].startswith("Answer to:")
    assert rag.histories[0] == []  # first question of a conversation has no history (cache-eligible)


def test_list_newest_first_hides_empty_and_searches(env):
    client, _, _ = env
    empty = new_id(client)
    first, second = new_id(client), new_id(client)
    client.post("/chat", json={"session_id": first, "message": "What is photosynthesis?"})
    client.post("/chat", json={"session_id": second, "message": "Explain Ohm's law simply."})
    listing = client.get("/conversations").json()
    assert [c["id"] for c in listing] == [second, first]
    assert empty not in [c["id"] for c in listing]
    assert [c["id"] for c in client.get("/conversations", params={"q": "photosynth"}).json()] == [first]
    assert client.get("/conversations", params={"q": "100%_"}).json() == []  # LIKE wildcards are escaped
    client.post("/chat", json={"session_id": first, "message": "More please"})  # activity moves it to the top
    assert client.get("/conversations").json()[0]["id"] == first


def test_delete_conversation_removes_messages_and_only_that(env):
    client, _, _ = env
    keep, gone = new_id(client), new_id(client)
    for cid in (keep, gone):
        client.post("/chat", json={"session_id": cid, "message": "What is an acid?"})
    assert client.delete(f"/conversations/{gone}").status_code == 204
    assert client.get(f"/conversations/{gone}").status_code == 404
    assert client.delete(f"/conversations/{gone}").status_code == 404
    assert [c["id"] for c in client.get("/conversations").json()] == [keep]
    assert len(client.get(f"/conversations/{keep}").json()["messages"]) == 2
    n = main.state["sessions"]._db.execute("SELECT COUNT(*) FROM messages WHERE conversation_id = ?", (gone,)).fetchone()[0]
    assert n == 0


def test_conversation_survives_backend_restart(tmp_path):
    db, up = tmp_path / "persist.db", tmp_path / "uploads"
    client, _ = make_client(db, up)
    cid = new_id(client)
    client.post("/chat", json={"session_id": cid, "message": "What is a chemical reaction?"})
    main.state["sessions"].close()  # "restart": a brand-new store/app state on the same file
    client2, rag2 = make_client(db, up)
    conv = client2.get(f"/conversations/{cid}").json()
    assert [m["role"] for m in conv["messages"]] == ["user", "assistant"]
    assert conv["messages"][1]["source"] == SRC["source"] and conv["messages"][1]["sources"] == SRC["sources"]
    client2.post("/chat", json={"session_id": cid, "message": "Explain it simply."})  # continue after restart
    assert rag2.histories[-1][0]["question"] == "What is a chemical reaction?"
    main.state["sessions"].close()


def test_image_question_is_stored_and_served(env):
    client, _, tmp = env
    cid = new_id(client)
    r = client.post("/solve-image", data={"session_id": cid}, files={"image": ("q.png", PNG, "image/png")})
    assert r.status_code == 200
    conv = client.get(f"/conversations/{cid}").json()
    user, ai = conv["messages"]
    assert user["message_type"] == "image" and user["has_image"] and user["content"] == "Find the focal length if R = 20 cm."
    assert ai["message_type"] == "solution" and ai["source"] == SRC["source"]
    assert conv["title"] == "Focal Length Numerical"
    files = list((tmp / "uploads").iterdir())
    assert len(files) == 1 and len(files[0].stem) == 32  # generated unique filename, no client name
    img = client.get(f"/conversations/{cid}/messages/{user['id']}/image")
    assert img.status_code == 200 and img.content == PNG and img.headers["content-type"] == "image/png"
    other = new_id(client)  # an image is only reachable through its own conversation
    assert client.get(f"/conversations/{other}/messages/{user['id']}/image").status_code == 404
    assert client.get(f"/conversations/{cid}/messages/{ai['id']}/image").status_code == 404
    client.delete(f"/conversations/{cid}")
    assert list((tmp / "uploads").iterdir()) == []  # deleting a chat removes its image too


def test_no_secrets_in_database(env):
    client, _, _ = env
    cid = new_id(client)
    client.post("/chat", json={"session_id": cid, "message": "What is an acid?"})
    cols = [r[1] for r in main.state["sessions"]._db.execute("PRAGMA table_info(messages)")]
    assert not {"api_key", "prompt", "system"} & set(cols)


@pytest.mark.parametrize("question,title", [
    ("What is a chemical reaction?", "Chemical Reaction"),
    ("R = 20 cm, find f", "Focal Length Numerical"),
    ("Why is the sky blue?", "Sky Blue"),
    ("Explain the difference between ionic and covalent bonding in great detail please", "Difference Between Ionic Covalent"),
    ("", "New chat"),
])
def test_titles_are_short_and_deterministic(question, title):
    assert make_title(question) == title
