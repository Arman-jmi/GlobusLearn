"""Persistent chat history in SQLite (data/globuslearn.db). Separate from the semantic answer cache.

One conversation == one chat session: its id is the `session_id` the /chat endpoints already use, and the stored
turns are what the RAG pipeline receives as multi-turn history. Only conversation content and source metadata are
stored -- never API keys, config or prompts. Uploaded question images live in data/uploads/ under generated names
and are only reachable through their message row (no client-supplied paths).
"""
import json
import re
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id         TEXT PRIMARY KEY,
    title      TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content         TEXT NOT NULL,
    message_type    TEXT NOT NULL DEFAULT 'text',   -- text | image | solution
    source          TEXT,                            -- e.g. "Chapter 9 – Light – Reflection and Refraction"
    sources         TEXT,                            -- JSON [{"chapter":..., "page":...}]
    in_scope        INTEGER,
    cache_hit       INTEGER,
    latency_ms      INTEGER,
    image_file      TEXT,                            -- generated filename inside data/uploads/
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, id);
CREATE INDEX IF NOT EXISTS idx_conv_updated ON conversations(updated_at DESC);
"""

_EXT = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}
_STOP = {
    "what", "whats", "is", "are", "was", "were", "the", "a", "an", "of", "in", "on", "to", "and", "or", "me", "my", "i",
    "do", "does", "did", "how", "why", "when", "where", "which", "who", "explain", "describe", "define", "tell", "about",
    "give", "please", "can", "you", "it", "this", "that", "simply", "simple", "words", "with", "for", "by", "from",
    "us", "briefly", "state", "write", "find", "show", "occur", "occurs", "happen", "happens", "mean", "means",
}
# numerical questions: a few keyword -> topic labels (deterministic, no LLM call)
_NUMERICAL_TOPICS = [
    (r"focal|\bf\b|radius of curvature|\bR\s*=", "Focal Length"),
    (r"mirror", "Mirror"), (r"lens|power of", "Lens"), (r"refractive|snell", "Refraction"),
    (r"resist|ohm|\bV\s*=|\bI\s*=", "Electricity"), (r"current|circuit|voltage|watt", "Electricity"),
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def make_title(question: str, max_words: int = 4) -> str:
    """Short deterministic title from the first question, e.g. 'What is a chemical reaction?' -> 'Chemical Reaction'."""
    text = re.sub(r"\s+", " ", question or "").strip()
    if not text:
        return "New chat"
    if re.search(r"\d", text) and "=" in text:
        for pattern, label in _NUMERICAL_TOPICS:
            if re.search(pattern, text, re.I):
                return f"{label} Numerical"
        return "Numerical Problem"
    words = [w for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9'’-]*", text)]
    keep = [w for w in words if w.lower() not in _STOP][:max_words] or words[:max_words]
    return " ".join(w if w.isupper() and len(w) > 1 else w.capitalize() for w in keep)[:40] or "New chat"


def _message(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["sources"] = json.loads(d["sources"]) if d["sources"] else []
    for k in ("in_scope", "cache_hit"):
        d[k] = None if d[k] is None else bool(d[k])
    d["has_image"] = bool(d.pop("image_file"))
    return d


class ConversationStore:
    def __init__(self, db_path: str | Path = ":memory:", upload_dir: str | Path | None = None) -> None:
        self.db_path = str(db_path)
        self.upload_dir = Path(upload_dir) if upload_dir else None
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.db_path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.executescript(SCHEMA)  # created automatically on first start
        self.purge_empty()

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # ---- conversations ----
    def create(self) -> str:
        cid = uuid.uuid4().hex
        now = _now()
        with self._lock, self._db:
            self._db.execute("INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, '', ?, ?)", (cid, now, now))
        return cid

    def exists(self, cid: str) -> bool:
        with self._lock:
            return self._db.execute("SELECT 1 FROM conversations WHERE id = ?", (cid,)).fetchone() is not None

    def list_all(self, query: str = "", limit: int = 50) -> list[dict]:
        """Newest first; conversations with no messages yet (fresh New Chat) are not listed."""
        sql = ("SELECT c.id, c.title, c.created_at, c.updated_at FROM conversations c "
               "WHERE EXISTS (SELECT 1 FROM messages m WHERE m.conversation_id = c.id)")
        args: list = []
        q = query.strip()
        if q:
            like = "%" + q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
            sql += (" AND (c.title LIKE ? ESCAPE '\\' OR EXISTS (SELECT 1 FROM messages m WHERE m.conversation_id = c.id "
                    "AND m.content LIKE ? ESCAPE '\\'))")
            args += [like, like]
        sql += " ORDER BY c.updated_at DESC, c.rowid DESC LIMIT ?"
        with self._lock:
            return [dict(r) for r in self._db.execute(sql, [*args, max(1, min(limit, 200))])]

    def get(self, cid: str) -> dict | None:
        with self._lock:
            c = self._db.execute("SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?", (cid,)).fetchone()
            if c is None:
                return None
            rows = self._db.execute("SELECT * FROM messages WHERE conversation_id = ? ORDER BY id", (cid,)).fetchall()
        return {**dict(c), "messages": [_message(r) for r in rows]}

    def delete(self, cid: str) -> bool:
        """Removes the conversation, its messages and its uploaded images. Nothing else (index/cache/config) is touched."""
        with self._lock, self._db:
            files = [r[0] for r in self._db.execute(
                "SELECT image_file FROM messages WHERE conversation_id = ? AND image_file IS NOT NULL", (cid,))]
            deleted = self._db.execute("DELETE FROM conversations WHERE id = ?", (cid,)).rowcount > 0
        for name in files:
            self._unlink(name)
        return deleted

    def purge_empty(self, older_than_hours: int = 1) -> None:
        """Housekeeping: drop never-used conversations (a New Chat the student left without asking anything)."""
        cutoff = datetime.now(timezone.utc).timestamp() - older_than_hours * 3600
        cutoff_iso = datetime.fromtimestamp(cutoff, timezone.utc).isoformat(timespec="microseconds")
        with self._lock, self._db:
            self._db.execute(
                "DELETE FROM conversations WHERE created_at < ? AND NOT EXISTS "
                "(SELECT 1 FROM messages m WHERE m.conversation_id = conversations.id)", (cutoff_iso,))

    # ---- messages ----
    def add_message(self, cid: str, role: str, content: str, *, message_type: str = "text", source: str | None = None,
                    sources: list | None = None, in_scope: bool | None = None, cache_hit: bool | None = None,
                    latency_ms: int | None = None, image_file: str | None = None) -> int:
        now = _now()
        with self._lock, self._db:
            cur = self._db.execute(
                "INSERT INTO messages (conversation_id, role, content, message_type, source, sources, in_scope, cache_hit, "
                "latency_ms, image_file, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (cid, role, content, message_type, source, json.dumps(sources) if sources else None,
                 None if in_scope is None else int(in_scope), None if cache_hit is None else int(cache_hit),
                 latency_ms, image_file, now))
            self._db.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, cid))
            if role == "user":  # the first question names the conversation
                self._db.execute("UPDATE conversations SET title = ? WHERE id = ? AND title = ''", (make_title(content), cid))
            return cur.lastrowid

    def add_turn(self, cid: str, question: str, answer: str, *, result: dict | None = None, image: bytes | None = None,
                 mime: str | None = None) -> None:
        """Store one Q&A turn. For photo questions `question` is the extracted text and the image is kept on disk."""
        result = result or {}
        image_file = self._save_image(image, mime) if image else None
        try:
            self.add_message(cid, "user", question, message_type="image" if image_file else "text", image_file=image_file)
            self.add_message(
                cid, "assistant", answer, message_type="solution" if image_file else "text", source=result.get("source"),
                sources=result.get("sources"), in_scope=result.get("in_scope"), cache_hit=result.get("cache_hit"),
                latency_ms=result.get("latency_ms"))
        except Exception:
            if image_file:
                self._unlink(image_file)
            raise

    def history(self, cid: str) -> list[dict]:
        """Completed turns as [{"question", "answer"}], the format RagPipeline.ask expects."""
        with self._lock:
            rows = self._db.execute("SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id", (cid,)).fetchall()
        turns, pending = [], None
        for role, content in rows:
            if role == "user":
                pending = content
            elif pending is not None:
                turns.append({"question": pending, "answer": content})
                pending = None
        return turns

    # ---- images ----
    def _save_image(self, data: bytes, mime: str | None) -> str | None:
        if self.upload_dir is None or mime not in _EXT:
            return None
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        name = uuid.uuid4().hex + _EXT[mime]
        (self.upload_dir / name).write_bytes(data)
        return name

    def _unlink(self, name: str) -> None:
        if self.upload_dir and re.fullmatch(r"[0-9a-f]{32}\.(jpg|png|webp)", name):
            (self.upload_dir / name).unlink(missing_ok=True)

    def image(self, cid: str, message_id: int) -> tuple[bytes, str] | None:
        """Image bytes + mime for a message in this conversation (looked up by row, never by client path)."""
        with self._lock:
            row = self._db.execute(
                "SELECT image_file FROM messages WHERE id = ? AND conversation_id = ?", (message_id, cid)).fetchone()
        name = row[0] if row else None
        if not name or not self.upload_dir or not re.fullmatch(r"[0-9a-f]{32}\.(jpg|png|webp)", name):
            return None
        path = self.upload_dir / name
        if not path.is_file():
            return None
        mime = next(m for m, ext in _EXT.items() if name.endswith(ext))
        return path.read_bytes(), mime
