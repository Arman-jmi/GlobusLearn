"""Streamlit UI for GlobusLearn: landing page -> chat workspace, NCERT source cards, cache info, chapter navigation.

All answers come from the FastAPI backend (separate server locally, embedded in-process on Streamlit Cloud) (/conversations, /chat, /solve-image); this file holds UI state and
layout only. Visual assets live in frontend/static and HTML fragments in frontend/theme.py.
"""
import json
import os
import re
from datetime import datetime
from pathlib import Path

import requests
import streamlit as st

import theme
from embedded_backend import backend_url
INDEX_META = Path(os.getenv("INDEX_META_PATH", Path(__file__).resolve().parent.parent / "data" / "index" / "index_meta.json"))

st.set_page_config(page_title="GlobusLearn · NCERT Class 10 Science", page_icon="🔬", layout="wide", initial_sidebar_state="auto")
theme.inject_assets()

BACKEND_URL = backend_url()  # external server if available, else the FastAPI app embedded in this process

# topic cards: (key, icon, title, description, accent, question sent to the existing chat endpoint)
TOPICS = [
    ("chem", "⚗️", "Chemical Reactions", "Equations, balancing and types of reactions", "#38bdf8", "What is a chemical reaction?"),
    ("light", "🔭", "Light & Optics", "Mirrors, lenses, reflection and refraction", "#818cf8", "What is the relation between R and f?"),
    ("elec", "⚡", "Electricity", "Circuits, Ohm's law, resistance and power", "#fbbf24", "Explain Ohm's law simply."),
    ("life", "🫀", "Life Processes", "Nutrition, respiration, transport and excretion", "#f472b6", "How does photosynthesis occur?"),
    ("hered", "🧬", "Heredity", "Genes, traits and how they are inherited", "#a78bfa", "Explain heredity in simple words."),
    ("env", "🌍", "Our Environment", "Ecosystems, food chains and the ozone layer", "#34d399", "What is an ecosystem?"),
]
NUMERICAL_EXAMPLE = "A concave mirror has a radius of curvature R = 20 cm. Find its focal length."
CONCEPT_EXAMPLE = "Explain refraction of light in simple words."
# quick actions: (key, icon, title, description, accent, action, question)
ACTIONS = [
    ("doubt", "💬", "Ask a Doubt", "Type any Class 10 Science question", "#60a5fa", "chat", None),
    ("photo", "📷", "Solve from Photo", "Upload a picture of a question", "#22d3ee", "photo", None),
    ("num", "🧮", "Solve a Numerical", "Step-by-step with formulas and units", "#a78bfa", "ask", NUMERICAL_EXAMPLE),
    ("concept", "📖", "Explain a Concept", "Simple explanations from NCERT", "#34d399", "ask", CONCEPT_EXAMPLE),
]
# empty-chat suggestions: (label shown, question sent)
SUGGESTIONS = [
    ("🌅  Why is the sky blue?", "Why is the sky blue?"),
    ("🔍  What happens at 2F?", "What happens when an object is placed at 2F of a convex lens?"),
    ("⚡  Explain Ohm's law simply", "Explain Ohm's law simply."),
    ("🧮  Solve R = 20 cm", NUMERICAL_EXAMPLE),
]
FOLLOWUPS = [
    ("💡 Explain simply", "Explain this simply."),
    ("🧮 Solve step by step", "Solve this numerically, step by step."),
    ("📖 Where in NCERT?", "Where is this covered in the NCERT textbook?"),
]
REFUSAL_FRIENDLY = (
    "I can help with questions covered in the NCERT Class 10 Science textbook. "
    "Try asking me about a Science topic from your textbook!"
)
UNAVAILABLE = "⚠️ The study engine is temporarily unavailable."
IMAGE_TYPES = ["jpg", "jpeg", "png", "webp"]
MAX_IMAGE_MB = 5
TEXT_STEPS = ["🔍 Understanding your question…", "📖 Searching your NCERT textbook…", "🧠 Building your answer…"]
PHOTO_STEPS = ["🔍 Reading your question…", "📖 Finding it in NCERT…", "🧠 Building your solution…"]

_FENCE = re.compile(r"(```.*?```|`[^`\n]+`)", re.S)
_BLOCK_BRACKET = re.compile(r"\\\[(.+?)\\\]", re.S)
_INLINE_PAREN = re.compile(r"\\\((.+?)\\\)", re.S)
_LINE_BRACKET = re.compile(r"^(\s*)\[\s*([^\[\]\n]*?(?:=|\\[A-Za-z]+)[^\[\]\n]*?)\s*\]\s*$", re.M)
_PAREN_LATEX = re.compile(r"\(\s*((?:[^()\n]|\([^()\n]*\))*?\\[A-Za-z]+(?:[^()\n]|\([^()\n]*\))*?)\s*\)")
# Section headings the solution mode emits (### Given / Formula / Calculation / Answer / Explanation)
_SECTION = re.compile(
    r"^#{1,4}\s*\**\s*(given|formula|calculation|final answer|answer|explanation)\s*\**\s*:?\s*$", re.I | re.M
)
SECTION_STYLE = {
    "given": ("given", "📋", "Given"),
    "formula": ("formula", "📐", "Formula"),
    "calculation": ("calc", "🧮", "Calculation"),
    "answer": ("answer", "✅", "Final Answer"),
    "final answer": ("answer", "✅", "Final Answer"),
    "explanation": ("explain", "💡", "Explanation"),
}


def _normalize_math(text: str) -> str:
    """Convert LaTeX delimiters to the $...$ / $$...$$ form Streamlit's Markdown renders."""
    def convert(seg: str) -> str:
        seg = _BLOCK_BRACKET.sub(lambda m: f"\n\n$${m.group(1).strip()}$$\n\n", seg)
        seg = _INLINE_PAREN.sub(lambda m: f"${m.group(1).strip()}$", seg)
        # Delimiters whose backslashes were lost: a line that is just "[ equation ]"
        seg = _LINE_BRACKET.sub(lambda m: f"{m.group(1)}$${m.group(2)}$$", seg)
        # "(f = \frac{R}{2})" style inline math: only when it contains a LaTeX command
        seg = _PAREN_LATEX.sub(lambda m: f"${m.group(1).strip()}$", seg)
        return seg

    parts = _FENCE.split(text)
    return "".join(p if i % 2 else convert(p) for i, p in enumerate(parts))


def split_sections(text: str) -> tuple[str, list[tuple[str, str]]]:
    """(text before the first known heading, [(heading, body), ...]). No headings -> (text, [])."""
    found = list(_SECTION.finditer(text))
    if not found:
        return text, []
    sections = []
    for i, m in enumerate(found):
        end = found[i + 1].start() if i + 1 < len(found) else len(text)
        sections.append((m.group(1).lower(), text[m.end():end].strip()))
    return text[:found[0].start()].strip(), sections


@st.cache_data(show_spinner=False)
def load_meta() -> dict:
    """Index metadata written at ingestion time (read-only): chapters + page count."""
    try:
        meta = json.loads(INDEX_META.read_text(encoding="utf-8"))
        chapters = sorted((int(n), str(name)) for n, name in meta.get("chapters", []))
        return {"chapters": chapters, "pages": int(meta.get("pages") or 0)}
    except (OSError, ValueError, TypeError):
        return {"chapters": [], "pages": 0}


# ---------------------------------------------------------------- backend calls (unchanged API usage)
def new_session() -> None:
    """New Chat: a fresh conversation id. Earlier conversations stay in the database / sidebar."""
    r = requests.post(f"{BACKEND_URL}/conversations", timeout=10)
    r.raise_for_status()
    st.session_state.session_id = r.json()["session_id"]
    st.session_state.messages = []
    st.query_params["c"] = st.session_state.session_id


def fetch_image(cid: str, message_id: int) -> bytes | None:
    try:
        r = requests.get(f"{BACKEND_URL}/conversations/{cid}/messages/{message_id}/image", timeout=10)
        return r.content if r.ok else None
    except requests.RequestException:
        return None


def to_ui_messages(conv: dict) -> list[dict]:
    """Stored messages -> the dicts the renderers use (sources / cache info come from the DB, not recomputed)."""
    out, pending_image = [], None
    for m in conv["messages"]:
        if m["role"] == "user":
            msg = {"role": "user", "content": m["content"]}
            if m["has_image"]:
                msg.update(content="📷 Uploaded a question", image=fetch_image(conv["id"], m["id"]))
                pending_image = m["content"]
            out.append(msg)
        else:
            msg = {"role": "assistant", "content": m["content"], "source": m["source"], "sources": m["sources"],
                   "in_scope": m["in_scope"], "latency_ms": m["latency_ms"]}
            if m["cache_hit"] is not None:
                msg["cache_hit"] = m["cache_hit"]
            if pending_image:
                msg["image_question"], pending_image = pending_image, None
            out.append(msg)
    return out


def open_conversation(cid: str) -> bool:
    try:
        r = requests.get(f"{BACKEND_URL}/conversations/{cid}", timeout=10)
        if r.status_code != 200:
            return False
        st.session_state.messages = to_ui_messages(r.json())
    except (requests.RequestException, ValueError, KeyError):
        return False
    st.session_state.session_id = cid
    st.session_state.chat_open = True
    st.query_params["c"] = cid
    return True


def load_recent(query: str) -> list[dict]:
    try:
        r = requests.get(f"{BACKEND_URL}/conversations", params={"q": query, "limit": 20}, timeout=5)
        return r.json() if r.ok else []
    except (requests.RequestException, ValueError):
        return []


def solve_image(job: dict) -> dict:
    """POST the image to the backend; returns the assistant message dict."""
    try:
        r = requests.post(
            f"{BACKEND_URL}/solve-image", data={"session_id": st.session_state.session_id},
            files={"image": (job["name"], job["data"])}, timeout=180,
        )
        if r.status_code in (400, 413, 415, 429):
            return {"role": "assistant", "content": "", "error": True, "error_text": "⚠️ " + str(r.json().get("detail", UNAVAILABLE))}
        r.raise_for_status()
        data = r.json()
        return {"role": "assistant", "content": data["answer"], "image_question": data.get("question"), **data}
    except (requests.RequestException, ValueError, KeyError):
        return {"role": "assistant", "content": UNAVAILABLE, "error": True}


def ask_backend(prompt: str) -> dict:
    try:
        r = requests.post(f"{BACKEND_URL}/chat", json={"session_id": st.session_state.session_id, "message": prompt}, timeout=120)
        r.raise_for_status()
        data = r.json()
        return {"role": "assistant", "content": data["answer"], **data}
    except (requests.RequestException, ValueError, KeyError):
        return {"role": "assistant", "content": UNAVAILABLE, "error": True}


# ---------------------------------------------------------------- UI callbacks (state only; run before the rerun)
def ask(question: str) -> None:
    st.session_state.pending = question
    st.session_state.chat_open = True
    st.session_state.show_upload = False


def open_chat(upload: bool = False) -> None:
    st.session_state.chat_open = True
    st.session_state.show_upload = upload


def toggle_upload(show: bool) -> None:
    st.session_state.show_upload = show


def request_reset(to_chat: bool) -> None:
    """New Chat (to_chat=True: empty chat workspace) or back to the landing page."""
    st.session_state.reset = "chat" if to_chat else "home"


def request_open(cid: str) -> None:
    st.session_state.open_id = cid


def request_delete(cid: str) -> None:
    st.session_state.confirm_delete = cid


def confirm_delete(cid: str) -> None:
    try:
        requests.delete(f"{BACKEND_URL}/conversations/{cid}", timeout=10)
    except requests.RequestException:
        pass
    st.session_state.pop("confirm_delete", None)
    if cid == st.session_state.get("session_id"):
        st.session_state.reset = "chat" if st.session_state.get("chat_open") else "home"


def cancel_delete() -> None:
    st.session_state.pop("confirm_delete", None)


def start_solve(data: bytes, name: str) -> None:
    st.session_state.image_job = {"data": data, "name": name}
    st.session_state.show_upload = False
    st.session_state.upload_nonce = st.session_state.get("upload_nonce", 0) + 1  # clears the uploader


def clear_upload() -> None:
    st.session_state.upload_nonce = st.session_state.get("upload_nonce", 0) + 1


# ---------------------------------------------------------------- formatting helpers
def relative_time(iso: str) -> str:
    try:
        then = datetime.fromisoformat(iso).astimezone()
    except ValueError:
        return ""
    now = datetime.now().astimezone()
    secs = (now - then).total_seconds()
    if secs < 60:
        return "Just now"
    if secs < 3600:
        return f"{int(secs // 60)} min ago"
    days = (now.date() - then.date()).days
    if days == 0:
        return f"{int(secs // 3600)} h ago"
    if days == 1:
        return "Yesterday"
    return f"{then:%b} {then.day}" + (f", {then.year}" if then.year != now.year else "")


def page_range(sources: list[dict]) -> str:
    pages = sorted({p["page"] for p in sources if isinstance(p.get("page"), int)})
    if not pages:
        return ""
    return f"{pages[0]}–{pages[-1]}" if pages[0] != pages[-1] else str(pages[0])


# ---------------------------------------------------------------- renderers
def render_meta(m: dict, idx: int) -> None:
    parts = []
    if m.get("source"):
        chap, sep, name = m["source"].partition(" – ")
        parts.append(theme.source_card(chap, name if sep else "", page_range(m.get("sources", []))))
    if "cache_hit" in m:
        parts.append(theme.answer_meta(bool(m.get("cache_hit")), (m.get("latency_ms") or 0) / 1000))
    if parts:
        with st.container(key=f"meta_{idx}"):
            st.markdown("".join(parts), unsafe_allow_html=True)


def render_body(text: str, idx: int) -> None:
    """Answer markdown; solution-mode headings become labelled panels (Given / Formula / Calculation / ...)."""
    intro, sections = split_sections(text)
    if intro:
        st.markdown(_normalize_math(intro))
    for j, (heading, body) in enumerate(sections):
        kind, icon, label = SECTION_STYLE[heading]
        with st.container(key=f"sec_{kind}_{idx}_{j}"):
            st.markdown(theme.section_label(icon, label), unsafe_allow_html=True)
            if body:
                st.markdown(_normalize_math(body))


def render_answer(m: dict, idx: int) -> None:
    st.markdown(theme.ai_head(), unsafe_allow_html=True)
    if m.get("image_question"):
        st.markdown(theme.image_question(m["image_question"]), unsafe_allow_html=True)
    if m.get("error"):
        st.markdown(theme.notice("error", "", m.get("error_text") or UNAVAILABLE), unsafe_allow_html=True)
    elif m.get("extraction_failed"):
        st.markdown(theme.notice("scope", "📷 Couldn't use this image", m["content"]), unsafe_allow_html=True)
    elif m.get("in_scope") is False:
        st.markdown(theme.notice("scope", "📚 Textbook only", REFUSAL_FRIENDLY), unsafe_allow_html=True)
    else:
        render_body(m["content"], idx)


def render_message(m: dict, idx: int) -> None:
    with st.chat_message(m["role"]):
        if m["role"] == "user":
            if m.get("image"):
                st.image(m["image"], width=240)
            st.markdown(m["content"])
        else:
            render_answer(m, idx)
            if not m.get("extraction_failed") and not m.get("error"):
                render_meta(m, idx)


def render_upload() -> None:
    """Photo panel: drop/click -> preview -> explicit 'Solve' (nothing is sent before that)."""
    ukey = f"uploader_{st.session_state.get('upload_nonce', 0)}"
    has_file = st.session_state.get(ukey) is not None
    with st.container(key="photo"):
        head = st.columns([5, 1], vertical_alignment="center")
        head[0].markdown(theme.photo_head(), unsafe_allow_html=True)
        with head[1], st.container(key="photo_close"):
            st.button("✕", key="photo_x", help="Close", on_click=toggle_upload, args=(False,))
        # the uploader stays mounted in both states so its value survives reruns; when a file is chosen it is hidden
        with st.container(key="photo_hidden" if has_file else "photodrop"):
            if not has_file:
                st.markdown(theme.drop_panel(MAX_IMAGE_MB), unsafe_allow_html=True)
            up = st.file_uploader(
                "Upload a photo or screenshot of a Science question", type=IMAGE_TYPES, key=ukey, label_visibility="collapsed",
            )
        if up is not None:
            with st.container(key="photo_preview"):
                st.image(up, caption="Your question")
            with st.container(key="photo_actions", horizontal=True):
                with st.container(key="solve", width="stretch"):
                    st.button("✨  Solve This Problem", type="primary", use_container_width=True,
                              on_click=start_solve, args=(up.getvalue(), up.name))
                st.button("Choose another", key="photo_clear", on_click=clear_upload)


def tilt_card(key: str, html_card: str, label: str, on_click, args=()) -> None:
    """HTML glass card with the real Streamlit button stretched invisibly over it (keyboard-focusable)."""
    with st.container(key=f"tilt_{key}"):
        st.markdown(html_card, unsafe_allow_html=True)
        st.button(label, key=f"btn_{key}", use_container_width=True, on_click=on_click, args=args)


def run_with_loader(steps: list[str], fn, *args) -> dict:
    holder = st.empty()
    holder.markdown(theme.loader(steps), unsafe_allow_html=True)
    try:
        return fn(*args)
    finally:
        holder.empty()


# ---------------------------------------------------------------- session
# The active conversation id lives in session_state (+ ?c= in the URL, so a browser refresh reopens it);
# all conversation content comes from the backend's SQLite database.
try:
    reset = st.session_state.pop("reset", None)
    if reset:
        for k in ("pending", "image_job", "session_id", "show_upload"):
            st.session_state.pop(k, None)
        st.session_state.chat_open = reset == "chat"
        st.session_state.upload_nonce = st.session_state.get("upload_nonce", 0) + 1
        st.query_params.pop("c", None)
    open_id = st.session_state.pop("open_id", None) or (None if "session_id" in st.session_state else st.query_params.get("c"))
    if open_id:
        for k in ("pending", "image_job", "show_upload"):
            st.session_state.pop(k, None)
        st.session_state.upload_nonce = st.session_state.get("upload_nonce", 0) + 1
        open_conversation(open_id)
    if "session_id" not in st.session_state:
        new_session()
except requests.RequestException:
    st.markdown(theme.notice("error", "", UNAVAILABLE), unsafe_allow_html=True)
    st.stop()

meta = load_meta()
chapters = meta["chapters"]

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    with st.container(key="side_brand"):
        st.markdown(theme.sidebar_brand(), unsafe_allow_html=True)
    with st.container(key="newchat"):
        st.button("＋  New Chat", type="primary", use_container_width=True, on_click=request_reset, args=(True,))
    st.markdown(theme.side_label("RECENT CHATS"), unsafe_allow_html=True)
    with st.container(key="chatsearch"):
        search = st.text_input("Search chats", placeholder="Search chats", label_visibility="collapsed", key="chat_search")
    recent = load_recent(search)
    pending_del = st.session_state.get("confirm_delete")
    if pending_del:
        with st.container(key="confirmdel"):
            st.markdown("Delete this conversation? This can't be undone.")
            c1, c2 = st.columns(2)
            c1.button("Delete", key="del_yes", use_container_width=True, on_click=confirm_delete, args=(pending_del,))
            c2.button("Cancel", key="del_no", use_container_width=True, on_click=cancel_delete)
    with st.container(key="recent"):
        if not recent:
            st.caption("No matching chats." if search else "Your conversations will appear here.")
        for conv in recent:
            active = conv["id"] == st.session_state.get("session_id")
            title = conv["title"] or "New chat"
            with st.container(key=f"{'crowact_' if active else 'crow_'}{conv['id']}"):
                st.markdown(theme.recent_row(title, relative_time(conv["updated_at"]), "💬"), unsafe_allow_html=True)
                st.button(f"Open: {title}", key=f"open_{conv['id']}", on_click=request_open, args=(conv["id"],))
                st.button("✕", key=f"del_{conv['id']}", help="Delete this conversation", on_click=request_delete,
                          args=(conv["id"],))
    st.markdown(theme.side_label("SCIENCE CHAPTERS"), unsafe_allow_html=True)
    with st.container(key="chapters"):
        if chapters:
            for num, name in chapters:
                st.button(f":blue[**{num:02d}**]  {name}", key=f"ch_{num}", use_container_width=True,
                          on_click=ask, args=(f"What is Chapter {num} ({name}) about?",))
        else:
            st.caption("Chapter list unavailable.")
    st.markdown(
        '<div class="gl-side-note"><b>✓ NCERT grounded.</b> Answers come only from the NCERT Class 10 Science '
        "textbook, with the chapter and pages shown.</div>",
        unsafe_allow_html=True,
    )

in_chat = (bool(st.session_state.messages) or st.session_state.get("chat_open", False)
           or "pending" in st.session_state or "image_job" in st.session_state)

# ---------------------------------------------------------------- landing page
if not in_chat:
    with st.container(key="nav"):
        ncols = st.columns([5, 1.3], vertical_alignment="center")
        ncols[0].markdown(theme.nav_links(), unsafe_allow_html=True)
        with ncols[1], st.container(key="nav_start"):
            st.button("Start Learning →", key="nav_cta", type="primary", on_click=open_chat)

    with st.container(key="hero"):
        hcols = st.columns([1.08, 1], vertical_alignment="center", gap="large")
        with hcols[0]:
            st.markdown(theme.hero_copy(), unsafe_allow_html=True)
            with st.container(key="hero_ctas", horizontal=True):
                st.button("Ask a Question →", key="cta_ask", type="primary", on_click=open_chat)
                st.button("📷  Upload a Problem", key="cta_upload", on_click=open_chat, args=(True,))
            st.markdown(theme.hero_stats(len(chapters), meta["pages"]), unsafe_allow_html=True)
        with hcols[1]:
            st.markdown(theme.science_core(), unsafe_allow_html=True)

    st.markdown(theme.section_head("EXPLORE SCIENCE", "Pick a topic to start with",
                                   "Each card asks GlobusLearn a starter question from that chapter.", "explore"),
                unsafe_allow_html=True)
    for row in range(0, len(TOPICS), 3):
        cols = st.columns(3, gap="medium")
        for col, (key, icon, title, desc, hue, q) in zip(cols, TOPICS[row:row + 3]):
            with col:
                tilt_card(f"topic_{key}", theme.topic_card(icon, title, desc, hue), title, ask, (q,))

    st.markdown(theme.section_head("QUICK ACTIONS", "What do you want to do?"), unsafe_allow_html=True)
    acols = st.columns(4, gap="medium")
    for col, (key, icon, title, desc, hue, action, q) in zip(acols, ACTIONS):
        with col:
            if action == "ask":
                tilt_card(f"act_{key}", theme.action_card(icon, title, desc, hue), title, ask, (q,))
            else:
                tilt_card(f"act_{key}", theme.action_card(icon, title, desc, hue), title, open_chat, (action == "photo",))

    st.markdown(theme.section_head("HOW IT WORKS", "How GlobusLearn Works", anchor="how"), unsafe_allow_html=True)
    st.markdown(theme.how_it_works(), unsafe_allow_html=True)

    if chapters:
        st.markdown(theme.section_head("CHAPTERS", "Every chapter of your textbook",
                                       "Tap a chapter for a quick overview, then ask follow-up questions.", "chapters"),
                    unsafe_allow_html=True)
        with st.container(key="chapter_chips", horizontal=True):
            for num, name in chapters:
                st.button(f"{num:02d} · {name}", key=f"chip_{num}", on_click=ask,
                          args=(f"What is Chapter {num} ({name}) about?",))

    st.markdown(theme.trust() + theme.footer(), unsafe_allow_html=True)
    st.stop()

# ---------------------------------------------------------------- chat workspace
current_title = next((c["title"] for c in recent if c["id"] == st.session_state.session_id and c["title"]), None)
if not current_title:
    first_user = next((m["content"] for m in st.session_state.messages if m["role"] == "user"), None)
    current_title = first_user or st.session_state.get("pending") or "New chat"
with st.container(key="topbar"):
    tcols = st.columns([1.3, 3, 1.3], vertical_alignment="center")
    with tcols[0], st.container(key="back"):
        st.button("← GlobusLearn", key="back_btn", on_click=request_reset, args=(False,))
    tcols[1].markdown(theme.chat_title(current_title[:80]), unsafe_allow_html=True)
    with tcols[2], st.container(key="newtop"):
        st.button("＋ New Chat", key="newtop_btn", on_click=request_reset, args=(True,))

# Composer first so it stays mounted (pinned to the bottom) while an answer is being generated.
sent = st.chat_input("Ask your Science doubt...", key="composer", accept_file=True, file_type=IMAGE_TYPES,
                     max_upload_size=MAX_IMAGE_MB)
if sent:
    files = list(getattr(sent, "files", None) or [])
    text = (getattr(sent, "text", None) if not isinstance(sent, str) else sent) or ""
    if files:
        st.session_state.image_job = {"data": files[0].getvalue(), "name": files[0].name}
        st.session_state.show_upload = False
    if text.strip():
        st.session_state.pending = text.strip()  # sent after the photo, if both were given

messages = st.session_state.messages
busy = "pending" in st.session_state or "image_job" in st.session_state

if not messages and not busy:
    st.markdown(theme.empty_state(), unsafe_allow_html=True)
    with st.container(key="suggest"):
        for row in range(0, len(SUGGESTIONS), 2):
            scols = st.columns(2)
            for col, (label, q) in zip(scols, SUGGESTIONS[row:row + 2]):
                col.button(label, key=f"sug_{q}", use_container_width=True, on_click=ask, args=(q,))
    if not st.session_state.get("show_upload"):
        with st.container(key="empty_photo"):
            st.button("📷  Solve from a photo instead", key="empty_photo_btn", on_click=toggle_upload, args=(True,))

for i, m in enumerate(messages):
    render_message(m, i)

# ---------------------------------------------------------------- in-flight requests
job = st.session_state.pop("image_job", None)
if job:
    user_msg = {"role": "user", "content": "📷 Uploaded a question", "image": job["data"]}
    messages.append(user_msg)
    render_message(user_msg, len(messages) - 1)
    with st.chat_message("assistant"):
        st.markdown(theme.ai_head(), unsafe_allow_html=True)
        msg = run_with_loader(PHOTO_STEPS, solve_image, job)
    messages.append(msg)
    st.rerun()

prompt = st.session_state.pop("pending", None)
if prompt:
    user_msg = {"role": "user", "content": prompt}
    messages.append(user_msg)
    render_message(user_msg, len(messages) - 1)
    with st.chat_message("assistant"):
        st.markdown(theme.ai_head(), unsafe_allow_html=True)
        msg = run_with_loader(TEXT_STEPS, ask_backend, prompt)
    messages.append(msg)
    st.rerun()

# ---------------------------------------------------------------- after the conversation
if messages:
    with st.container(key="followups", horizontal=True):
        for label, prompt_text in FOLLOWUPS:
            st.button(label, key=f"fu_{label}", on_click=ask, args=(prompt_text,))
        if not st.session_state.get("show_upload"):
            st.button("📷 Solve from photo", key="fu_photo", on_click=toggle_upload, args=(True,))

if st.session_state.get("show_upload"):
    render_upload()
