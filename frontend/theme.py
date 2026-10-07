"""Presentation layer for the Streamlit UI: static CSS/JS assets and small HTML fragment builders.

Everything here is purely visual. Fragments are built without blank lines or leading indentation so
Streamlit's Markdown renderer treats them as a single raw-HTML block. All dynamic text is escaped by the caller
or here with html.escape.
"""
import html
import math
from pathlib import Path

import streamlit as st

STATIC = Path(__file__).resolve().parent / "static"


@st.cache_resource(show_spinner=False)
def _read_cached(name: str, mtime: float) -> str:
    return (STATIC / name).read_text(encoding="utf-8")


def _read(name: str) -> str:
    """Asset text, read from disk only once per file version (the mtime is part of the cache key)."""
    return _read_cached(name, (STATIC / name).stat().st_mtime)


def inject_assets() -> None:
    """Global stylesheet + the small interaction script (parallax / card tilt)."""
    st.markdown(f"<style>{_read('styles.css')}</style>", unsafe_allow_html=True)
    st.html(f"<script>{_read('interactions.js')}</script>", unsafe_allow_javascript=True)


def _h(text: str) -> str:
    return html.escape(str(text))


def logo_mark(uid: str, size: int = 34) -> str:
    """GlobusLearn mark: a glowing core with one tilted orbit and an electron."""
    return (
        f'<svg class="gl-mark" width="{size}" height="{size}" viewBox="0 0 40 40" fill="none" aria-hidden="true">'
        f'<defs><linearGradient id="lg{uid}" x1="0" y1="0" x2="40" y2="40" gradientUnits="userSpaceOnUse">'
        '<stop stop-color="#60a5fa"/><stop offset=".55" stop-color="#818cf8"/><stop offset="1" stop-color="#c084fc"/>'
        f'</linearGradient><radialGradient id="rg{uid}" cx="50%" cy="50%" r="50%">'
        '<stop stop-color="#e0e7ff"/><stop offset=".45" stop-color="#818cf8"/><stop offset="1" stop-color="#6366f1" stop-opacity="0"/>'
        "</radialGradient></defs>"
        f'<circle cx="20" cy="20" r="9" fill="url(#rg{uid})" opacity=".55"/>'
        f'<circle cx="20" cy="20" r="4.2" fill="url(#lg{uid})"/>'
        f'<ellipse cx="20" cy="20" rx="16.5" ry="7" stroke="url(#lg{uid})" stroke-width="2" transform="rotate(-32 20 20)"/>'
        f'<ellipse cx="20" cy="20" rx="16.5" ry="7" stroke="url(#lg{uid})" stroke-width="1.1" opacity=".45" transform="rotate(38 20 20)"/>'
        '<circle cx="33.6" cy="11.6" r="2.3" fill="#7dd3fc"/></svg>'
    )


def brand(uid: str, sub: str = "NCERT • AI • Science") -> str:
    return (
        f'<div class="gl-brand">{logo_mark(uid)}<div class="gl-brand-text"><b>GlobusLearn</b>'
        f"<small>{_h(sub)}</small></div></div>"
    )


def nav_links() -> str:
    return (
        '<div class="gl-nav">' + brand("nav") +
        '<nav class="gl-nav-links" aria-label="Sections">'
        '<a href="#explore">Explore</a><a href="#how">How it works</a><a href="#chapters">Chapters</a></nav></div>'
    )


def hero_copy() -> str:
    return (
        '<div class="gl-hero-copy gl-fade">'
        '<span class="gl-pill"><i>✦</i> AI-POWERED NCERT LEARNING</span>'
        '<h1 class="gl-hero-title">Learn Science.<br><span class="gl-grad">Understand Everything.</span></h1>'
        '<p class="gl-hero-sub">Your intelligent Class 10 Science companion — grounded entirely in your NCERT textbook.</p>'
        "</div>"
    )


def hero_stats(chapters: int, pages: int) -> str:
    items = []
    if chapters:
        items.append(f"<span><b>{chapters}</b> chapters</span>")
    if pages:
        items.append(f"<span><b>{pages}</b> textbook pages indexed</span>")
    items.append("<span><b>NCERT</b> sources on every answer</span>")
    return '<div class="gl-hero-stats gl-fade d2">' + "".join(items) + "</div>"


def science_core() -> str:
    """The hero 'Science Core': a glass sphere with orbits, nucleus, DNA helix, light rays and floating formulas.
    Pure SVG/CSS; the cursor parallax sets --rx/--ry on .gl-core from interactions.js."""
    rungs = "".join(
        f'<line x1="{x}" y1="{296 - a:.1f}" x2="{x}" y2="{296 + a:.1f}"/>'
        for x in range(150, 226, 8)
        if (a := 8 * abs(math.sin(math.pi * (x - 144) / 54))) > 1.5
    )
    particles = "".join(f'<i style="--x:{x}%;--y:{y}%;--d:{d}s;--s:{s}px"></i>' for x, y, d, s in [
        (8, 22, 0, 3), (18, 78, 1.4, 2), (88, 18, 2.2, 3), (92, 64, .6, 2), (72, 92, 3.1, 3), (4, 52, 2.6, 2),
        (30, 6, 1.9, 2), (62, 4, .3, 3), (96, 40, 3.6, 2), (40, 96, 1.1, 2), (14, 92, 4.2, 3), (80, 84, 2.9, 2),
    ])
    return (
        '<div class="gl-core-stage gl-fade d1" aria-hidden="true"><div class="gl-core">'
        '<div class="gl-core-rays"></div><div class="gl-core-halo"></div>'
        f'<div class="gl-core-particles">{particles}</div>'
        '<div class="gl-core-sphere">'
        '<svg viewBox="0 0 400 400" class="gl-core-svg">'
        '<defs><radialGradient id="nuc" cx="50%" cy="50%" r="50%"><stop stop-color="#f0f4ff"/>'
        '<stop offset=".35" stop-color="#a5b4fc"/><stop offset="1" stop-color="#6366f1" stop-opacity="0"/></radialGradient>'
        '<linearGradient id="orb" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#60a5fa"/><stop offset="1" stop-color="#c084fc"/></linearGradient></defs>'
        '<g class="o o1"><ellipse cx="200" cy="200" rx="150" ry="52" stroke="url(#orb)"/><circle cx="350" cy="200" r="5" class="e"/></g>'
        '<g class="o o2"><ellipse cx="200" cy="200" rx="150" ry="52" stroke="url(#orb)" transform="rotate(60 200 200)"/>'
        '<circle cx="125" cy="70" r="4.5" class="e c"/></g>'
        '<g class="o o3"><ellipse cx="200" cy="200" rx="150" ry="52" stroke="url(#orb)" transform="rotate(-60 200 200)"/>'
        '<circle cx="125" cy="330" r="4.5" class="e v"/></g>'
        '<circle cx="200" cy="200" r="58" fill="url(#nuc)" class="nucleus"/>'
        '<circle cx="190" cy="192" r="9" fill="#e0e7ff"/><circle cx="210" cy="196" r="8" fill="#a5b4fc"/>'
        '<circle cx="200" cy="212" r="8.5" fill="#c4b5fd"/><circle cx="214" cy="212" r="6" fill="#e0e7ff" opacity=".8"/>'
        '<g class="dna"><path d="M144 296 C 162 278, 180 314, 198 296 S 234 278, 228 296"/>'
        '<path d="M144 296 C 162 314, 180 278, 198 296 S 234 314, 228 296" class="b"/>'
        f'<g class="r">{rungs}</g></g>'
        '<g class="sym"><text x="92" y="132">∑</text><text x="292" y="296">λ</text><text x="296" y="118">Ω</text>'
        '<text x="104" y="250">π</text></g>'
        "</svg>"
        '<div class="gl-core-gloss"></div></div>'
        '<span class="gl-chip-f f1">V = IR</span><span class="gl-chip-f f2">1/f = 1/v + 1/u</span>'
        '<span class="gl-chip-f f3">R = 2f</span><span class="gl-chip-f f4">C₆H₁₂O₆</span>'
        '<div class="gl-core-shadow"></div>'
        "</div></div>"
    )


def section_head(eyebrow: str, title: str, sub: str = "", anchor: str = "") -> str:
    aid = f' id="{_h(anchor)}"' if anchor else ""
    sub_html = f"<p>{_h(sub)}</p>" if sub else ""
    return f'<div class="gl-sec"{aid}><span class="gl-eyebrow">{_h(eyebrow)}</span><h2>{_h(title)}</h2>{sub_html}</div>'


def topic_card(icon: str, title: str, desc: str, hue: str) -> str:
    return (
        f'<div class="gl-card gl-topic" style="--hue:{hue}"><div class="gl-card-glow"></div>'
        f'<div class="gl-icon3d"><span>{icon}</span></div>'
        f'<div class="gl-card-title">{_h(title)}</div><div class="gl-card-desc">{_h(desc)}</div>'
        '<div class="gl-card-go">Ask GlobusLearn <span>→</span></div></div>'
    )


def action_card(icon: str, title: str, desc: str, hue: str) -> str:
    return (
        f'<div class="gl-card gl-action" style="--hue:{hue}"><div class="gl-card-glow"></div>'
        f'<div class="gl-icon3d sm"><span>{icon}</span></div>'
        f'<div><div class="gl-card-title">{_h(title)}</div><div class="gl-card-desc">{_h(desc)}</div></div>'
        '<span class="gl-arrow">→</span></div>'
    )


def how_it_works() -> str:
    steps = [
        ("01", "ASK", "Ask any Class 10 Science doubt — typed, or as a photo of the question.", "💬"),
        ("02", "GROUND", "GlobusLearn searches your NCERT textbook for the passages that answer it.", "📖"),
        ("03", "UNDERSTAND", "Get a clear explanation with the chapter and pages it came from.", "💡"),
    ]
    cards = "".join(
        f'<div class="gl-step"><div class="gl-step-top"><span class="n">{n}</span><span class="i">{i}</span></div>'
        f"<h3>{t}</h3><p>{_h(d)}</p></div>"
        for n, t, d, i in steps
    )
    return f'<div class="gl-steps"><div class="gl-steps-line"></div>{cards}</div>'


def trust() -> str:
    badges = ["NCERT Grounded", "Chapter Sources", "Smart Cache", "Multi-turn Learning", "Photo Problem Solving"]
    chips = "".join(f'<span class="gl-badge"><i>✓</i>{b}</span>' for b in badges)
    return (
        '<div class="gl-trust"><div class="gl-trust-glow"></div>'
        "<h2>Built around your textbook.</h2>"
        "<p>Answers come only from the NCERT Class 10 Science textbook. If a question isn’t covered there, "
        "GlobusLearn says so instead of guessing.</p>"
        f'<div class="gl-badges">{chips}</div></div>'
    )


def footer() -> str:
    return (
        f'<div class="gl-footer">{brand("foot")}'
        "<span>AI study companion for NCERT Class 10 Science. Always check important answers with your textbook "
        "and teacher.</span></div>"
    )


def sidebar_brand() -> str:
    return brand("side", "Your Science Workspace")


def side_label(text: str) -> str:
    return f'<div class="gl-side-label">{_h(text)}</div>'


def recent_row(title: str, when: str, icon: str) -> str:
    return (
        f'<div class="gl-row"><span class="gl-row-ic">{icon}</span>'
        f'<span class="gl-row-main"><span class="gl-row-t">{_h(title)}</span>'
        f'<span class="gl-row-tm">{_h(when)}</span></span></div>'
    )


def chat_title(title: str) -> str:
    return (
        f'<div class="gl-chat-title"><span class="gl-live"></span><span class="t">{_h(title)}</span>'
        "<small>NCERT Class 10 Science</small></div>"
    )


def empty_state() -> str:
    return (
        '<div class="gl-empty gl-fade">'
        f'<div class="gl-empty-mark">{logo_mark("empty", 56)}</div>'
        "<h2>Hey! 👋<br><span class=\"gl-grad\">What are you curious about today?</span></h2>"
        "<p>Ask a doubt, solve a numerical or upload a photo of a question. "
        "Every answer comes from your NCERT textbook.</p></div>"
    )


def ai_head() -> str:
    return f'<div class="gl-ai-head">{logo_mark("ai", 22)}<b>GlobusLearn</b><span>AI</span></div>'


def section_label(icon: str, label: str) -> str:
    return f'<div class="gl-sec-label"><span>{icon}</span>{_h(label)}</div>'


def image_question(text: str) -> str:
    return f'<div class="gl-qcard"><div class="qt">📷 QUESTION READ FROM YOUR PHOTO</div><div class="qb">{_h(text)}</div></div>'


def notice(kind: str, title: str, body: str) -> str:
    title_html = f"<b>{_h(title)}</b>" if title else ""
    return f'<div class="gl-notice {kind}">{title_html}<p>{_h(body)}</p></div>'


def source_card(chapter: str, name: str, pages: str) -> str:
    title = f"{_h(chapter)} · {_h(name)}" if name else _h(chapter)
    noun = "Pages" if "–" in pages else "Page"
    pages_html = f'<div class="sp">{noun} {_h(pages)}</div>' if pages else ""
    return (
        '<div class="gl-source"><div class="hd"><span class="bk">📖</span>NCERT SOURCE</div>'
        f'<div class="sc">{title}</div>{pages_html}'
        '<div class="ok"><i>✓</i>Answer grounded in NCERT</div></div>'
    )


def answer_meta(cached: bool, seconds: float) -> str:
    if cached:
        return f'<div class="gl-meta cached"><i>⚡</i>Cached · {seconds:.1f}s</div>'
    return f'<div class="gl-meta"><i>✦</i>Generated · {seconds:.1f}s</div>'


def loader(steps: list[str]) -> str:
    """Animated 'thinking' state; the steps fade through in sequence with CSS only."""
    spans = "".join(f'<span style="--i:{i};--n:{len(steps)}">{_h(s)}</span>' for i, s in enumerate(steps))
    return (
        '<div class="gl-loader" role="status" aria-live="polite"><div class="gl-loader-orb"><i></i><i></i><i></i></div>'
        f'<div class="gl-loader-steps">{spans}</div><div class="gl-loader-bar"></div></div>'
    )


def drop_panel(max_mb: int) -> str:
    return (
        '<div class="gl-drop"><div class="gl-drop-ic">📷</div>'
        "<b>Drop your question here</b><span>or click to upload</span>"
        f"<small>JPG · PNG · WEBP · up to {max_mb} MB</small></div>"
    )


def photo_head() -> str:
    return (
        '<div class="gl-photo-head"><b>📷 Solve from a photo</b>'
        "<span>A clear photo or screenshot of one Science question works best.</span></div>"
    )
