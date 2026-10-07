"""Run the existing FastAPI app inside the Streamlit process (Streamlit Community Cloud has one process only).

If BACKEND_URL is set, or something already answers on localhost:8000 (the normal local setup), that server is
used and nothing is started. Otherwise app.main:app is served by uvicorn on a free loopback port in a background
thread, and the FAISS index is built from data/ncert_science.pdf first if it is missing. No RAG logic lives here.
"""
import os
import socket
import sys
import threading
import time
from pathlib import Path

import requests
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_URL = "http://localhost:8000"


def _healthy(url: str) -> bool:
    try:
        return requests.get(f"{url}/health", timeout=2).ok
    except requests.RequestException:
        return False


def _export_secrets() -> None:
    """Streamlit Cloud Secrets -> environment, so app.config (pydantic-settings) sees OPENAI_API_KEY etc."""
    try:
        for key, value in st.secrets.items():
            if isinstance(value, (str, int, float)):
                os.environ.setdefault(str(key), str(value))
    except Exception:  # no secrets.toml locally: .env is used instead
        pass


@st.cache_resource(show_spinner="Starting the study engine (first start builds the NCERT index, ~1-2 min)…")
def backend_url() -> str:
    explicit = os.getenv("BACKEND_URL")
    if explicit:
        return explicit
    if _healthy(DEFAULT_URL):
        return DEFAULT_URL

    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)  # settings use repo-relative data/ paths
    _export_secrets()
    import uvicorn

    from app.config import get_settings
    from app.ingestion import build_index

    cfg = get_settings()
    if not (ROOT / cfg.index_dir / "index_meta.json").exists():
        build_index()

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config("app.main:app", host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    url = f"http://127.0.0.1:{port}"
    for _ in range(240):  # the lifespan loads the embedding model + index
        if _healthy(url):
            return url
        time.sleep(0.5)
    raise RuntimeError("The embedded study engine did not start (is OPENAI_API_KEY set in Secrets?)")
