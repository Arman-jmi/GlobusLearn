"""CLI entry point: rebuild the FAISS index from data/ncert_science.pdf into data/index (old index is deleted)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ingestion import build_index  # noqa: E402

if __name__ == "__main__":
    meta = build_index()
    print(f"Indexed {meta['pages']} pages -> {meta['chunks']} chunks from {meta['source']}")
    for number, name in meta["chapters"]:
        print(f"  Chapter {number} - {name}")
