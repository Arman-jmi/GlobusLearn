"""Safe semantic cache backed by FAISS; a hit skips retrieval and answer generation (no LLM call).

Keys are standalone (history-resolved) questions, so "Give me an example" is never matched
against another conversation's "Give me an example". Only grounded, in-scope answers are stored,
and the cache is bound to the index fingerprint, so a rebuilt index/PDF invalidates every entry.
"""
import threading

import faiss
import numpy as np

from app.config import get_settings


class SemanticCache:
    def __init__(self, embeddings, fingerprint: str) -> None:
        self._emb = embeddings
        self.fingerprint = fingerprint
        self._index = None
        self._entries: list[dict] = []
        self._lock = threading.Lock()

    def _vec(self, text: str) -> np.ndarray:
        v = np.asarray(self._emb.embed_query(text), dtype="float32")
        return (v / (np.linalg.norm(v) or 1.0)).reshape(1, -1)

    def get(self, question: str) -> dict | None:
        with self._lock:
            if self._index is None or not self._entries:
                return None
            scores, ids = self._index.search(self._vec(question), 1)
        if ids[0][0] >= 0 and scores[0][0] >= get_settings().cache_similarity_threshold:
            return self._entries[ids[0][0]]
        return None

    def put(self, question: str, value: dict) -> None:
        v = self._vec(question)
        with self._lock:
            if self._index is None:
                self._index = faiss.IndexFlatIP(v.shape[1])
            self._index.add(v)
            self._entries.append(value)
