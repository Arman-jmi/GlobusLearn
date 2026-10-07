# NCERT Class 10 Science Chatbot

RAG chatbot that answers only from the NCERT Class 10 Science textbook, cites chapters,
declines out-of-scope questions, supports multi-turn chat, and uses a safe semantic cache.

**Stack:** Python, LangChain/LangGraph, FAISS, FastAPI, Streamlit, OpenAI (`gpt-5.4-mini`).

## Layout

```
app/            FastAPI backend (config, schemas, ingestion, rag, cache, sessions, main)
frontend/       Streamlit UI
scripts/        build_index.py (one-off index build)
data/ncert_science.pdf   NCERT Class 10 Science textbook (the only source)
data/raw/       unused
data/index/     FAISS indexes (not committed)
```

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env   # then set OPENAI_API_KEY (never commit .env)
```

## LLM configuration

| Variable | Default |
|---|---|
| `OPENAI_API_KEY` | required (`LLM_API_KEY` also accepted) |
| `LLM_BASE_URL` | `https://api.openai.com/v1` |
| `LLM_MODEL` | `gpt-5.4-mini` |

Any other OpenAI-compatible provider works by changing these three values.

## How a question is answered

1. Chapter name/number lookups are answered from index metadata.
2. A query-understanding step rewrites the message into a standalone question and a textbook-style search query
   (it never answers, and returns `NOT_TEXTBOOK` for clearly non-science messages, which are refused).
3. FAISS retrieval + the relevance gate run on the search query; the answer is generated only from the retrieved
   excerpts, given the student's original wording (numerical questions are worked using the textbook's own formulas).
4. Numerical answers are not cached because they depend on the student's values.

## Run

```
# rebuilds data/index from scratch (deletes any old index); the API refuses an index built from a different PDF
python scripts/build_index.py
uvicorn app.main:app --reload
streamlit run frontend/streamlit_app.py
```

Tests: `python -m pytest tests` (needs the index built; uses a fake LLM, no key required).
