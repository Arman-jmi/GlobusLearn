"""Doubt-solving retrieval: informal / numerical / follow-up wording must reach the right textbook content,
while out-of-book questions are still refused. Scripted-LLM tests need no key; `live` tests call OpenAI."""
import pytest
from langchain_core.messages import AIMessage

from app.config import get_settings
from app.ingestion import load_index
from app.rag import REFUSAL, RagPipeline

OPTICS = "Light – Reflection and Refraction"
FOCAL_QUESTIONS = [
    "what is focal length if R = 20 cm?",
    "R=20 find focal length",
    "focal length when radius of curvature is 20 cm",
]


class ScriptedLLM:
    """Stands in for the model. Rewrite step: maps shorthand to a textbook-style query (what the real model does).
    Answer step: records the context/question it was given and replies from that context only."""
    def __init__(self):
        self.contexts, self.answer_inputs, self.rewrites = [], [], []

    def invoke(self, messages):
        sys, human = messages[0].content, messages[1].content
        if "retrieval index" in sys:
            q = human.split("Latest message:")[-1].strip()
            low = q.lower()
            self.rewrites.append(q)
            if "prime minister" in low or "python" in low:
                return AIMessage(content="NOT_TEXTBOOK")
            if "focal length" in low:
                return AIMessage(content=f"QUESTION: {q}\nSEARCH: focal length and radius of curvature relationship for spherical mirrors")
            if "example" in low and "chemical reaction" in human.split("Latest message:")[0].lower():
                return AIMessage(content="QUESTION: What are examples of chemical reactions?\nSEARCH: examples of chemical reactions")
            return AIMessage(content=f"QUESTION: {q}\nSEARCH: {q}")
        self.answer_inputs.append(human)
        self.contexts.append(human.split("</context>")[0])
        return AIMessage(content="Grounded answer from excerpts.")


@pytest.fixture()
def llm():
    return ScriptedLLM()


@pytest.fixture()
def rag(llm):
    store, meta = load_index()
    return RagPipeline(store, meta, llm)


@pytest.mark.parametrize("question", FOCAL_QUESTIONS)
def test_numeric_focal_length_not_refused_and_cited_from_optics(rag, llm, question):
    r = rag.ask(question, [])
    assert r["in_scope"] and r["answer"] != REFUSAL
    assert OPTICS in r["source"] and r["source"].startswith("Chapter 9")
    assert r["sources"] and all(s["chapter"] == OPTICS for s in r["sources"][:2])
    assert all(isinstance(s["page"], int) for s in r["sources"])
    # the retrieved context carries the textbook relationship R = 2f, and the original wording reaches the answer step
    assert "R = 2f" in llm.contexts[-1]
    assert question in llm.answer_inputs[-1]


def test_normal_textbook_question_still_works(rag):
    r = rag.ask("What is a chemical reaction?", [])
    assert r["in_scope"] and r["source"].startswith("Chapter 1")


@pytest.mark.parametrize("question", ["Who is the Prime Minister of India?", "What is Python?", "Tell me today's weather."])
def test_out_of_book_still_refused(rag, llm, question):
    r = rag.ask(question, [])
    assert r["answer"] == REFUSAL and not r["in_scope"] and r["source"] is None and not llm.contexts


def test_unrelated_query_cannot_pass_gate_via_rewrite(rag, llm):
    """Even if a rewrite produced a science-sounding query, the gate/answer step must still be able to refuse."""
    class Sneaky(ScriptedLLM):
        def invoke(self, messages):
            if "retrieval index" in messages[0].content:
                return AIMessage(content="QUESTION: Who won the cricket world cup?\nSEARCH: cricket world cup winner")
            return super().invoke(messages)
    rag.llm = Sneaky()
    r = rag.ask("Who won the cricket world cup?", [])
    assert r["answer"] == REFUSAL


def test_followup_resolved_through_history(rag, llm):
    first = rag.ask("What is a chemical reaction?", [])
    hist = [{"question": "What is a chemical reaction?", "answer": first["answer"]}]
    r = rag.ask("give an example", hist)
    assert r["in_scope"] and r["source"].startswith("Chapter 1")


def test_cache_hit_skips_llm_and_numeric_answers_are_not_cached(rag, llm):
    rag.ask("Explain Ohm's law", [])
    n = len(llm.contexts)
    r = rag.ask("Explain Ohm's law", [])
    assert r["cache_hit"] and len(llm.contexts) == n  # served with no rewrite/answer call
    rag.ask(FOCAL_QUESTIONS[0], [])
    assert not rag.ask(FOCAL_QUESTIONS[0], [])["cache_hit"]  # answers depend on the student's numbers


# -- real OpenAI ------------------------------------------------------------------------------------
live = pytest.mark.skipif(not get_settings().llm_api_key, reason="OPENAI_API_KEY not set")


@pytest.fixture(scope="module")
def live_rag():
    return RagPipeline.from_settings()


@live
@pytest.mark.parametrize("question", FOCAL_QUESTIONS)
def test_live_focal_length_calculation(live_rag, question):
    r = live_rag.ask(question, [])
    assert r["in_scope"] and r["answer"] != REFUSAL
    assert "2f" in r["answer"].replace(" ", "").lower()  # textbook relation R = 2f is shown
    assert "10" in r["answer"]  # f = 20 / 2
    assert r["source"].startswith("Chapter 9") and r["sources"]


@live
@pytest.mark.parametrize("question", ["Who is the Prime Minister of India?", "Tell me today's weather.", "What is Python?"])
def test_live_out_of_book_refused(live_rag, question):
    assert live_rag.ask(question, [])["answer"] == REFUSAL
