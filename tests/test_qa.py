"""Tests for QA chain — happy paths, failure paths, structured output verification."""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

from src.qa.schemas import CareerAnswer, SkillGap
from src.qa.chain import CareerQAChain, _format_context, _unique_sources
from src.rag.retriever import Retriever, ScoredDocument


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _make_scored_doc(text: str, source: str = "cv.pdf", score: float = 0.85) -> ScoredDocument:
    doc = Document(page_content=text, metadata={"source": source, "page": 1})
    return ScoredDocument(document=doc, score=score)


def _make_career_answer(**kwargs) -> CareerAnswer:
    defaults = {
        "answer": "You are a strong match for this role.",
        "confidence": 0.85,
        "skill_gaps": [],
        "sources": ["cv.pdf", "jd.pdf"],
        "follow_up_questions": ["What is your notice period?", "Do you have visa sponsorship?"],
    }
    defaults.update(kwargs)
    return CareerAnswer(**defaults)


def _make_mock_retriever(scored_docs: list[ScoredDocument]) -> MagicMock:
    retriever = MagicMock(spec=Retriever)
    retriever.retrieve_all.return_value = scored_docs
    return retriever


# ─── Schema tests ─────────────────────────────────────────────────────────────


class TestCareerAnswerSchema:
    def test_valid_construction(self):
        answer = CareerAnswer(
            answer="Strong fit",
            confidence=0.9,
            skill_gaps=[SkillGap(skill="PyTorch", importance="critical", suggestion="Take fast.ai course")],
            sources=["cv.pdf"],
            follow_up_questions=["Tell me about your projects"],
        )
        assert answer.confidence == 0.9
        assert len(answer.skill_gaps) == 1
        assert answer.skill_gaps[0].importance == "critical"

    def test_confidence_below_zero_raises(self):
        with pytest.raises(Exception):
            CareerAnswer(answer="x", confidence=-0.1)

    def test_confidence_above_one_raises(self):
        with pytest.raises(Exception):
            CareerAnswer(answer="x", confidence=1.1)

    def test_defaults_are_empty_lists(self):
        answer = CareerAnswer(answer="short answer", confidence=0.5)
        assert answer.skill_gaps == []
        assert answer.sources == []
        assert answer.follow_up_questions == []

    def test_skill_gap_importance_must_be_literal(self):
        with pytest.raises(Exception):
            SkillGap(skill="Python", importance="mandatory", suggestion="Learn it")

    def test_skill_gap_valid_values(self):
        critical = SkillGap(skill="Kubernetes", importance="critical", suggestion="Take CKA course")
        nice = SkillGap(skill="Rust", importance="nice-to-have", suggestion="Build a small project")
        assert critical.importance == "critical"
        assert nice.importance == "nice-to-have"


# ─── _format_context ─────────────────────────────────────────────────────────


class TestFormatContext:
    def test_includes_source_and_content(self):
        sd = _make_scored_doc("Python developer", source="cv.pdf", score=0.9)
        ctx = _format_context([sd])
        assert "cv.pdf" in ctx
        assert "Python developer" in ctx
        assert "0.90" in ctx

    def test_multiple_docs_separated(self):
        docs = [
            _make_scored_doc("CV content", source="cv.pdf", score=0.9),
            _make_scored_doc("JD content", source="jd.pdf", score=0.8),
        ]
        ctx = _format_context(docs)
        assert "---" in ctx
        assert "cv.pdf" in ctx
        assert "jd.pdf" in ctx

    def test_numbered_entries(self):
        docs = [_make_scored_doc(f"doc {i}") for i in range(3)]
        ctx = _format_context(docs)
        assert "[1]" in ctx
        assert "[2]" in ctx
        assert "[3]" in ctx


# ─── _unique_sources ─────────────────────────────────────────────────────────


class TestUniqueSources:
    def test_deduplicates(self):
        docs = [
            _make_scored_doc("chunk 1", source="cv.pdf"),
            _make_scored_doc("chunk 2", source="cv.pdf"),
            _make_scored_doc("chunk 3", source="jd.pdf"),
        ]
        sources = _unique_sources(docs)
        assert sources == ["cv.pdf", "jd.pdf"]

    def test_preserves_order(self):
        docs = [
            _make_scored_doc("text", source="jd.pdf"),
            _make_scored_doc("text", source="cv.pdf"),
        ]
        sources = _unique_sources(docs)
        assert sources[0] == "jd.pdf"


# ─── CareerQAChain ───────────────────────────────────────────────────────────


class TestCareerQAChain:
    # ── Happy paths ──────────────────────────────────────────────────────────

    def test_happy_path_returns_career_answer(self, mock_llm):
        scored_docs = [
            _make_scored_doc("ML engineer 5 years Python", source="cv.pdf"),
            _make_scored_doc("We require Python and ML experience", source="jd.pdf"),
        ]
        retriever = _make_mock_retriever(scored_docs)
        expected = _make_career_answer()

        chain = CareerQAChain(retriever=retriever)
        chain._chain = MagicMock()
        chain._chain.invoke.return_value = expected

        result = chain.ask("How well do I fit this role?")

        assert isinstance(result, CareerAnswer)
        assert result.confidence > 0.0
        assert result.answer != ""
        retriever.retrieve_all.assert_called_once_with("How well do I fit this role?")

    def test_sources_populated_from_retrieval_when_llm_returns_empty(self, mock_llm):
        """If LLM returns empty sources, chain falls back to retrieved doc sources."""
        scored_docs = [_make_scored_doc("content", source="cv.pdf")]
        retriever = _make_mock_retriever(scored_docs)
        llm_answer = _make_career_answer(sources=[])

        chain = CareerQAChain(retriever=retriever)
        chain._chain = MagicMock()
        chain._chain.invoke.return_value = llm_answer

        result = chain.ask("What are my strengths?")
        assert "cv.pdf" in result.sources

    def test_skill_gaps_returned_for_fit_question(self, mock_llm):
        scored_docs = [
            _make_scored_doc("5 years Go, some Python", source="cv.pdf"),
            _make_scored_doc("Requires PyTorch, TensorFlow, MLOps", source="jd.pdf"),
        ]
        retriever = _make_mock_retriever(scored_docs)
        expected = _make_career_answer(
            skill_gaps=[
                SkillGap(skill="PyTorch", importance="critical", suggestion="Complete fast.ai course"),
                SkillGap(skill="TensorFlow", importance="nice-to-have", suggestion="Follow TF tutorials"),
            ]
        )
        chain = CareerQAChain(retriever=retriever)
        chain._chain = MagicMock()
        chain._chain.invoke.return_value = expected

        result = chain.ask("What skills am I missing?")
        assert len(result.skill_gaps) == 2
        assert result.skill_gaps[0].skill == "PyTorch"

    # ── Failure paths ─────────────────────────────────────────────────────────

    def test_failure_path_empty_question_raises(self, mock_llm):
        """An empty question must raise ValueError immediately."""
        retriever = _make_mock_retriever([])
        chain = CareerQAChain(retriever=retriever)

        with pytest.raises(ValueError, match="empty"):
            chain.ask("")

    def test_failure_path_whitespace_only_question_raises(self, mock_llm):
        retriever = _make_mock_retriever([])
        chain = CareerQAChain(retriever=retriever)

        with pytest.raises(ValueError, match="empty"):
            chain.ask("   ")

    def test_failure_path_no_documents_returns_graceful_response(self, mock_llm):
        """
        When retriever returns no docs, chain returns a helpful CareerAnswer
        with confidence=0.0 rather than raising an exception.
        """
        retriever = _make_mock_retriever([])
        chain = CareerQAChain(retriever=retriever)
        result = chain.ask("What skills am I missing?")

        assert isinstance(result, CareerAnswer)
        assert result.confidence == 0.0
        assert "upload" in result.answer.lower()
        assert len(result.follow_up_questions) > 0

    def test_failure_path_llm_error_propagates(self, mock_llm):
        """If the LLM call fails, the exception propagates."""
        scored_docs = [_make_scored_doc("some context")]
        retriever = _make_mock_retriever(scored_docs)

        chain = CareerQAChain(retriever=retriever)
        chain._chain = MagicMock()
        chain._chain.invoke.side_effect = RuntimeError("OpenAI API error: 401 Unauthorized")

        with pytest.raises(RuntimeError, match="401 Unauthorized"):
            chain.ask("What are my skill gaps?")

    def test_failure_path_query_returns_no_relevant_chunks(self, mock_llm):
        """Retriever returns empty — chain returns graceful low-confidence response."""
        retriever = _make_mock_retriever([])
        chain = CareerQAChain(retriever=retriever)

        result = chain.ask("Tell me about quantum computing skills")
        assert result.confidence == 0.0
        assert isinstance(result, CareerAnswer)

    # ── Integration: full flow with mocked LLM ────────────────────────────────

    def test_integration_full_flow_mocked_llm(self, mock_llm):
        """
        End-to-end: retriever returns docs → chain formats context →
        LLM (mocked) returns structured CareerAnswer → result validated.
        """
        cv_doc = _make_scored_doc(
            "Bhavadharani: 5 years Go, Python, ML at Elara, MSc AI",
            source="bhavadharani_cv.pdf",
            score=0.92,
        )
        jd_doc = _make_scored_doc(
            "Senior ML Engineer. Requires Python, PyTorch, LLM experience.",
            source="newpage_jd.pdf",
            score=0.88,
        )
        retriever = _make_mock_retriever([cv_doc, jd_doc])

        mock_answer = CareerAnswer(
            answer="You have strong Python and ML experience. PyTorch is a gap.",
            confidence=0.78,
            skill_gaps=[
                SkillGap(
                    skill="PyTorch",
                    importance="critical",
                    suggestion="Complete fast.ai Practical Deep Learning course",
                )
            ],
            sources=["bhavadharani_cv.pdf", "newpage_jd.pdf"],
            follow_up_questions=[
                "Have you used PyTorch in any personal projects?",
                "What LLM frameworks have you used?",
            ],
        )

        chain = CareerQAChain(retriever=retriever)
        chain._chain = MagicMock()
        chain._chain.invoke.return_value = mock_answer

        result = chain.ask("What skills am I missing for this ML engineer role?")

        assert isinstance(result, CareerAnswer)
        assert 0.0 <= result.confidence <= 1.0
        assert len(result.skill_gaps) >= 1
        assert all(g.importance in ("critical", "nice-to-have") for g in result.skill_gaps)
        assert len(result.sources) >= 1
        assert len(result.follow_up_questions) >= 1
        chain._chain.invoke.assert_called_once()
        call_kwargs = chain._chain.invoke.call_args[0][0]
        assert "question" in call_kwargs
        assert "context" in call_kwargs
        assert "Bhavadharani" in call_kwargs["context"]
        assert "PyTorch" in call_kwargs["context"]
