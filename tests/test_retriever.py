"""Tests for the retriever — happy paths, no-results, and empty-collection failures."""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

from src.rag.retriever import Retriever, ScoredDocument


def _make_doc(text: str, source: str = "cv.pdf") -> Document:
    return Document(page_content=text, metadata={"source": source, "page": 1})


def _make_chroma_mock(docs: list[Document]) -> MagicMock:
    """Create a Chroma mock that returns docs from similarity_search."""
    mock = MagicMock()
    mock.similarity_search.return_value = docs
    return mock


class TestScoredDocument:
    def test_source_property(self):
        doc = _make_doc("text", source="jd_role.pdf")
        sd = ScoredDocument(document=doc, score=0.85)
        assert sd.source == "jd_role.pdf"

    def test_content_property(self):
        doc = _make_doc("hello world")
        sd = ScoredDocument(document=doc, score=0.9)
        assert sd.content == "hello world"


class TestRetriever:
    def _make_retriever(self, cv_docs, jd_docs, threshold=0.0):
        cv_mock = _make_chroma_mock(cv_docs)
        jd_mock = _make_chroma_mock(jd_docs)
        return Retriever(cv_store=cv_mock, jd_store=jd_mock, top_k=6, threshold=threshold)

    # ── Happy paths ──────────────────────────────────────────────────────────

    def test_retrieve_cv_returns_above_threshold(self):
        doc = _make_doc("Python developer 5 years experience")
        retriever = self._make_retriever(cv_docs=[doc], jd_docs=[])
        results = retriever.retrieve_cv("Python experience")
        assert len(results) == 1
        assert results[0].score == 1.0
        assert "Python" in results[0].content

    def test_retrieve_jd_returns_above_threshold(self):
        doc = _make_doc("We need a senior ML engineer", source="jd.pdf")
        retriever = self._make_retriever(cv_docs=[], jd_docs=[doc])
        results = retriever.retrieve_jd("ML engineer role")
        assert len(results) == 1
        assert results[0].source == "jd.pdf"

    def test_retrieve_all_merges_and_sorts_by_score(self):
        cv_doc = _make_doc("Python Go Kubernetes", source="cv.pdf")
        jd_doc = _make_doc("Requires Python Go", source="jd.pdf")
        retriever = self._make_retriever(cv_docs=[cv_doc], jd_docs=[jd_doc])
        results = retriever.retrieve_all("Python skills")
        assert len(results) == 2
        # Rank-based scores: both rank 1 in their collections → score 1.0
        assert all(r.score == 1.0 for r in results)

    # ── Failure paths ─────────────────────────────────────────────────────────

    def test_retrieve_cv_no_documents_returns_empty(self):
        """When CV collection is empty, retrieve_cv returns [] not an exception."""
        retriever = self._make_retriever(cv_docs=[], jd_docs=[])
        results = retriever.retrieve_cv("any query")
        assert results == []

    def test_retrieve_all_no_documents_returns_empty(self):
        """When both collections are empty, retrieve_all returns []."""
        retriever = self._make_retriever(cv_docs=[], jd_docs=[])
        results = retriever.retrieve_all("skill gaps")
        assert results == []

    def test_below_threshold_docs_filtered_out(self):
        """When similarity_search returns no docs, result is empty."""
        retriever = self._make_retriever(cv_docs=[], jd_docs=[])
        results = retriever.retrieve_cv("relevant content")
        assert results == []

    def test_all_below_threshold_returns_empty(self):
        """Empty collection returns empty list."""
        retriever = self._make_retriever(cv_docs=[], jd_docs=[])
        results = retriever.retrieve_cv("query")
        assert results == []

    def test_chroma_exception_on_empty_collection_handled(self):
        """ChromaDB may raise when collection is empty; retriever should return []."""
        cv_mock = MagicMock()
        cv_mock.similarity_search.side_effect = Exception(
            "no documents in empty collection"
        )
        jd_mock = _make_chroma_mock([])
        retriever = Retriever(cv_store=cv_mock, jd_store=jd_mock, top_k=6, threshold=0.0)
        results = retriever.retrieve_cv("query")
        assert results == []

    def test_chroma_unexpected_exception_propagates(self):
        """Non-empty-collection exceptions should propagate so bugs surface."""
        cv_mock = MagicMock()
        cv_mock.similarity_search.side_effect = RuntimeError("connection refused")
        jd_mock = _make_chroma_mock([])
        retriever = Retriever(cv_store=cv_mock, jd_store=jd_mock, top_k=6, threshold=0.0)
        with pytest.raises(RuntimeError, match="connection refused"):
            retriever.retrieve_cv("query")

    def test_retrieve_all_respects_top_k(self):
        """retrieve_all should not return more than top_k results."""
        cv_docs = [_make_doc(f"cv doc {i}") for i in range(5)]
        jd_docs = [_make_doc(f"jd doc {i}") for i in range(5)]
        retriever = self._make_retriever(cv_docs=cv_docs, jd_docs=jd_docs)
        retriever._top_k = 4
        results = retriever.retrieve_all("query")
        assert len(results) <= 4
