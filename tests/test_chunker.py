"""Tests for document chunker — happy paths and failure paths."""

import pytest
from langchain_core.documents import Document

from src.ingestion.chunker import chunk_documents


def _make_doc(text: str, source: str = "test.pdf", doc_type: str = "cv") -> Document:
    return Document(
        page_content=text,
        metadata={"source": source, "page": 1, "doc_type": doc_type},
    )


class TestChunkDocuments:
    def test_happy_path_single_doc_under_chunk_size(self):
        doc = _make_doc("Short text that fits in one chunk.")
        chunks = chunk_documents([doc], chunk_size=500, chunk_overlap=50)
        assert len(chunks) >= 1
        assert all(isinstance(c, Document) for c in chunks)

    def test_happy_path_long_doc_produces_multiple_chunks(self):
        long_text = " ".join(["word"] * 500)  # well over 500 chars
        doc = _make_doc(long_text)
        chunks = chunk_documents([doc], chunk_size=200, chunk_overlap=20)
        assert len(chunks) > 1

    def test_metadata_inherited_from_source(self):
        doc = _make_doc("Some content", source="jd_role.pdf", doc_type="jd")
        chunks = chunk_documents([doc], chunk_size=500, chunk_overlap=50)
        for chunk in chunks:
            assert chunk.metadata["source"] == "jd_role.pdf"
            assert chunk.metadata["doc_type"] == "jd"

    def test_chunk_index_is_sequential_per_source(self):
        long_text = "sentence. " * 200
        doc = _make_doc(long_text, source="cv.pdf")
        chunks = chunk_documents([doc], chunk_size=100, chunk_overlap=10)
        indices = [c.metadata["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_chunk_index_resets_per_source(self):
        """Each source document gets its own chunk index counter."""
        doc_a = _make_doc("alpha " * 100, source="a.pdf")
        doc_b = _make_doc("beta " * 100, source="b.pdf")
        chunks = chunk_documents([doc_a, doc_b], chunk_size=100, chunk_overlap=10)
        a_chunks = [c for c in chunks if c.metadata["source"] == "a.pdf"]
        b_chunks = [c for c in chunks if c.metadata["source"] == "b.pdf"]
        assert a_chunks[0].metadata["chunk_index"] == 0
        assert b_chunks[0].metadata["chunk_index"] == 0

    def test_failure_path_empty_list_raises(self):
        """Chunking an empty list must raise ValueError, not return []."""
        with pytest.raises(ValueError, match="empty"):
            chunk_documents([])

    def test_multiple_documents_all_chunked(self):
        docs = [
            _make_doc(f"Document {i} content with enough words here." * 10, source=f"doc_{i}.pdf")
            for i in range(3)
        ]
        chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=10)
        sources = {c.metadata["source"] for c in chunks}
        assert len(sources) == 3

    def test_chunk_size_respected(self):
        long_text = "a" * 2000
        doc = _make_doc(long_text)
        chunks = chunk_documents([doc], chunk_size=200, chunk_overlap=20)
        # All chunks should be at most chunk_size characters (with small tolerance)
        for chunk in chunks:
            assert len(chunk.page_content) <= 220  # allow slight splitter variance
