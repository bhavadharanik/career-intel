"""ChromaDB vector store with disk persistence."""

import uuid
from typing import Optional

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from src.config import (
    CHROMA_DB_PATH,
    CV_COLLECTION,
    JD_COLLECTION,
    EMBEDDING_MODEL,
    OPENAI_API_KEY,
)


def _build_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY,
    )


class VectorStore:
    """
    Manages two persistent ChromaDB collections — one for CV chunks,
    one for job description chunks — with disk persistence so embeddings
    survive application restarts without recomputation.

    CV and JD documents are stored separately so that retrieval can be
    scoped to one collection or queried across both, depending on the
    question type.
    """

    def __init__(self, persist_directory: str = CHROMA_DB_PATH) -> None:
        self._persist_dir = persist_directory
        self._embeddings = _build_embeddings()
        self._cv_store: Optional[Chroma] = None
        self._jd_store: Optional[Chroma] = None
        self._init_stores()

    def _init_stores(self) -> None:
        """Initialise (or load from disk) both collections."""
        self._cv_store = Chroma(
            collection_name=CV_COLLECTION,
            embedding_function=self._embeddings,
            persist_directory=self._persist_dir,
        )
        self._jd_store = Chroma(
            collection_name=JD_COLLECTION,
            embedding_function=self._embeddings,
            persist_directory=self._persist_dir,
        )

    def add_cv_chunks(self, chunks: list[Document]) -> int:
        """
        Add CV chunks to the CV collection.

        Returns:
            Number of chunks added.
        """
        if not chunks:
            raise ValueError("Cannot add an empty list of CV chunks.")
        ids = [str(uuid.uuid4()) for _ in chunks]
        self._cv_store.add_documents(chunks, ids=ids)
        return len(chunks)

    def add_jd_chunks(self, chunks: list[Document]) -> int:
        """
        Add job description chunks to the JD collection.

        Returns:
            Number of chunks added.
        """
        if not chunks:
            raise ValueError("Cannot add an empty list of JD chunks.")
        ids = [str(uuid.uuid4()) for _ in chunks]
        self._jd_store.add_documents(chunks, ids=ids)
        return len(chunks)

    def cv_store(self) -> Chroma:
        """Return the raw Chroma CV store (for retriever use)."""
        return self._cv_store

    def jd_store(self) -> Chroma:
        """Return the raw Chroma JD store (for retriever use)."""
        return self._jd_store

    def reset(self) -> None:
        """
        Clear all documents from both collections.
        Used between user sessions or during testing.
        """
        self._cv_store.delete_collection()
        self._jd_store.delete_collection()
        self._init_stores()

    def cv_count(self) -> int:
        """Return the number of CV chunks currently stored."""
        return self._cv_store._collection.count()

    def jd_count(self) -> int:
        """Return the number of JD chunks currently stored."""
        return self._jd_store._collection.count()
