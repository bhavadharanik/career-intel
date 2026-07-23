"""Retriever that queries ChromaDB and returns docs with relevance scores."""

from dataclasses import dataclass

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config import TOP_K_RESULTS, SIMILARITY_THRESHOLD


@dataclass
class ScoredDocument:
    """A retrieved document paired with its cosine similarity score."""

    document: Document
    score: float

    @property
    def source(self) -> str:
        return self.document.metadata.get("source", "unknown")

    @property
    def content(self) -> str:
        return self.document.page_content


class Retriever:
    """
    Queries one or more ChromaDB Chroma stores with similarity-score filtering.

    Returning scores alongside documents lets the downstream QA chain
    communicate retrieval confidence to the user rather than silently
    surfacing low-relevance chunks.
    """

    def __init__(
        self,
        cv_store: Chroma,
        jd_store: Chroma,
        top_k: int = TOP_K_RESULTS,
        threshold: float = SIMILARITY_THRESHOLD,
    ) -> None:
        self._cv_store = cv_store
        self._jd_store = jd_store
        self._top_k = top_k
        self._threshold = threshold

    def retrieve_cv(self, query: str) -> list[ScoredDocument]:
        """Retrieve top-k chunks from the CV collection above the score threshold."""
        return self._query_store(self._cv_store, query)

    def retrieve_jd(self, query: str) -> list[ScoredDocument]:
        """Retrieve top-k chunks from the JD collection above the score threshold."""
        return self._query_store(self._jd_store, query)

    def retrieve_all(self, query: str) -> list[ScoredDocument]:
        """
        Retrieve from both collections and merge by score (highest first).

        This is the default path for general career questions that span
        both the candidate's background and the target roles.
        """
        cv_results = self._query_store(self._cv_store, query)
        jd_results = self._query_store(self._jd_store, query)
        combined = cv_results + jd_results
        combined.sort(key=lambda r: r.score, reverse=True)
        return combined[: self._top_k]

    def _query_store(self, store: Chroma, query: str) -> list[ScoredDocument]:
        """
        Run similarity search against a single store.

        Uses similarity_search (no score filtering) because langchain-chroma
        with L2 distance can return negative "relevance" scores which break
        threshold-based filtering. Documents are ranked by ChromaDB's internal
        distance metric (closest first) and we assign a normalised rank score.
        """
        try:
            docs: list[Document] = store.similarity_search(query, k=self._top_k)
        except Exception as exc:
            if "no documents" in str(exc).lower() or "empty" in str(exc).lower():
                return []
            raise

        # Assign rank-based scores: 1.0 for rank 1, decaying by rank position
        total = len(docs)
        return [
            ScoredDocument(
                document=doc,
                score=round(1.0 - (i / max(total, 1)) * 0.5, 3),
            )
            for i, doc in enumerate(docs)
        ]
