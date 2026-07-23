"""Document chunking using RecursiveCharacterTextSplitter."""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_documents(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """
    Split a list of Documents into smaller chunks for embedding.

    Uses RecursiveCharacterTextSplitter which splits on paragraphs → sentences
    → words → characters, preserving semantic boundaries where possible.

    Args:
        documents: Raw Documents from the parser.
        chunk_size: Target size of each chunk in characters (default 500).
        chunk_overlap: Number of characters to overlap between chunks (default 50).

    Returns:
        List of chunked Documents with inherited metadata plus a chunk index.

    Raises:
        ValueError: If the documents list is empty.
    """
    if not documents:
        raise ValueError("Cannot chunk an empty document list.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = splitter.split_documents(documents)

    # Attach a chunk index within each source document for traceability
    source_counters: dict[str, int] = {}
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        idx = source_counters.get(source, 0)
        chunk.metadata["chunk_index"] = idx
        source_counters[source] = idx + 1

    return chunks
