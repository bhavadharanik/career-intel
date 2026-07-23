"""Document parsing for PDF and plain text files."""

import io
from pathlib import Path
from typing import Union

from pypdf import PdfReader
from langchain_core.documents import Document


def parse_pdf(source: Union[str, bytes, Path], source_name: str = "cv.pdf") -> list[Document]:
    """
    Parse a PDF file or bytes into LangChain Documents.

    Args:
        source: File path (str/Path) or raw PDF bytes.
        source_name: Name to attach as document metadata.

    Returns:
        List of Documents, one per page.

    Raises:
        ValueError: If the PDF has no extractable text.
    """
    if isinstance(source, (str, Path)):
        reader = PdfReader(str(source))
    elif isinstance(source, (bytes, bytearray)):
        reader = PdfReader(io.BytesIO(source))
    else:
        raise TypeError(f"Unsupported source type: {type(source)}")

    docs: list[Document] = []
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": source_name,
                        "page": page_num + 1,
                        "doc_type": _infer_doc_type(source_name),
                    },
                )
            )

    if not docs:
        raise ValueError(
            f"No extractable text found in '{source_name}'. "
            "The PDF may be scanned or image-based."
        )

    return docs


def parse_text(text: str, source_name: str = "document.txt") -> list[Document]:
    """
    Wrap plain text in a LangChain Document.

    Args:
        text: Raw text content.
        source_name: Name to attach as document metadata.

    Returns:
        List containing a single Document.

    Raises:
        ValueError: If text is empty or whitespace-only.
    """
    cleaned = text.strip()
    if not cleaned:
        raise ValueError(f"Text content is empty for source '{source_name}'.")

    return [
        Document(
            page_content=cleaned,
            metadata={
                "source": source_name,
                "page": 1,
                "doc_type": _infer_doc_type(source_name),
            },
        )
    ]


def _infer_doc_type(source_name: str) -> str:
    """Infer whether a document is a CV or job description from its name."""
    lower = source_name.lower()
    if any(kw in lower for kw in ("cv", "resume", "curriculum")):
        return "cv"
    if any(kw in lower for kw in ("jd", "job", "description", "role", "position")):
        return "jd"
    return "unknown"
