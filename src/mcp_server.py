"""
MCP (Model Context Protocol) server for Career Intelligence Assistant.

Exposes the RAG pipeline as MCP tools so any MCP-compatible client
(Claude Desktop, Cursor, custom agents) can query career fit, skill gaps,
and interview preparation without going through the Streamlit UI.

Usage:
    python -m src.mcp_server

Or add to Claude Desktop's mcp_servers config:
    {
      "career-intel": {
        "command": "python",
        "args": ["-m", "src.mcp_server"],
        "cwd": "/path/to/career-intel"
      }
    }
"""

import sys
from pathlib import Path

# Ensure src is importable when run as a module
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastmcp import FastMCP
from src.ingestion.chunker import chunk_documents
from src.ingestion.parser import parse_text
from src.qa.chain import CareerQAChain
from src.rag.retriever import Retriever
from src.rag.vectorstore import VectorStore

mcp = FastMCP(
    name="career-intel",
    instructions=(
        "Career Intelligence Assistant. Upload a CV and job descriptions, "
        "then ask questions about fit, skill gaps, and interview preparation."
    ),
)

# Shared vector store — persists to ./chroma_db between calls
_vs = VectorStore()


def _get_chain() -> CareerQAChain:
    retriever = Retriever(cv_store=_vs.cv_store(), jd_store=_vs.jd_store())
    return CareerQAChain(retriever=retriever)


@mcp.tool
def upload_cv(text: str) -> str:
    """
    Upload CV content as plain text. Chunks and embeds it into the CV collection.

    Args:
        text: Full CV content as a string.

    Returns:
        Confirmation message with number of chunks stored.
    """
    if not text or not text.strip():
        return "Error: CV text cannot be empty."
    docs = parse_text(text, source_name="cv.txt")
    chunks = chunk_documents(docs)
    count = _vs.add_cv_chunks(chunks)
    return f"CV uploaded successfully. {count} chunks stored and ready for questions."


@mcp.tool
def upload_job_description(text: str, job_title: str = "Job") -> str:
    """
    Upload a job description as plain text. Chunks and embeds it into the JD collection.

    Args:
        text: Full job description content as a string.
        job_title: Optional label for the job (used in source attribution).

    Returns:
        Confirmation message with number of chunks stored.
    """
    if not text or not text.strip():
        return "Error: Job description text cannot be empty."
    source_name = f"{job_title.replace(' ', '_')}.txt"
    docs = parse_text(text, source_name=source_name)
    chunks = chunk_documents(docs)
    count = _vs.add_jd_chunks(chunks)
    return f"Job description '{job_title}' uploaded successfully. {count} chunks stored."


@mcp.tool
def ask(question: str) -> dict:
    """
    Ask a question about CV-to-job fit, skill gaps, or interview preparation.

    Retrieves relevant chunks from the uploaded CV and job descriptions,
    then generates a structured answer with confidence score, skill gaps,
    sources, and follow-up questions.

    Args:
        question: Natural language question about career fit.

    Returns:
        Structured answer with answer, confidence, skill_gaps, sources,
        and follow_up_questions.
    """
    chain = _get_chain()
    result = chain.ask(question)
    return {
        "answer": result.answer,
        "confidence": result.confidence,
        "skill_gaps": [
            {"skill": g.skill, "importance": g.importance, "suggestion": g.suggestion}
            for g in result.skill_gaps
        ],
        "sources": result.sources,
        "follow_up_questions": result.follow_up_questions,
    }


@mcp.tool
def list_loaded_documents() -> dict:
    """
    Show how many CV and job description chunks are currently loaded.

    Returns:
        Dict with cv_chunks and jd_chunks counts.
    """
    return {
        "cv_chunks": _vs.cv_count(),
        "jd_chunks": _vs.jd_count(),
        "status": "ready" if _vs.cv_count() > 0 or _vs.jd_count() > 0 else "no documents loaded",
    }


@mcp.tool
def reset_documents() -> str:
    """
    Clear all uploaded CV and job description documents from the vector store.

    Returns:
        Confirmation message.
    """
    _vs.reset()
    return "All documents cleared. Upload a new CV and job description to start fresh."


if __name__ == "__main__":
    mcp.run()
