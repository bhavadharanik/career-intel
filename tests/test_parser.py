"""Tests for document parser — happy paths and failure paths."""

import io
import textwrap

import pytest
from pypdf import PdfWriter
from langchain_core.documents import Document

from src.ingestion.parser import parse_pdf, parse_text, _infer_doc_type


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _make_pdf_bytes(pages: list[str]) -> bytes:
    """Create a minimal in-memory PDF with the given page texts."""
    writer = PdfWriter()
    for text in pages:
        page = writer.add_blank_page(width=612, height=792)
        # pypdf's blank pages have no text by default; we use add_annotation
        # workaround: write via a stream annotation so extract_text() works.
        # Simpler: just write raw PDF bytes for the text stream.
    # Instead, build raw minimal PDF so extract_text works.
    # Use a known-good raw PDF structure.
    raw = _raw_pdf_with_text(pages)
    return raw


def _raw_pdf_with_text(pages: list[str]) -> bytes:
    """Generate a minimal valid PDF with readable text on each page."""
    objects: list[bytes] = []
    offsets: list[int] = []

    def add_obj(content: bytes) -> int:
        idx = len(objects) + 1
        objects.append(content)
        return idx

    page_stream_ids: list[tuple[int, int]] = []
    for text in pages:
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = f"BT /F1 12 Tf 50 700 Td ({escaped}) Tj ET".encode()
        length = len(stream)
        stream_obj = (
            f"<< /Length {length} >>\nstream\n".encode() + stream + b"\nendstream"
        )
        content_id = add_obj(stream_obj)
        page_id = add_obj(
            f"<< /Type /Page /MediaBox [0 0 612 792] /Contents {content_id} 0 R "
            f"/Resources << /Font << /F1 << /Type /Font /Subtype /Type1 "
            f"/BaseFont /Helvetica >> >> >> >>".encode()
        )
        page_stream_ids.append((page_id, content_id))

    pages_refs = " ".join(f"{pid} 0 R" for pid, _ in page_stream_ids)
    pages_id = add_obj(
        f"<< /Type /Pages /Kids [{pages_refs}] /Count {len(pages)} >>".encode()
    )
    for pid, _ in page_stream_ids:
        # patch parent ref — rebuild page objects with /Parent
        obj_idx = pid - 1
        existing = objects[obj_idx].decode()
        objects[obj_idx] = existing.replace(
            "<< /Type /Page", f"<< /Type /Page /Parent {pages_id} 0 R"
        ).encode()

    catalog_id = add_obj(f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode())

    body = b"%PDF-1.4\n"
    xref_offsets: list[int] = [0]  # object 0 offset placeholder
    for i, obj_bytes in enumerate(objects, start=1):
        xref_offsets.append(len(body))
        body += f"{i} 0 obj\n".encode() + obj_bytes + b"\nendobj\n"

    xref_pos = len(body)
    n = len(objects) + 1
    body += f"xref\n0 {n}\n".encode()
    body += b"0000000000 65535 f \n"
    for off in xref_offsets[1:]:
        body += f"{off:010d} 00000 n \n".encode()
    body += (
        f"trailer\n<< /Size {n} /Root {catalog_id} 0 R >>\n"
        f"startxref\n{xref_pos}\n%%EOF\n"
    ).encode()
    return body


# ─── parse_pdf ────────────────────────────────────────────────────────────────


class TestParsePdf:
    def test_happy_path_single_page(self):
        pdf_bytes = _make_pdf_bytes(["Python developer with 5 years experience"])
        docs = parse_pdf(pdf_bytes, source_name="cv.pdf")
        assert len(docs) >= 1
        assert any("Python" in d.page_content for d in docs)
        assert docs[0].metadata["source"] == "cv.pdf"
        assert docs[0].metadata["doc_type"] == "cv"

    def test_happy_path_multi_page(self):
        pdf_bytes = _make_pdf_bytes(["Page one content", "Page two content"])
        docs = parse_pdf(pdf_bytes, source_name="resume.pdf")
        assert len(docs) == 2
        assert docs[0].metadata["page"] == 1
        assert docs[1].metadata["page"] == 2

    def test_failure_path_empty_pdf_raises(self):
        """A PDF with no extractable text must raise ValueError, not return []."""
        writer = PdfWriter()
        writer.add_blank_page(width=612, height=792)
        buf = io.BytesIO()
        writer.write(buf)
        empty_pdf_bytes = buf.getvalue()

        with pytest.raises(ValueError, match="No extractable text"):
            parse_pdf(empty_pdf_bytes, source_name="blank.pdf")

    def test_failure_path_invalid_type_raises(self):
        """Passing an unsupported type must raise TypeError."""
        with pytest.raises(TypeError):
            parse_pdf(12345, source_name="bad.pdf")

    def test_metadata_doc_type_jd(self):
        pdf_bytes = _make_pdf_bytes(["We are hiring a senior engineer"])
        docs = parse_pdf(pdf_bytes, source_name="job-description-swe.pdf")
        assert docs[0].metadata["doc_type"] == "jd"


# ─── parse_text ───────────────────────────────────────────────────────────────


class TestParseText:
    def test_happy_path(self):
        docs = parse_text("Experienced ML engineer", source_name="cv.txt")
        assert len(docs) == 1
        assert "ML engineer" in docs[0].page_content
        assert docs[0].metadata["source"] == "cv.txt"

    def test_failure_path_empty_string_raises(self):
        """Empty or whitespace-only text must raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            parse_text("   ", source_name="empty.txt")

    def test_failure_path_newlines_only_raises(self):
        with pytest.raises(ValueError, match="empty"):
            parse_text("\n\n\n", source_name="blank.txt")

    def test_metadata_page_is_one(self):
        docs = parse_text("Some text", source_name="notes.txt")
        assert docs[0].metadata["page"] == 1


# ─── _infer_doc_type ─────────────────────────────────────────────────────────


class TestInferDocType:
    @pytest.mark.parametrize(
        "name,expected",
        [
            ("my_cv.pdf", "cv"),
            ("resume_v2.pdf", "cv"),
            ("curriculum_vitae.pdf", "cv"),
            ("job_description.pdf", "jd"),
            ("jd_senior_engineer.txt", "jd"),
            ("role_spec.pdf", "jd"),
            ("position_brief.pdf", "jd"),
            ("random_notes.txt", "unknown"),
        ],
    )
    def test_infer(self, name, expected):
        assert _infer_doc_type(name) == expected
