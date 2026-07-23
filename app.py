"""Career Intelligence Assistant — Streamlit UI."""

import tempfile
from pathlib import Path

import streamlit as st

from src.config import CHROMA_DB_PATH
from src.ingestion.chunker import chunk_documents
from src.ingestion.parser import parse_pdf, parse_text
from src.qa.chain import CareerQAChain
from src.qa.schemas import CareerAnswer
from src.rag.retriever import Retriever
from src.rag.vectorstore import VectorStore

# ─── Page config ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Career Intelligence Assistant",
    page_icon="🧭",
    layout="wide",
)

# ─── Session state initialisation ───────────────────────────────────────────


def _init_state() -> None:
    defaults = {
        "vectorstore": None,
        "chat_history": [],
        "cv_uploaded": False,
        "jd_count": 0,
        "cv_name": "",
        "jd_names": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init_state()


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _get_or_create_vectorstore() -> VectorStore:
    if st.session_state.vectorstore is None:
        st.session_state.vectorstore = VectorStore(persist_directory=CHROMA_DB_PATH)
    return st.session_state.vectorstore


def _get_or_create_chain(vs: VectorStore) -> CareerQAChain:
    # Always recreate the chain so it picks up the latest vectorstore state.
    # ChromaDB collections are fetched fresh each time — this ensures newly
    # uploaded documents are visible to the retriever immediately.
    retriever = Retriever(
        cv_store=vs.cv_store(),
        jd_store=vs.jd_store(),
    )
    return CareerQAChain(retriever=retriever)


def _ingest_cv(uploaded_file) -> None:
    vs = _get_or_create_vectorstore()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        docs = parse_pdf(tmp_path, source_name=uploaded_file.name)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    chunks = chunk_documents(docs)
    vs.add_cv_chunks(chunks)
    st.session_state.cv_uploaded = True
    st.session_state.cv_name = uploaded_file.name


def _ingest_jd(uploaded_file) -> None:
    vs = _get_or_create_vectorstore()
    name = uploaded_file.name

    if name.lower().endswith(".pdf"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        try:
            docs = parse_pdf(tmp_path, source_name=name)
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    else:
        content = uploaded_file.read().decode("utf-8", errors="replace")
        docs = parse_text(content, source_name=name)

    chunks = chunk_documents(docs)
    vs.add_jd_chunks(chunks)
    st.session_state.jd_count += 1
    if name not in st.session_state.jd_names:
        st.session_state.jd_names.append(name)


def _reset_session() -> None:
    vs = st.session_state.get("vectorstore")
    if vs is not None:
        vs.reset()
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    _init_state()
    st.rerun()


def _render_answer(answer: CareerAnswer) -> None:
    """Render a CareerAnswer in the chat UI."""
    st.markdown(answer.answer)

    col_conf, col_src = st.columns([1, 2])
    with col_conf:
        conf_pct = int(answer.confidence * 100)
        colour = "green" if conf_pct >= 70 else ("orange" if conf_pct >= 40 else "red")
        st.markdown(f"**Confidence:** :{colour}[{conf_pct}%]")
    with col_src:
        if answer.sources:
            st.markdown(f"**Sources:** {', '.join(answer.sources)}")

    if answer.skill_gaps:
        with st.expander("Skill Gaps Identified", expanded=True):
            for gap in answer.skill_gaps:
                badge = "🔴" if gap.importance == "critical" else "🟡"
                st.markdown(f"{badge} **{gap.skill}** — {gap.suggestion}")

    if answer.follow_up_questions:
        with st.expander("Suggested Follow-up Questions"):
            for q in answer.follow_up_questions:
                st.markdown(f"- {q}")


# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("Upload Documents")

    st.subheader("Your CV")
    cv_file = st.file_uploader(
        "Upload CV (PDF)",
        type=["pdf"],
        key="cv_uploader",
        help="Upload your CV or resume as a PDF.",
    )
    if cv_file is not None and not st.session_state.cv_uploaded:
        with st.spinner("Parsing and embedding CV…"):
            try:
                _ingest_cv(cv_file)
                st.success(f"CV uploaded: {cv_file.name}")
            except ValueError as e:
                st.error(str(e))

    if st.session_state.cv_uploaded:
        st.info(f"CV: {st.session_state.cv_name}")

    st.divider()

    st.subheader("Job Descriptions")
    jd_files = st.file_uploader(
        "Upload JDs (PDF or TXT)",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        key="jd_uploader",
        help="Upload one or more job descriptions.",
    )
    existing_names = set(st.session_state.jd_names)
    if jd_files:
        for jd_file in jd_files:
            if jd_file.name not in existing_names:
                with st.spinner(f"Processing {jd_file.name}…"):
                    try:
                        _ingest_jd(jd_file)
                        st.success(f"Added: {jd_file.name}")
                        existing_names.add(jd_file.name)
                    except ValueError as e:
                        st.error(str(e))

    if st.session_state.jd_names:
        st.markdown("**Loaded JDs:**")
        for name in st.session_state.jd_names:
            st.markdown(f"- {name}")

    st.divider()
    if st.button("Clear & Reset", type="secondary", use_container_width=True):
        _reset_session()

# ─── Main chat interface ──────────────────────────────────────────────────────

st.title("Career Intelligence Assistant")
st.caption(
    "Upload your CV and job descriptions in the sidebar, then ask questions about fit, "
    "skill gaps, and interview preparation."
)

# Sample questions
if not st.session_state.chat_history:
    st.markdown("**Try asking:**")
    sample_questions = [
        "What skills am I missing for this role?",
        "How does my experience align with the job description?",
        "What should I prepare for the interview?",
        "Which job is the best fit for me?",
        "What are my strongest selling points for this role?",
    ]
    cols = st.columns(len(sample_questions))
    for col, q in zip(cols, sample_questions):
        with col:
            if st.button(q, use_container_width=True):
                st.session_state.pending_question = q

# Replay chat history
for entry in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(entry["question"])
    with st.chat_message("assistant"):
        _render_answer(entry["answer"])

# Handle pending question from sample button
pending = st.session_state.pop("pending_question", None)

# Chat input
user_input = st.chat_input("Ask a question about your CV and job fit…")
question = pending or user_input

if question:
    with st.chat_message("user"):
        st.markdown(question)

    vs = _get_or_create_vectorstore()
    chain = _get_or_create_chain(vs)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                answer = chain.ask(question)
            except Exception as e:
                answer = CareerAnswer(
                    answer=f"An error occurred while processing your question: {e}",
                    confidence=0.0,
                    skill_gaps=[],
                    sources=[],
                    follow_up_questions=[],
                )
        _render_answer(answer)

    st.session_state.chat_history.append(
        {"question": question, "answer": answer}
    )
