"""QA chain using structured output — no hand-parsed JSON."""

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.config import LLM_MODEL, LLM_TEMPERATURE, OPENAI_API_KEY
from src.qa.schemas import CareerAnswer
from src.rag.retriever import Retriever, ScoredDocument

_SYSTEM_PROMPT = """\
You are a Career Intelligence Assistant. Your job is to help candidates understand
how well their CV matches a job description, identify skill gaps, and prepare for
interviews.

SCOPE GUARDRAIL: Only answer questions about the uploaded CV and job descriptions.
If the question is off-topic (e.g. general knowledge, coding help, weather, anything
unrelated to career fit or the uploaded documents), respond with:
answer="I can only answer questions about your CV and job descriptions. Please upload
your documents and ask something like: What skills am I missing? How does my experience
align with this role?", confidence=0.0, skill_gaps=[], sources=[], follow_up_questions=[].

You MUST respond with a structured answer that includes:
- A direct, honest answer to the question
- A confidence score reflecting how well the provided context supports your answer
- Relevant skill gaps (only when the question is about fit, gaps, or readiness)
- The source documents you used
- 2-3 useful follow-up questions

If the context does not contain enough information to answer confidently, say so
clearly in the answer field and set a low confidence score (< 0.4). Do NOT
fabricate information not present in the context.

Context from uploaded documents:
{context}
"""

_HUMAN_PROMPT = "{question}"


class CareerQAChain:
    """
    RAG chain that retrieves relevant document chunks then calls GPT-4o-mini
    with structured output constraints via .with_structured_output(CareerAnswer).

    This guarantees the LLM response always deserialises into a CareerAnswer
    object — removing the manual JSON parsing that caused the Focused rejection.
    """

    def __init__(self, retriever: Retriever) -> None:
        self._retriever = retriever
        self._llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            openai_api_key=OPENAI_API_KEY,
        ).with_structured_output(CareerAnswer)

        self._prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _SYSTEM_PROMPT),
                ("human", _HUMAN_PROMPT),
            ]
        )

        self._chain = self._prompt | self._llm

    def ask(self, question: str) -> CareerAnswer:
        """
        Run the full RAG pipeline for a user question.

        Steps:
        1. Retrieve relevant chunks from CV and JD collections.
        2. Format chunks into a context string with source attribution.
        3. Call the LLM with structured output constraints.
        4. Return a fully validated CareerAnswer (Pydantic guarantees types).

        Args:
            question: Natural language question from the user.

        Returns:
            CareerAnswer with answer, confidence, skill_gaps, sources,
            and follow_up_questions populated.

        Raises:
            ValueError: If question is empty.
            RuntimeError: If no documents have been uploaded yet.
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")
        if len(question) > 1000:
            raise ValueError("Question is too long. Please keep it under 1000 characters.")

        scored_docs = self._retriever.retrieve_all(question)

        if not scored_docs:
            # Return a graceful structured response rather than an error
            # so the UI can display it properly
            return CareerAnswer(
                answer=(
                    "I don't have enough context to answer this question. "
                    "Please upload your CV and at least one job description first."
                ),
                confidence=0.0,
                skill_gaps=[],
                sources=[],
                follow_up_questions=[
                    "Could you upload your CV so I can analyse it?",
                    "Could you upload a job description for the role you're targeting?",
                ],
            )

        context = _format_context(scored_docs)
        unique_sources = _unique_sources(scored_docs)

        result: CareerAnswer = self._chain.invoke(
            {"context": context, "question": question}
        )

        # Ensure sources list reflects what was actually retrieved
        if not result.sources:
            result.sources = unique_sources

        return result


def _format_context(scored_docs: list[ScoredDocument]) -> str:
    """
    Format retrieved chunks into a numbered context block with source labels.

    Including the source name and relevance score lets the LLM cite its
    evidence and communicate uncertainty proportionally.
    """
    parts: list[str] = []
    for i, sd in enumerate(scored_docs, start=1):
        parts.append(
            f"[{i}] Source: {sd.source} (relevance: {sd.score:.2f})\n{sd.content}"
        )
    return "\n\n---\n\n".join(parts)


def _unique_sources(scored_docs: list[ScoredDocument]) -> list[str]:
    """Return deduplicated list of source names in retrieval order."""
    seen: set[str] = set()
    sources: list[str] = []
    for sd in scored_docs:
        if sd.source not in seen:
            sources.append(sd.source)
            seen.add(sd.source)
    return sources
