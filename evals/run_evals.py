"""
Evaluation runner for Career Intelligence Assistant.

Tests actual LLM behaviour using semantic similarity scoring — not keyword matching.
Results are logged to LangSmith when LANGCHAIN_TRACING_V2=true.

Usage:
    python evals/run_evals.py
    python evals/run_evals.py --verbose
"""

import argparse
import json
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from langsmith import Client
from langsmith.schemas import Run

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import LANGCHAIN_PROJECT, LANGCHAIN_TRACING_V2
from src.ingestion.chunker import chunk_documents
from src.ingestion.parser import parse_text
from src.qa.chain import CareerQAChain
from src.qa.schemas import CareerAnswer
from src.rag.retriever import Retriever
from src.rag.vectorstore import VectorStore

DATASET_PATH = Path(__file__).parent / "dataset.json"
SIMILARITY_PASS_THRESHOLD = 0.6  # cosine similarity for semantic checks


# ─── Semantic similarity scorer ──────────────────────────────────────────────


def _embed(texts: list[str]) -> list[list[float]]:
    """Embed texts using OpenAI text-embedding-3-small."""
    from openai import OpenAI
    from src.config import OPENAI_API_KEY

    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return [item.embedding for item in response.data]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    import math

    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _semantic_contains(answer_text: str, expected_terms: list[str]) -> dict[str, float]:
    """
    Check whether the answer semantically covers each expected term.
    Returns a dict of term -> cosine similarity score.
    Uses embedding similarity rather than exact string match.
    """
    if not expected_terms:
        return {}

    texts = [answer_text] + expected_terms
    embeddings = _embed(texts)
    answer_emb = embeddings[0]
    term_embs = embeddings[1:]

    return {
        term: _cosine_similarity(answer_emb, term_emb)
        for term, term_emb in zip(expected_terms, term_embs)
    }


# ─── LLM-as-judge ────────────────────────────────────────────────────────────


def _llm_judge_groundedness(answer: str, context: str) -> dict[str, Any]:
    """
    LLM-as-judge: direct assessment of whether the answer is grounded in context.

    Scores 0.0-1.0. A grounded answer only makes claims supported by the
    retrieved chunks — it does not hallucinate facts not present in context.
    """
    from openai import OpenAI
    from src.config import OPENAI_API_KEY

    client = OpenAI(api_key=OPENAI_API_KEY)

    prompt = f"""You are an expert evaluator assessing whether an AI answer is grounded in the provided context.

CONTEXT (retrieved document chunks):
{context}

ANSWER TO EVALUATE:
{answer}

Score the answer on GROUNDEDNESS: does every factual claim in the answer come from the context above?

Respond in this exact JSON format:
{{
  "score": <float between 0.0 and 1.0>,
  "reasoning": "<one sentence explanation>",
  "hallucinated_claims": ["<any claims not supported by context>"]
}}

Score guide:
1.0 = every claim is directly supported by the context
0.7 = most claims supported, minor extrapolation
0.5 = mix of grounded and ungrounded claims
0.3 = significant claims not in context
0.0 = answer ignores context entirely"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )

    import json as _json
    parsed = _json.loads(response.choices[0].message.content)
    return {
        "score": float(parsed.get("score", 0.0)),
        "reasoning": parsed.get("reasoning", ""),
        "hallucinated_claims": parsed.get("hallucinated_claims", []),
        "passed": float(parsed.get("score", 0.0)) >= 0.7,
    }


def _llm_judge_pairwise(question: str, answer_a: str, answer_b: str) -> dict[str, Any]:
    """
    LLM-as-judge: pairwise comparison of two answers to the same question.

    Returns which answer is better (A, B, or tie) and why.
    Used to compare prompt variants or retrieval strategies.
    """
    from openai import OpenAI
    from src.config import OPENAI_API_KEY

    client = OpenAI(api_key=OPENAI_API_KEY)

    prompt = f"""You are an expert evaluator comparing two AI answers to the same career question.

QUESTION: {question}

ANSWER A:
{answer_a}

ANSWER B:
{answer_b}

Which answer is better for a candidate trying to understand their career fit?
Consider: accuracy, specificity, actionability, and clarity.

Respond in this exact JSON format:
{{
  "winner": "<A, B, or tie>",
  "reasoning": "<two sentence explanation>",
  "scores": {{"A": <0.0-1.0>, "B": <0.0-1.0>}}
}}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )

    import json as _json
    parsed = _json.loads(response.choices[0].message.content)
    return {
        "winner": parsed.get("winner", "tie"),
        "reasoning": parsed.get("reasoning", ""),
        "scores": parsed.get("scores", {}),
    }


# ─── Per-eval-case runner ─────────────────────────────────────────────────────


def _run_case(case: dict[str, Any], verbose: bool) -> dict[str, Any]:
    """Run a single evaluation case and return a result dict."""
    eval_id = case["id"]
    check_type = case["check_type"]
    cv_text = case.get("cv_text", "")
    jd_text = case.get("jd_text", "")

    # Build an ephemeral in-memory VectorStore for this eval case
    with tempfile.TemporaryDirectory() as tmpdir:
        vs = VectorStore(persist_directory=tmpdir)

        if cv_text.strip():
            cv_docs = parse_text(cv_text, source_name="eval_cv.txt")
            cv_chunks = chunk_documents(cv_docs)
            vs.add_cv_chunks(cv_chunks)

        if jd_text.strip():
            jd_docs = parse_text(jd_text, source_name="eval_jd.txt")
            jd_chunks = chunk_documents(jd_docs)
            vs.add_jd_chunks(jd_chunks)

        retriever = Retriever(cv_store=vs.cv_store(), jd_store=vs.jd_store())
        chain = CareerQAChain(retriever=retriever)

        start = time.time()
        answer: CareerAnswer = chain.ask(case["question"])
        latency_ms = int((time.time() - start) * 1000)

        # Retrieve context separately for groundedness checks
        scored_docs = retriever.retrieve_all(case["question"])
        retrieved_context = "\n\n---\n\n".join(
            f"[{i+1}] {sd.content}" for i, sd in enumerate(scored_docs)
        )

    result: dict[str, Any] = {
        "id": eval_id,
        "description": case["description"],
        "question": case["question"],
        "check_type": check_type,
        "latency_ms": latency_ms,
        "answer_preview": answer.answer[:200],
        "confidence": answer.confidence,
        "skill_gaps": [g.skill for g in answer.skill_gaps],
        "sources": answer.sources,
        "follow_up_questions": answer.follow_up_questions,
        "passed": False,
        "checks": {},
    }

    # ── Confidence range check (all cases) ───────────────────────────────────
    conf_low, conf_high = case["expected_confidence_range"]
    conf_ok = conf_low <= answer.confidence <= conf_high
    result["checks"]["confidence_in_range"] = {
        "expected": f"{conf_low}-{conf_high}",
        "actual": answer.confidence,
        "passed": conf_ok,
    }

    # ── Type-specific checks ──────────────────────────────────────────────────

    if check_type == "no_context":
        result["passed"] = conf_ok
        return result

    if check_type in ("skill_gaps", "readiness"):
        expected_gaps = case.get("expected_gaps", [])
        if expected_gaps:
            gap_text = answer.answer
            if answer.skill_gaps:
                gap_text += " " + " ".join(g.skill for g in answer.skill_gaps)
            scores = _semantic_contains(gap_text, expected_gaps)
            gaps_covered = sum(1 for s in scores.values() if s >= SIMILARITY_PASS_THRESHOLD)
            coverage_ratio = gaps_covered / len(expected_gaps)
            result["checks"]["skill_gap_coverage"] = {
                "expected_terms": expected_gaps,
                "similarity_scores": {k: round(v, 3) for k, v in scores.items()},
                "coverage_ratio": round(coverage_ratio, 2),
                "passed": coverage_ratio >= 0.5,  # at least half the gaps mentioned
            }
            result["passed"] = conf_ok and coverage_ratio >= 0.5
        else:
            result["passed"] = conf_ok
        return result

    if check_type in ("alignment", "advice", "recommendation", "strengths"):
        expected_keywords = case.get("expected_keywords", [])
        if expected_keywords:
            scores = _semantic_contains(answer.answer, expected_keywords)
            keywords_covered = sum(1 for s in scores.values() if s >= SIMILARITY_PASS_THRESHOLD)
            coverage_ratio = keywords_covered / len(expected_keywords)
            result["checks"]["keyword_coverage"] = {
                "expected_terms": expected_keywords,
                "similarity_scores": {k: round(v, 3) for k, v in scores.items()},
                "coverage_ratio": round(coverage_ratio, 2),
                "passed": coverage_ratio >= 0.5,
            }
            result["passed"] = conf_ok and coverage_ratio >= 0.5
        else:
            result["passed"] = conf_ok
        return result

    if check_type == "follow_ups_non_trivial":
        has_follow_ups = len(answer.follow_up_questions) >= 2
        # Each follow-up should be semantically distinct from a generic placeholder
        generic = "What else would you like to know?"
        if has_follow_ups and answer.follow_up_questions:
            scores = _embed(answer.follow_up_questions + [generic])
            follow_up_embs = scores[:-1]
            generic_emb = scores[-1]
            non_generic = all(
                _cosine_similarity(f, generic_emb) < 0.95
                for f in follow_up_embs
            )
            result["checks"]["follow_ups_non_trivial"] = {
                "count": len(answer.follow_up_questions),
                "non_generic": non_generic,
                "passed": non_generic and has_follow_ups,
            }
            result["passed"] = conf_ok and non_generic and has_follow_ups
        else:
            result["checks"]["follow_ups_non_trivial"] = {
                "count": len(answer.follow_up_questions),
                "passed": False,
            }
            result["passed"] = False
        return result

    if check_type == "groundedness":
        if retrieved_context.strip():
            judge_result = _llm_judge_groundedness(answer.answer, retrieved_context)
            result["checks"]["groundedness"] = judge_result
            result["passed"] = conf_ok and judge_result["passed"]
        else:
            result["checks"]["groundedness"] = {"passed": False, "reasoning": "No context retrieved"}
            result["passed"] = False
        return result

    if check_type == "pairwise":
        # Run a second answer with a simpler prompt variant for comparison
        variant_question = case.get("variant_question", case["question"])
        answer_b: CareerAnswer = chain.ask(variant_question)
        pairwise_result = _llm_judge_pairwise(
            question=case["question"],
            answer_a=answer.answer,
            answer_b=answer_b.answer,
        )
        result["checks"]["pairwise"] = pairwise_result
        result["answer_b_preview"] = answer_b.answer[:200]
        # Pass if answer A wins or ties
        result["passed"] = conf_ok and pairwise_result["winner"] in ("A", "tie")
        return result

    # Default: only confidence check
    result["passed"] = conf_ok
    return result


# ─── LangSmith logging ────────────────────────────────────────────────────────


def _log_to_langsmith(results: list[dict[str, Any]]) -> None:
    """Log eval results to LangSmith as a dataset run."""
    if not LANGCHAIN_TRACING_V2:
        print("LangSmith tracing disabled — skipping upload.")
        return

    try:
        client = Client()
        for res in results:
            client.create_run(
                name=f"eval_{res['id']}",
                run_type="chain",
                inputs={"question": res["question"]},
                outputs={
                    "answer": res["answer_preview"],
                    "confidence": res["confidence"],
                    "skill_gaps": res["skill_gaps"],
                    "passed": res["passed"],
                    "checks": res["checks"],
                },
                project_name=LANGCHAIN_PROJECT,
                extra={"check_type": res["check_type"], "latency_ms": res["latency_ms"]},
            )
        print(f"Logged {len(results)} eval results to LangSmith project '{LANGCHAIN_PROJECT}'.")
    except Exception as e:
        print(f"LangSmith logging failed (non-fatal): {e}")


# ─── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Career Intelligence Assistant evals")
    parser.add_argument("--verbose", action="store_true", help="Print full result details")
    args = parser.parse_args()

    with open(DATASET_PATH) as f:
        dataset = json.load(f)

    print(f"Running {len(dataset)} eval cases…\n")
    results: list[dict[str, Any]] = []

    for case in dataset:
        print(f"  [{case['id']}] {case['description'][:60]}…", end=" ", flush=True)
        try:
            result = _run_case(case, verbose=args.verbose)
            status = "PASS" if result["passed"] else "FAIL"
            print(f"{status} (confidence={result['confidence']:.2f}, {result['latency_ms']}ms)")
            results.append(result)
        except Exception as e:
            print(f"ERROR: {e}")
            results.append(
                {
                    "id": case["id"],
                    "description": case["description"],
                    "passed": False,
                    "error": str(e),
                    "checks": {},
                }
            )

        if args.verbose and results:
            last = results[-1]
            print(f"    Answer: {last.get('answer_preview', '')[:100]}")
            for check_name, check_data in last.get("checks", {}).items():
                print(f"    {check_name}: {check_data}")
            print()

    passed = sum(1 for r in results if r.get("passed"))
    total = len(results)
    print(f"\nResults: {passed}/{total} passed ({100 * passed // total}%)")

    _log_to_langsmith(results)

    # Save results JSON for CI
    output_path = Path(__file__).parent / "eval_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Full results saved to {output_path}")

    if passed < total * 0.75:
        print("WARN: Less than 75% of evals passed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
