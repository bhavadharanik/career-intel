"""Pydantic models for structured LLM output.

Using .with_structured_output() means the LLM is constrained to return
valid JSON matching these schemas — no hand-parsing, no json.loads(),
no KeyError surprises at runtime.
"""

from typing import Literal

from pydantic import BaseModel, Field


class SkillGap(BaseModel):
    """A single skill the candidate is missing or needs to strengthen."""

    skill: str = Field(description="Name of the skill or competency")
    importance: Literal["critical", "nice-to-have"] = Field(
        description="How important this skill is for the target role"
    )
    suggestion: str = Field(
        description="One concrete action the candidate can take to address this gap"
    )


class CareerAnswer(BaseModel):
    """Structured response from the Career Intelligence Assistant."""

    answer: str = Field(description="Direct, specific answer to the user's question")
    confidence: float = Field(
        description=(
            "Confidence in the answer based on available context, 0.0-1.0. "
            "Use 0.0-0.4 when context is sparse, 0.5-0.7 when partial, "
            "0.8-1.0 when the context clearly supports the answer."
        ),
        ge=0.0,
        le=1.0,
    )
    skill_gaps: list[SkillGap] = Field(
        default=[],
        description=(
            "Skill gaps identified from comparing CV against job description. "
            "Only populate when the question is about fit, gaps, or readiness."
        ),
    )
    sources: list[str] = Field(
        default=[],
        description="Names of the source documents used to form this answer",
    )
    follow_up_questions: list[str] = Field(
        default=[],
        description="2-3 follow-up questions the user might want to ask next",
    )
