"""Prompt candidate model for evolutionary optimization."""

from typing import Optional

from pydantic import BaseModel, Field

from .metrics import PromptMetrics


class PromptCandidate(BaseModel):
    """Prompt candidate in evolutionary population."""

    id: str
    text: str = Field(description="Prompt text")
    metrics: PromptMetrics
    generation: int = Field(ge=0, description="Generation number")
    parent_id: Optional[str] = None
    mutation_notes: Optional[str] = None

    def dominates(self, other: "PromptCandidate") -> bool:
        """Check if this candidate Pareto-dominates another candidate."""
        better_or_equal = (
            self.metrics.accuracy >= other.metrics.accuracy and
            self.metrics.false_negative_rate <= other.metrics.false_negative_rate and
            self.metrics.false_positive_rate <= other.metrics.false_positive_rate and
            self.metrics.cost_tokens <= other.metrics.cost_tokens
        )

        strictly_better = (
            self.metrics.accuracy > other.metrics.accuracy or
            self.metrics.false_negative_rate < other.metrics.false_negative_rate or
            self.metrics.false_positive_rate < other.metrics.false_positive_rate or
            self.metrics.cost_tokens < other.metrics.cost_tokens
        )

        return better_or_equal and strictly_better

    def __str__(self) -> str:
        return f"Candidate(gen={self.generation}, {self.metrics})"
