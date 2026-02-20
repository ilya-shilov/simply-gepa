"""Prompt quality metrics for multi-objective optimization."""

from pydantic import BaseModel, Field


class PromptMetrics(BaseModel):
    """Prompt quality metrics for multi-objective optimization."""

    accuracy: float = Field(ge=0.0, le=1.0, description="Correct answers ratio")
    false_negative_rate: float = Field(ge=0.0, le=1.0, description="Missed errors ratio")
    false_positive_rate: float = Field(ge=0.0, le=1.0, description="False alarms ratio")
    cost_tokens: int = Field(ge=0, description="Prompt length in tokens")
    latency_ms: float = Field(ge=0.0, description="Average response time")
    total_examples: int
    correct: int
    false_negatives: int
    false_positives: int
    not_enough_count: int

    def __str__(self) -> str:
        return (
            f"Acc={self.accuracy:.2%}, FN={self.false_negative_rate:.2%}, "
            f"FP={self.false_positive_rate:.2%}, Cost={self.cost_tokens}tok"
        )
