"""Prompt quality metrics for multi-objective optimization."""

from pydantic import BaseModel, Field


class PromptMetrics(BaseModel):
    """Prompt quality metrics for multi-objective optimization."""

    accuracy: float = Field(ge=0.0, le=1.0, description="Correct answers ratio")
    macro_f1: float = Field(default=0.0, ge=0.0, le=1.0, description="Macro-averaged F1 across all classes")
    false_negative_rate: float = Field(ge=0.0, le=1.0, description="Missed errors ratio")
    false_positive_rate: float = Field(ge=0.0, le=1.0, description="False alarms ratio")
    cost_tokens: int = Field(ge=0, description="Prompt length in tokens")
    latency_ms: float = Field(ge=0.0, description="Average response time")
    total_examples: int
    correct: int
    true_positives: int = 0
    true_negatives: int = 0
    false_negatives: int
    false_positives: int
    not_enough_count: int

    def __str__(self) -> str:
        return (
            f"Acc={self.accuracy:.2%}, MacroF1={self.macro_f1:.2%}, "
            f"FN={self.false_negative_rate:.2%}, FP={self.false_positive_rate:.2%}, "
            f"Cost={self.cost_tokens}tok"
        )
