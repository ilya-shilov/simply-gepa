"""Optimization result models."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from .candidate import PromptCandidate
from .metrics import PromptMetrics


class OptimizationResult(BaseModel):
    """GEPA optimization result with Pareto frontier and error analysis."""

    run_id: str
    criterion_name: str
    started_at: datetime
    finished_at: datetime
    duration_seconds: float
    baseline_prompt: str
    baseline_metrics: PromptMetrics
    pareto_frontier: List[PromptCandidate]
    all_generations: List[List[PromptCandidate]]
    recommended_prompt: PromptCandidate
    error_analysis: Optional[Dict[str, Any]] = None
    dataset_size: int
    num_generations: int
    converged: bool
    convergence_generation: Optional[int] = None

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
