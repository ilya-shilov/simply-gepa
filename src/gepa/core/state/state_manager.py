"""Optimization state manager."""

import json
from pathlib import Path
from typing import List

from ...models import PromptCandidate, PromptMetrics, OptimizationConfig


class OptimizerStateManager:
    """Persist and restore optimization state."""

    def __init__(self, config: OptimizationConfig):
        """Initialize state manager."""
        self.config = config

    def resolve_state_path(self, resume_from: str) -> Path:
        """Resolve resume state path."""
        path = Path(resume_from)
        if path.is_dir():
            return path / "state.json"
        return path

    def get_runs_dir(self) -> Path:
        """Get runs directory for state persistence."""
        if getattr(self.config, "runs_dir", None):
            return Path(self.config.runs_dir)
        return Path.cwd() / "data" / "gepa" / "runs"

    def save_state(
        self,
        run_id: str,
        generation: int,
        population: List[PromptCandidate],
        baseline_prompt: str,
        baseline_metrics: PromptMetrics,
        all_generations: List[List[PromptCandidate]]
    ) -> None:
        """Save optimization state for resume."""
        run_dir = self.get_runs_dir() / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        state_path = run_dir / "state.json"
        state = {
            "run_id": run_id,
            "generation": generation,
            "baseline_prompt": baseline_prompt,
            "baseline_metrics": baseline_metrics.model_dump(),
            "population": [candidate.model_dump() for candidate in population],
            "all_generations": [
                [candidate.model_dump() for candidate in generation_candidates]
                for generation_candidates in all_generations
            ]
        }
        state_path.write_text(json.dumps(state), encoding="utf-8")

    def load_state(self, state_path: Path) -> dict:
        """Load optimization state for resume."""
        state = json.loads(state_path.read_text(encoding="utf-8"))
        baseline_metrics = PromptMetrics.model_validate(state["baseline_metrics"])
        population = [
            PromptCandidate.model_validate(item)
            for item in state["population"]
        ]
        all_generations = [
            [PromptCandidate.model_validate(item) for item in generation_candidates]
            for generation_candidates in state.get("all_generations", [])
        ]
        return {
            "run_id": state["run_id"],
            "generation": state["generation"],
            "baseline_prompt": state["baseline_prompt"],
            "baseline_metrics": baseline_metrics,
            "population": population,
            "all_generations": all_generations
        }
