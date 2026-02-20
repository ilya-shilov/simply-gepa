"""Pareto selection for multi-objective optimization."""

from typing import List

from loguru import logger

from ..models import OptimizationConfig, PromptCandidate


class ParetoSelector:
    """Pareto frontier selection for prompt candidates."""

    def __init__(self, config: OptimizationConfig):
        """Initialize Pareto selector with weighting config."""
        self.config = config

    def get_pareto_frontier(self, candidates: List[PromptCandidate]) -> List[PromptCandidate]:
        """Extract non-dominated candidates from population."""
        if not candidates:
            return []

        pareto_frontier: List[PromptCandidate] = []

        for candidate in candidates:
            is_dominated = False

            for other in candidates:
                if other.id == candidate.id:
                    continue

                if other.dominates(candidate):
                    is_dominated = True
                    break

            if not is_dominated:
                pareto_frontier.append(candidate)

        logger.debug(
            f"Pareto frontier: {len(pareto_frontier)} / {len(candidates)} candidates"
        )

        return pareto_frontier

    def select_population(
        self,
        candidates: List[PromptCandidate],
        population_size: int
    ) -> List[PromptCandidate]:
        """Select next generation population using Pareto frontier and diversity."""
        frontier = self.get_pareto_frontier(candidates)

        if len(frontier) >= population_size:
            sorted_frontier = sorted(
                frontier,
                key=lambda c: (
                    -c.metrics.accuracy,
                    c.metrics.false_negative_rate,
                    c.metrics.cost_tokens
                )
            )
            return sorted_frontier[:population_size]

        selected = frontier.copy()
        remaining = [c for c in candidates if c not in frontier]

        remaining_sorted = sorted(
            remaining,
            key=lambda c: (
                -c.metrics.accuracy,
                c.metrics.false_negative_rate,
                c.metrics.cost_tokens
            )
        )

        needed = population_size - len(selected)
        selected.extend(remaining_sorted[:needed])

        logger.debug(
            f"Selected population: {len(frontier)} Pareto + "
            f"{len(selected) - len(frontier)} diverse = {len(selected)} total"
        )

        return selected

    def get_recommended_candidate(self, frontier: List[PromptCandidate]) -> PromptCandidate:
        """Select best-balanced candidate from Pareto frontier."""
        if not frontier:
            raise ValueError("Cannot recommend from empty frontier")

        def score(c: PromptCandidate) -> float:
            return (
                self.config.priority_accuracy * c.metrics.accuracy
                - self.config.priority_fn_reduction * c.metrics.false_negative_rate
                - self.config.priority_fp_reduction * c.metrics.false_positive_rate
                - self.config.priority_cost_reduction * c.metrics.cost_tokens
            )

        best = max(frontier, key=score)

        logger.info(f"Recommended candidate: {best.id} with score={score(best):.3f}")

        return best
