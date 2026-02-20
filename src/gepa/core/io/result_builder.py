"""Optimization result builder."""

import time
from datetime import datetime
from typing import List

from loguru import logger
from rich.console import Console

from ...analysis import ErrorAnalyzer
from ...models import OptimizationConfig, OptimizationResult, PromptCandidate, PromptMetrics
from ..pareto import ParetoSelector


class ResultBuilder:
    """Build and log optimization results."""

    def __init__(
        self,
        config: OptimizationConfig,
        pareto_selector: ParetoSelector,
        error_analyzer: ErrorAnalyzer,
        console: Console
    ):
        """Initialize result builder."""
        self.config = config
        self.pareto_selector = pareto_selector
        self.error_analyzer = error_analyzer
        self.console = console

    def build(
        self,
        run_id: str,
        baseline_prompt: str,
        baseline_metrics: PromptMetrics,
        population: List[PromptCandidate],
        all_generations: List[List[PromptCandidate]],
        dataset_size: int,
        start_time: float
    ) -> OptimizationResult:
        """Build optimization result."""
        pareto_frontier = self.pareto_selector.get_pareto_frontier(population)
        recommended = self.pareto_selector.get_recommended_candidate(pareto_frontier)
        error_analysis = self._analyze_errors(recommended, self.config.dataset_path)
        elapsed = time.time() - start_time
        return OptimizationResult(
            run_id=run_id,
            criterion_name=self.config.criterion_name,
            started_at=datetime.fromtimestamp(start_time),
            finished_at=datetime.now(),
            duration_seconds=elapsed,
            baseline_prompt=baseline_prompt,
            baseline_metrics=baseline_metrics,
            pareto_frontier=pareto_frontier,
            all_generations=all_generations,
            recommended_prompt=recommended,
            error_analysis=error_analysis,
            dataset_size=dataset_size,
            num_generations=len(all_generations),
            converged=len(all_generations) < self.config.num_generations,
            convergence_generation=len(all_generations) if len(all_generations) < self.config.num_generations else None
        )

    def log_result(self, result: OptimizationResult, evaluation_cache_size: int) -> None:
        """Log optimization result."""
        logger.success(f"Optimization complete in {result.duration_seconds:.1f}s")
        logger.info(f"Cache statistics: {evaluation_cache_size} unique prompts cached")
        self._print_results(result, evaluation_cache_size)

    def _analyze_errors(self, recommended: PromptCandidate, dataset_path: str) -> dict:
        """Analyze errors for recommended candidate."""
        self.console.print("\n[bold yellow]Analyzing errors...[/bold yellow]")
        return self.error_analyzer.analyze_errors(
            candidate=recommended,
            dataset_path=dataset_path,
            criterion_name=self.config.criterion_name
        )

    def _print_results(self, result: OptimizationResult, cache_size: int) -> None:
        """Print optimization results."""
        self.console.print("\n[bold green]+----------------------------------------------+[/bold green]")
        self.console.print("[bold green]|       GEPA Optimization Results              |[/bold green]")
        self.console.print("[bold green]+----------------------------------------------+[/bold green]\n")

        self.console.print(f"Run ID: [cyan]{result.run_id}[/cyan]")
        self.console.print(f"Duration: [cyan]{result.duration_seconds:.1f}s[/cyan]")
        self.console.print(f"Converged: [cyan]{result.converged}[/cyan]")

        total_evaluations = result.num_generations * self.config.population_size * 2 + 1
        cached_count = total_evaluations - cache_size
        cache_savings = (cached_count / total_evaluations * 100) if total_evaluations > 0 else 0

        self.console.print(f"Cache: [cyan]{cache_size} unique prompts, "
                          f"~{cache_savings:.0f}% LLM calls saved[/cyan]\n")

        self.console.print("[bold]Baseline Metrics:[/bold]")
        self.console.print(f"  {result.baseline_metrics}\n")

        self.console.print(f"[bold]Pareto Frontier ({len(result.pareto_frontier)} candidates):[/bold]")

        for i, candidate in enumerate(result.pareto_frontier, 1):
            is_recommended = candidate.id == result.recommended_prompt.id
            marker = "* " if is_recommended else "  "

            self.console.print(f"\n{marker}[bold cyan][{i}][/bold cyan] {candidate.metrics}")

            if is_recommended:
                self.console.print("    [green]^ RECOMMENDED[/green]")

        self.console.print()
