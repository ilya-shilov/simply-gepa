"""GEPA Optimizer - Main genetic-Pareto optimization algorithm."""

import asyncio
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Awaitable, Callable, Dict, List, Optional, Tuple

from loguru import logger
try:
    from rich.console import Console
except Exception:  # pragma: no cover - optional dependency guard
    class Console:  # type: ignore[override]
        """Fallback console when rich is not installed."""

        def print(self, *args, **kwargs) -> None:
            if args:
                print(*args)

from ...analysis import ErrorAnalyzer
from ...clients import BaseLLMClient
from ...models import (
    DatasetEntry,
    OptimizationConfig,
    OptimizationResult,
    PromptCandidate,
    PromptMetrics,
)
from ...visualization import FileVisualizer, LiveVisualizer
from ..crossover import PromptCrossover
from ..evaluator import PromptEvaluator
from .evolution_engine import EvolutionEngine
from ..io.mutation_logger import MutationLogger
from ..ui.progress_tracker import ProgressTracker
from ..mutator import PromptMutator
from ..pareto import ParetoSelector
from ..io.result_builder import ResultBuilder
from ..state.state_manager import OptimizerStateManager
from ..ui.visualizer_manager import VisualizerManager

EvalFn = Callable[[str, Dict], Awaitable[str]]
CompareFn = Callable[[str, str], bool]
FailureFormatFn = Callable[[DatasetEntry], str]

CONVERGENCE_WINDOW = 3
CONVERGENCE_ACCURACY_DELTA = 0.01


class GEPAOptimizer:
    """Genetic-Pareto optimizer for prompt engineering."""

    def __init__(
        self,
        llm_client: BaseLLMClient,
        config: OptimizationConfig,
        visualizer: Optional[LiveVisualizer] = None,
        eval_fn: Optional[EvalFn] = None,
        compare_fn: Optional[CompareFn] = None,
        failure_format_fn: Optional[FailureFormatFn] = None,
        meta_llm_client: Optional[BaseLLMClient] = None
    ):
        """Initialize GEPA optimizer."""
        self.llm = llm_client
        self.config = config
        self.console = Console()

        if visualizer is None:
            if FileVisualizer is None:
                raise ImportError(
                    "Visualization dependencies are not installed. "
                    "Install with `pip install gepa[viz]`."
                )
            plots_dir = str(Path(config.runs_dir) / "plots")
            self.visualizer: LiveVisualizer = FileVisualizer(output_dir=plots_dir)
        else:
            self.visualizer = visualizer

        meta_llm = meta_llm_client or llm_client

        self.evaluator = PromptEvaluator(
            llm_client=llm_client,
            eval_fn=eval_fn,
            compare_fn=compare_fn
        )
        self.mutator = PromptMutator(
            llm_client=meta_llm,
            config=config,
            failure_format_fn=failure_format_fn
        )
        self.crossover = PromptCrossover(llm_client=meta_llm, config=config)
        self.pareto_selector = ParetoSelector(config)
        self.error_analyzer = ErrorAnalyzer(meta_llm)
        self.state_manager = OptimizerStateManager(config)
        self.mutation_logger = MutationLogger(self.state_manager.get_runs_dir())
        self.result_builder = ResultBuilder(
            config=config,
            pareto_selector=self.pareto_selector,
            error_analyzer=self.error_analyzer,
            console=self.console
        )
        self.visualizer_manager = VisualizerManager(self.visualizer)
        self.evolution_engine = EvolutionEngine(
            config=config,
            mutator=self.mutator,
            crossover=self.crossover,
            pareto_selector=self.pareto_selector,
            mutation_logger=self.mutation_logger,
            evaluate_with_cache=self._evaluate_with_cache,
            visualizer=self.visualizer
        )

        self.dataset: List[DatasetEntry] = []
        self.all_generations: List[List[PromptCandidate]] = []
        self.evaluation_cache: dict = {}
        self.run_id: Optional[str] = None

    def optimize(self, baseline_prompt: str, resume_from: Optional[str] = None) -> OptimizationResult:
        """Run GEPA optimization."""
        start_time = time.time()
        self._initialize_run_id()
        self.dataset = self.evaluator.load_dataset(Path(self.config.dataset_path))
        self.visualizer_manager.start()
        baseline_prompt, baseline_metrics, population, start_generation = self._initialize_population(
            baseline_prompt,
            resume_from
        )
        self.mutation_logger.set_run_id(self.run_id or "gepa_run")
        self._log_run_settings()
        self.visualizer_manager.add_baseline(population)
        population = self._run_generations(
            population=population,
            baseline_prompt=baseline_prompt,
            baseline_metrics=baseline_metrics,
            start_generation=start_generation
        )
        result = self.result_builder.build(
            run_id=self.run_id or "gepa_run",
            baseline_prompt=baseline_prompt,
            baseline_metrics=baseline_metrics,
            population=population,
            all_generations=self.all_generations,
            dataset_size=len(self.dataset),
            start_time=start_time
        )
        self.result_builder.log_result(result, len(self.evaluation_cache))
        self.visualizer_manager.show_final(result.pareto_frontier)
        return result

    def _initialize_run_id(self) -> None:
        """Initialize run metadata."""
        self.run_id = f"gepa_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def _initialize_population(
        self,
        baseline_prompt: str,
        resume_from: Optional[str]
    ) -> Tuple[str, PromptMetrics, List[PromptCandidate], int]:
        """Initialize population and baseline metrics."""
        if resume_from:
            return self._resume_from_state(resume_from)
        return self._initialize_from_baseline(baseline_prompt)

    def _resume_from_state(
        self,
        resume_from: str
    ) -> Tuple[str, PromptMetrics, List[PromptCandidate], int]:
        """Restore state from a previous run."""
        state_path = self.state_manager.resolve_state_path(resume_from)
        state = self.state_manager.load_state(state_path)
        self.run_id = state["run_id"]
        self.all_generations = state["all_generations"]
        logger.info(f"Resuming GEPA optimization: {self.run_id}")
        return (
            state["baseline_prompt"],
            state["baseline_metrics"],
            state["population"],
            state["generation"] + 1
        )

    def _initialize_from_baseline(
        self,
        baseline_prompt: str
    ) -> Tuple[str, PromptMetrics, List[PromptCandidate], int]:
        """Evaluate baseline and build initial population."""
        logger.info(f"Starting GEPA optimization: {self.run_id}")
        self.console.print("\n[bold blue]Evaluating baseline prompt...[/bold blue]")
        baseline_metrics, _ = self._evaluate_with_cache(baseline_prompt)
        self.console.print(f"[green]Baseline: {baseline_metrics}[/green]\n")
        baseline_candidate = self._create_baseline_candidate(baseline_prompt, baseline_metrics)
        self.all_generations = []
        return baseline_prompt, baseline_metrics, [baseline_candidate], 1

    def _create_baseline_candidate(
        self,
        baseline_prompt: str,
        baseline_metrics: PromptMetrics
    ) -> PromptCandidate:
        """Create baseline candidate."""
        return PromptCandidate(
            id=str(uuid.uuid4()),
            text=baseline_prompt,
            metrics=baseline_metrics,
            generation=0,
            parent_id=None,
            mutation_notes="Baseline prompt"
        )

    def _log_run_settings(self) -> None:
        """Log core run settings."""
        logger.info(f"Generations: {self.config.num_generations}")
        logger.info(f"Population: {self.config.population_size}")

    def _run_generations(
        self,
        population: List[PromptCandidate],
        baseline_prompt: str,
        baseline_metrics: PromptMetrics,
        start_generation: int
    ) -> List[PromptCandidate]:
        """Run optimization generations."""
        with ProgressTracker(self.config.num_generations, self.config.population_size) as progress:
            progress.set_baseline(baseline_metrics.accuracy)
            progress.set_start_generation(start_generation)
            for generation in range(start_generation, self.config.num_generations + 1):
                population = self._run_generation(
                    generation=generation,
                    population=population,
                    baseline_prompt=baseline_prompt,
                    baseline_metrics=baseline_metrics,
                    progress=progress
                )
                if self._check_convergence(generation):
                    logger.info(f"Converged at generation {generation}")
                    break
        return population

    def _run_generation(
        self,
        generation: int,
        population: List[PromptCandidate],
        baseline_prompt: str,
        baseline_metrics: PromptMetrics,
        progress: ProgressTracker
    ) -> List[PromptCandidate]:
        """Execute a single generation."""
        self._log_generation_header(generation)
        temperature = self._get_mutation_temperature(generation)
        new_candidates = self.evolution_engine.generate_candidates(
            population=population,
            generation=generation,
            temperature=temperature
        )
        all_candidates = population + new_candidates
        population = self.pareto_selector.select_population(
            all_candidates,
            self.config.population_size
        )
        self.all_generations.append(population.copy())
        self.state_manager.save_state(
            run_id=self.run_id or "gepa_run",
            generation=generation,
            population=population,
            baseline_prompt=baseline_prompt,
            baseline_metrics=baseline_metrics,
            all_generations=self.all_generations
        )
        self.visualizer_manager.update_generation(generation, population)
        best = max(population, key=lambda c: c.metrics.accuracy)
        progress.update_generation(generation, best.metrics.accuracy)
        progress.advance()
        return population

    def _log_generation_header(self, generation: int) -> None:
        """Log generation header."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Generation {generation}/{self.config.num_generations}")
        logger.info(f"{'='*60}")

    def _get_mutation_temperature(self, generation: int) -> float:
        """Get mutation temperature for current generation."""
        if not self.config.adaptive_mutation_enabled:
            return self.config.mutation_temperature
        if self.config.num_generations <= 1:
            return self.config.adaptive_mutation_end_temperature
        progress = (generation - 1) / (self.config.num_generations - 1)
        start_temp = self.config.adaptive_mutation_start_temperature
        end_temp = self.config.adaptive_mutation_end_temperature
        return start_temp - (start_temp - end_temp) * progress

    def _evaluate_with_cache(self, prompt_text: str):
        """Evaluate prompt with caching to avoid redundant LLM calls."""
        if prompt_text in self.evaluation_cache:
            logger.debug(f"Cache HIT for prompt ({len(prompt_text)} chars)")
            return self.evaluation_cache[prompt_text]
        logger.debug(f"Cache MISS for prompt ({len(prompt_text)} chars) - evaluating...")
        metrics, failed_examples = self._evaluate_sync(prompt_text)
        self.evaluation_cache[prompt_text] = (metrics, failed_examples)
        logger.debug(f"Cached result (cache size: {len(self.evaluation_cache)})")
        return metrics, failed_examples

    def _evaluate_sync(self, prompt_text: str):
        """Synchronous wrapper for async evaluation."""
        if self.config.early_stop_enabled:
            return asyncio.run(
                self.evaluator.evaluate_prompt_with_early_stopping(
                    prompt_text=prompt_text,
                    dataset=self.dataset,
                    criterion=self.config.criterion_name,
                    sample_ratio=self.config.early_stop_sample_ratio,
                    accuracy_threshold=self.config.early_stop_accuracy_threshold
                )
            )
        return asyncio.run(
            self.evaluator.evaluate_prompt(
                prompt_text=prompt_text,
                dataset=self.dataset,
                criterion=self.config.criterion_name
            )
        )

    def _check_convergence(self, generation: int, window: int = CONVERGENCE_WINDOW) -> bool:
        """Check if optimization has converged."""
        if len(self.all_generations) < window:
            return False
        recent_gens = self.all_generations[-window:]
        accuracies = [
            max(gen, key=lambda c: c.metrics.accuracy).metrics.accuracy
            for gen in recent_gens
        ]
        max_acc = max(accuracies)
        min_acc = min(accuracies)
        return (max_acc - min_acc) < CONVERGENCE_ACCURACY_DELTA
