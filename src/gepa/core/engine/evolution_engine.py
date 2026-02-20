"""Evolution engine for candidate generation."""

import random
import uuid
from difflib import SequenceMatcher
from typing import Callable, List, Optional, Tuple

from loguru import logger

from ...models import DatasetEntry, OptimizationConfig, PromptCandidate
from ...visualization import LiveVisualizer
from ..crossover import PromptCrossover
from ..mutator import PromptMutator
from ..io.mutation_logger import MutationLogger
from ..pareto import ParetoSelector

MIN_CROSSOVER_POPULATION = 2
MIN_CROSSOVER_OFFSPRING = 1
PARENT_ID_PREFIX_LENGTH = 8
ZERO_RATE = 0.0
DIVERSITY_SIMILARITY_THRESHOLD = 0.7


class EvolutionEngine:
    """Generate new candidates for each generation."""

    def __init__(
        self,
        config: OptimizationConfig,
        mutator: PromptMutator,
        crossover: PromptCrossover,
        pareto_selector: ParetoSelector,
        mutation_logger: MutationLogger,
        evaluate_with_cache: Callable[[str], tuple],
        visualizer: Optional[LiveVisualizer] = None
    ):
        """Initialize evolution engine."""
        self.config = config
        self.mutator = mutator
        self.crossover = crossover
        self.pareto_selector = pareto_selector
        self.mutation_logger = mutation_logger
        self.evaluate_with_cache = evaluate_with_cache
        self.visualizer = visualizer

    def generate_candidates(
        self,
        population: List[PromptCandidate],
        generation: int,
        temperature: float
    ) -> List[PromptCandidate]:
        """Generate mutation, crossover, and diversity candidates."""
        new_candidates = self._generate_mutation_candidates(
            population=population,
            generation=generation,
            temperature=temperature
        )
        new_candidates.extend(
            self._generate_crossover_candidates(
                population=population,
                generation=generation
            )
        )
        new_candidates.extend(
            self._generate_diversity_candidates(
                population=population,
                generation=generation,
                temperature=temperature
            )
        )
        return new_candidates

    def _generate_mutation_candidates(
        self,
        population: List[PromptCandidate],
        generation: int,
        temperature: float
    ) -> List[PromptCandidate]:
        """Generate mutation candidates for a generation."""
        new_candidates: List[PromptCandidate] = []
        for parent in population:
            _, failed_examples = self.evaluate_with_cache(parent.text)
            if not failed_examples:
                logger.debug(f"Candidate {parent.id} has no failures, skipping mutation")
                continue
            improved_text, reflection = self.mutator.mutate_prompt_with_reflection(
                candidate=parent,
                failed_examples=failed_examples,
                generation=generation,
                temperature=temperature
            )
            improved_metrics, _ = self.evaluate_with_cache(improved_text)
            new_candidate = PromptCandidate(
                id=str(uuid.uuid4()),
                text=improved_text,
                metrics=improved_metrics,
                generation=generation,
                parent_id=parent.id,
                mutation_notes=f"Mutated from {parent.id[:PARENT_ID_PREFIX_LENGTH]}"
            )
            new_candidates.append(new_candidate)
            self.mutation_logger.append(
                candidate=new_candidate,
                parent=parent,
                generation=generation,
                reflection=reflection,
                failed_examples=failed_examples,
                mutation_kind="mutation"
            )
            if self.visualizer:
                self.visualizer.add_candidate(new_candidate)
            logger.info(
                f"Mutation: {parent.metrics.accuracy:.2%} -> {improved_metrics.accuracy:.2%}"
            )
        return new_candidates

    def _generate_crossover_candidates(
        self,
        population: List[PromptCandidate],
        generation: int
    ) -> List[PromptCandidate]:
        """Generate crossover candidates for a generation."""
        if not self.config.crossover_enabled:
            return []
        if len(population) < MIN_CROSSOVER_POPULATION:
            return []
        offspring_count = self._get_crossover_offspring_count(self.config.population_size)
        if offspring_count <= 0:
            return []
        parent_pool = self._select_crossover_parent_pool(population)
        pairs = self._select_crossover_pairs(parent_pool, offspring_count)
        new_candidates: List[PromptCandidate] = []
        for parent_a, parent_b in pairs:
            hybrid_text = self.crossover.crossover_prompts(
                prompt_a=parent_a.text,
                prompt_b=parent_b.text,
                temperature=self.config.crossover_temperature
            )
            hybrid_metrics, _ = self.evaluate_with_cache(hybrid_text)
            new_candidate = PromptCandidate(
                id=str(uuid.uuid4()),
                text=hybrid_text,
                metrics=hybrid_metrics,
                generation=generation,
                parent_id=parent_a.id,
                mutation_notes=(
                    f"Crossover {parent_a.id[:PARENT_ID_PREFIX_LENGTH]}"
                    f"+{parent_b.id[:PARENT_ID_PREFIX_LENGTH]}"
                )
            )
            new_candidates.append(new_candidate)
            if self.visualizer:
                self.visualizer.add_candidate(new_candidate)
            logger.info(
                "Crossover: "
                f"{parent_a.metrics.accuracy:.2%} + "
                f"{parent_b.metrics.accuracy:.2%} -> "
                f"{hybrid_metrics.accuracy:.2%}"
            )
        return new_candidates

    def _generate_diversity_candidates(
        self,
        population: List[PromptCandidate],
        generation: int,
        temperature: float
    ) -> List[PromptCandidate]:
        """Generate exploration candidates when diversity is low."""
        if not self._calculate_diversity(population):
            return []
        parent = self._select_parent_with_failures(population)
        if not parent:
            return []
        _, failed_examples = self.evaluate_with_cache(parent.text)
        if not failed_examples:
            return []
        improved_text, reflection = self.mutator.mutate_prompt_with_reflection(
            candidate=parent,
            failed_examples=failed_examples,
            generation=generation,
            temperature=temperature
        )
        improved_metrics, _ = self.evaluate_with_cache(improved_text)
        new_candidate = PromptCandidate(
            id=str(uuid.uuid4()),
            text=improved_text,
            metrics=improved_metrics,
            generation=generation,
            parent_id=parent.id,
            mutation_notes=f"Diversity from {parent.id[:PARENT_ID_PREFIX_LENGTH]}"
        )
        self.mutation_logger.append(
            candidate=new_candidate,
            parent=parent,
            generation=generation,
            reflection=reflection,
            failed_examples=failed_examples,
            mutation_kind="diversity"
        )
        if self.visualizer:
            self.visualizer.add_candidate(new_candidate)
        logger.info(
            f"Diversity mutation: {parent.metrics.accuracy:.2%} -> {improved_metrics.accuracy:.2%}"
        )
        return [new_candidate]

    def _select_parent_with_failures(
        self,
        population: List[PromptCandidate]
    ) -> Optional[PromptCandidate]:
        """Select a random parent that has failures."""
        candidates = population.copy()
        random.shuffle(candidates)
        for parent in candidates:
            _, failed_examples = self.evaluate_with_cache(parent.text)
            if failed_examples:
                return parent
        return None

    def _calculate_diversity(self, population: List[PromptCandidate]) -> bool:
        """Detect low diversity using average prompt similarity."""
        texts = [candidate.text for candidate in population]
        avg_similarity = self._average_similarity(texts)
        if avg_similarity > DIVERSITY_SIMILARITY_THRESHOLD:
            logger.warning(
                f"Low diversity ({avg_similarity:.1%}), forcing exploration"
            )
            return True
        return False

    def _average_similarity(self, texts: List[str]) -> float:
        """Compute average pairwise similarity."""
        total_similarity = ZERO_RATE
        pairs = 0
        for i, text_a in enumerate(texts):
            for text_b in texts[i + 1:]:
                similarity = SequenceMatcher(None, text_a, text_b).ratio()
                total_similarity += similarity
                pairs += 1
        if pairs == 0:
            return ZERO_RATE
        return total_similarity / pairs

    def _select_crossover_parent_pool(
        self,
        population: List[PromptCandidate]
    ) -> List[PromptCandidate]:
        """Select parent pool for crossover."""
        frontier = self.pareto_selector.get_pareto_frontier(population)
        pool = frontier if frontier else population
        pool = sorted(pool, key=lambda c: c.metrics.accuracy, reverse=True)
        pool_size = min(self.config.crossover_parent_pool_size, len(pool))
        return pool[:pool_size]

    def _select_crossover_pairs(
        self,
        parent_pool: List[PromptCandidate],
        offspring_count: int
    ) -> List[Tuple[PromptCandidate, PromptCandidate]]:
        """Select parent pairs for crossover."""
        if len(parent_pool) < MIN_CROSSOVER_POPULATION:
            return []
        pairs = [
            (parent_pool[i], parent_pool[j])
            for i in range(len(parent_pool))
            for j in range(i + 1, len(parent_pool))
        ]
        random.shuffle(pairs)
        return pairs[:min(offspring_count, len(pairs))]

    def _get_crossover_offspring_count(self, population_size: int) -> int:
        """Calculate number of crossover offspring."""
        if self.config.crossover_rate <= ZERO_RATE:
            return 0
        count = int(population_size * self.config.crossover_rate)
        if count < MIN_CROSSOVER_OFFSPRING:
            count = MIN_CROSSOVER_OFFSPRING
        return count
