"""Visualization manager."""

from typing import List, Optional

from loguru import logger

from ...models import PromptCandidate
from ...visualization import LiveVisualizer


class VisualizerManager:
    """Manage visualization lifecycle and updates."""

    def __init__(self, visualizer: Optional[LiveVisualizer]):
        """Initialize visualizer manager."""
        self.visualizer = visualizer

    def start(self) -> None:
        """Start visualizer if available."""
        if self.visualizer:
            self.visualizer.start()
            logger.info("Live visualization started")

    def add_baseline(self, population: List[PromptCandidate]) -> None:
        """Add baseline candidates."""
        if self.visualizer:
            for candidate in population:
                self.visualizer.add_candidate(candidate, is_baseline=candidate.generation == 0)

    def update_generation(self, generation: int, population: List[PromptCandidate]) -> None:
        """Update visualizer for a generation."""
        if self.visualizer:
            self.visualizer.update_generation(generation, population)

    def show_final(self, pareto_frontier: List[PromptCandidate]) -> None:
        """Show final visualization."""
        if self.visualizer:
            self.visualizer.show_final(pareto_frontier)
