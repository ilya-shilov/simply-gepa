"""Progress tracker for optimization runs using tqdm."""

import time
from types import TracebackType
from typing import Optional, Type

from tqdm import tqdm


class ProgressTracker:
    """Track optimization progress with tqdm."""

    def __init__(
        self,
        num_generations: int,
        population_size: int = 1,
        dataset_size: int = 0
    ):
        """Initialize progress tracker."""
        self.num_generations = num_generations
        self.population_size = population_size
        self.dataset_size = dataset_size
        self.baseline_accuracy = 0.0
        self.best_accuracy = 0.0
        self._pbar: Optional[tqdm] = None
        self._start_time: Optional[float] = None

    def start(self) -> None:
        """Start the progress bar."""
        self._start_time = time.time()
        self._pbar = tqdm(
            total=self.num_generations,
            desc="GEPA Optimization",
            unit="gen",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            dynamic_ncols=True,
        )

    def close(self) -> None:
        """Close the progress bar."""
        if self._pbar:
            self._pbar.close()
            self._pbar = None

    def __enter__(self) -> "ProgressTracker":
        """Enter progress context."""
        self.start()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        """Exit progress context."""
        self.close()

    def set_start_generation(self, start_generation: int) -> None:
        """Advance progress for resumed runs."""
        if start_generation > 1 and self._pbar is not None:
            self._pbar.update(start_generation - 1)

    def set_baseline(self, accuracy: float) -> None:
        """Set baseline accuracy for improvement display."""
        self.baseline_accuracy = accuracy
        if self._pbar:
            self._pbar.set_postfix({"baseline": f"{accuracy:.1%}"})

    def update_generation(self, generation: int, best_accuracy: float) -> None:
        """Update generation progress with accuracy info."""
        if best_accuracy > self.best_accuracy:
            self.best_accuracy = best_accuracy

        improvement = (self.best_accuracy - self.baseline_accuracy) * 100 if self.baseline_accuracy else 0
        if self._pbar:
            self._pbar.set_postfix({
                "gen": f"{generation}/{self.num_generations}",
                "best": f"{self.best_accuracy:.1%}",
                "impr": f"{improvement:+.1f}%",
            })

    def advance(self, amount: int = 1) -> None:
        """Advance progress by amount."""
        if self._pbar:
            self._pbar.update(amount)

    def on_request_complete(self, completed: int, total: int) -> None:
        """Callback for LLM request completion tracking."""
        if self._pbar:
            self._pbar.set_postfix_str(
                f"gen={self._pbar.n}/{self.num_generations} "
                f"best={self.best_accuracy:.1%} "
                f"reqs={completed}/{total}",
                refresh=True,
            )
