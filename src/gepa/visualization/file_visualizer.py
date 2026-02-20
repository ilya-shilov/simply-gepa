"""File-based visualizer that saves PNG snapshots of the evolution tree."""

import shutil
import time
from pathlib import Path
from typing import List, Optional, Set

from loguru import logger

from ..models import PromptCandidate
from .live_visualizer import LiveVisualizer

DEFAULT_UPDATE_INTERVAL = 60.0


class FileVisualizer(LiveVisualizer):
    """Visualizer that saves evolution tree as PNG files."""

    def __init__(
        self,
        title: str = "GEPA Evolution Tree",
        output_dir: str = "plots",
        update_interval: float = DEFAULT_UPDATE_INTERVAL,
    ):
        """Initialize file visualizer with output directory and throttling."""
        super().__init__(title)
        self.output_dir = Path(output_dir)
        self.update_interval = update_interval
        self._last_update = 0.0
        self._plot_count = 0

    def start(self) -> None:
        """Start visualization and create output directory."""
        if self.running:
            return
        self.running = True
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._last_update = time.time()
        logger.info(f"FileVisualizer started. Output: {self.output_dir}")

    def add_candidate(self, candidate: PromptCandidate, is_baseline: bool = False) -> None:
        """Add candidate and force plot update on baseline."""
        super().add_candidate(candidate, is_baseline)
        if is_baseline:
            acc = candidate.metrics.accuracy
            fn_rate = candidate.metrics.false_negative_rate
            fp_rate = candidate.metrics.false_positive_rate
            logger.success(
                f"Baseline: accuracy={acc:.1%}, FN={fn_rate:.1%}, FP={fp_rate:.1%}"
            )
            self._last_update = 0
            self._update_plot()

    def update_generation(self, generation: int, population: List[PromptCandidate]) -> None:
        """Update visualization after generation completes."""
        best_acc = max((c.metrics.accuracy for c in population), default=0)
        logger.info(
            f"Generation {generation} complete. "
            f"Candidates: {len(self.candidates)}, Best: {best_acc:.1%}"
        )
        super().update_generation(generation, population)

    def _update_plot(self, highlight_ids: Optional[Set[str]] = None) -> None:
        """Render evolution tree to PNG with throttling."""
        if not self.running:
            return

        now = time.time()
        if (now - self._last_update) < self.update_interval:
            return

        self._last_update = now
        self._plot_count += 1

        if len(self.graph.nodes) == 0:
            return

        self._render_to_file(highlight_ids)

    def _render_to_file(self, highlight_ids: Optional[Set[str]] = None) -> None:
        """Render the evolution tree graph to a PNG file."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import networkx as nx
        from matplotlib.patches import Patch

        fig, ax = plt.subplots(figsize=(14, 10))

        pos = {}
        max_gen = max(self.generation_candidates.keys()) if self.generation_candidates else 0

        for gen, candidate_ids in self.generation_candidates.items():
            num = len(candidate_ids)
            y = 1.0 - (gen / max(max_gen, 1))
            for i, cid in enumerate(candidate_ids):
                x = 0.5 if num == 1 else 0.1 + (i / (num - 1)) * 0.8
                pos[cid] = (x, y)

        node_colors = []
        node_sizes = []

        for node_id in self.graph.nodes():
            candidate = self.candidates[node_id]
            accuracy = candidate.metrics.accuracy

            if accuracy >= 0.8:
                color = "#2ecc71"
            elif accuracy >= 0.6:
                color = "#f39c12"
            else:
                color = "#e74c3c"

            if self.graph.nodes[node_id].get("is_baseline"):
                color = "#3498db"

            if highlight_ids and node_id in highlight_ids:
                color = "#9b59b6"

            node_colors.append(color)
            node_sizes.append(1000 if self.graph.nodes[node_id].get("is_baseline") else 600)

        nx.draw_networkx_nodes(
            self.graph, pos, node_color=node_colors,
            node_size=node_sizes, ax=ax, alpha=0.9
        )
        nx.draw_networkx_edges(
            self.graph, pos, edge_color="#95a5a6",
            arrows=True, arrowsize=15, width=2, ax=ax, alpha=0.6
        )

        labels = {}
        for node_id in self.graph.nodes():
            candidate = self.candidates[node_id]
            gen = candidate.generation
            acc = candidate.metrics.accuracy
            if self.graph.nodes[node_id].get("is_baseline"):
                labels[node_id] = f"G{gen}\nBaseline\n{acc:.1%}"
            else:
                labels[node_id] = f"G{gen}\n{acc:.1%}"

        nx.draw_networkx_labels(self.graph, pos, labels, font_size=8, font_weight="bold", ax=ax)

        best_acc = max(self.best_accuracy_per_gen.values()) if self.best_accuracy_per_gen else 0
        ax.set_title(
            f"{self.title}\n"
            f"Generations: {len(self.generation_candidates)} | "
            f"Candidates: {len(self.candidates)} | "
            f"Best Accuracy: {best_acc:.1%}",
            fontsize=14, fontweight="bold", pad=20
        )

        legend_elements = [
            Patch(facecolor="#3498db", label="Baseline"),
            Patch(facecolor="#2ecc71", label="Accuracy >= 80%"),
            Patch(facecolor="#f39c12", label="Accuracy 60-80%"),
            Patch(facecolor="#e74c3c", label="Accuracy < 60%"),
        ]
        if highlight_ids:
            legend_elements.append(Patch(facecolor="#9b59b6", label="Pareto Frontier"))

        ax.legend(handles=legend_elements, loc="upper left", fontsize=9, framealpha=0.9)
        ax.axis("off")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        plt.tight_layout()

        filename = self.output_dir / f"evolution_{self._plot_count:03d}.png"
        fig.savefig(filename, dpi=150)
        plt.close(fig)
        logger.info(f"Saved evolution plot: {filename}")

    def show_final(self, pareto_frontier: List[PromptCandidate]) -> None:
        """Save final plot with Pareto frontier highlighted and copy as final.png."""
        self._last_update = 0
        highlight_ids = {c.id for c in pareto_frontier}
        self._update_plot(highlight_ids=highlight_ids)

        if self._plot_count > 0:
            last_file = self.output_dir / f"evolution_{self._plot_count:03d}.png"
            final_file = self.output_dir / "final.png"
            if last_file.exists():
                shutil.copy(last_file, final_file)
                logger.info(f"Saved final visualization: {final_file}")
