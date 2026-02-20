"""Live visualization of GEPA optimization progress."""

import threading
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation

from ..models import PromptCandidate, PromptMetrics


class LiveVisualizer:
    """Real-time visualizer for GEPA evolution tree."""

    def __init__(self, title: str = "GEPA Evolution Tree"):
        """Initialize live visualizer."""
        self.title = title
        self.graph = nx.DiGraph()
        self.candidates: Dict[str, PromptCandidate] = {}
        self.generation_candidates: Dict[int, List[str]] = {}

                             
        self.fig = None
        self.ax = None
        self.lock = threading.Lock()
        self.running = False
        self.needs_update = False                         

                                           
        self.best_accuracy_per_gen: Dict[int, float] = {}

    def start(self):
        """Start the visualization (PyCharm Plots compatible)."""
        if self.running:
            return

        self.running = True
                                                                                

    def stop(self):
        """Stop the visualization."""
        self.running = False
        if self.fig:
            plt.ioff()

    def add_candidate(
        self,
        candidate: PromptCandidate,
        is_baseline: bool = False
    ):
        """Add a candidate to the visualization."""
        with self.lock:
                             
            self.candidates[candidate.id] = candidate

                          
            self.graph.add_node(
                candidate.id,
                generation=candidate.generation,
                accuracy=candidate.metrics.accuracy,
                is_baseline=is_baseline
            )

                                  
            if candidate.parent_id and candidate.parent_id in self.graph:
                self.graph.add_edge(candidate.parent_id, candidate.id)

                                 
            gen = candidate.generation
            if gen not in self.generation_candidates:
                self.generation_candidates[gen] = []
            self.generation_candidates[gen].append(candidate.id)

                                                      
            if gen not in self.best_accuracy_per_gen:
                self.best_accuracy_per_gen[gen] = candidate.metrics.accuracy
            else:
                self.best_accuracy_per_gen[gen] = max(
                    self.best_accuracy_per_gen[gen],
                    candidate.metrics.accuracy
                )

                                                                        
            self.needs_update = True

    def update_generation(self, generation: int, population: List[PromptCandidate]):
        """Update visualization after generation completes."""
        with self.lock:
                                                               
            surviving_ids = {c.id for c in population}

                                                             
            if self.running and self.needs_update:
                self._update_plot(highlight_ids=surviving_ids)
                self.needs_update = False              

    def _update_plot(self, highlight_ids: Optional[set] = None):
        """Update the plot visualization."""
        if not self.running:
            return

                                             
        plt.close('all')                          
        self.fig, self.ax = plt.subplots(figsize=(14, 10))
        self.fig.suptitle(self.title, fontsize=16, fontweight='bold')

        if len(self.graph.nodes) == 0:
            self.ax.text(
                0.5, 0.5,
                "Waiting for candidates...",
                ha='center',
                va='center',
                fontsize=14
            )
            plt.pause(0.01)
            return

                                                  
        pos = {}
        max_gen = max(self.generation_candidates.keys())

        for gen, candidate_ids in self.generation_candidates.items():
            num_candidates = len(candidate_ids)
            y = 1.0 - (gen / max(max_gen, 1))                                

            for i, cand_id in enumerate(candidate_ids):
                                     
                if num_candidates == 1:
                    x = 0.5
                else:
                    x = 0.1 + (i / (num_candidates - 1)) * 0.8

                pos[cand_id] = (x, y)

                                                         
        node_colors = []
        node_sizes = []

        for node_id in self.graph.nodes():
            candidate = self.candidates[node_id]
            accuracy = candidate.metrics.accuracy

                                                               
            if accuracy >= 0.8:
                color = '#2ecc71'         
            elif accuracy >= 0.6:
                color = '#f39c12'          
            else:
                color = '#e74c3c'       

                                     
            if self.graph.nodes[node_id].get('is_baseline'):
                color = '#3498db'                     

            if highlight_ids and node_id in highlight_ids:
                color = '#9b59b6'                              

            node_colors.append(color)
            node_sizes.append(1000 if self.graph.nodes[node_id].get('is_baseline') else 600)

                    
        nx.draw_networkx_nodes(
            self.graph,
            pos,
            node_color=node_colors,
            node_size=node_sizes,
            ax=self.ax,
            alpha=0.9
        )

        nx.draw_networkx_edges(
            self.graph,
            pos,
            edge_color='#95a5a6',
            arrows=True,
            arrowsize=15,
            width=2,
            ax=self.ax,
            alpha=0.6,
            arrowstyle='->'
        )

                                  
        labels = {}
        for node_id in self.graph.nodes():
            candidate = self.candidates[node_id]
            gen = candidate.generation
            acc = candidate.metrics.accuracy

            if self.graph.nodes[node_id].get('is_baseline'):
                labels[node_id] = f"G{gen}\nBaseline\n{acc:.1%}"
            else:
                labels[node_id] = f"G{gen}\n{acc:.1%}"

        nx.draw_networkx_labels(
            self.graph,
            pos,
            labels,
            font_size=8,
            font_weight='bold',
            ax=self.ax
        )

                                   
        num_candidates = len(self.candidates)
        num_generations = len(self.generation_candidates)

        if self.best_accuracy_per_gen:
            best_overall = max(self.best_accuracy_per_gen.values())
            title = (
                f"{self.title}\n"
                f"Generations: {num_generations} | "
                f"Candidates: {num_candidates} | "
                f"Best Accuracy: {best_overall:.1%}"
            )
        else:
            title = f"{self.title}\nCandidates: {num_candidates}"

        self.ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

                    
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#3498db', label='Baseline'),
            Patch(facecolor='#2ecc71', label='Accuracy ≥ 80%'),
            Patch(facecolor='#f39c12', label='Accuracy 60-80%'),
            Patch(facecolor='#e74c3c', label='Accuracy < 60%'),
        ]

        if highlight_ids:
            legend_elements.append(
                Patch(facecolor='#9b59b6', label='Pareto Frontier')
            )

        self.ax.legend(
            handles=legend_elements,
            loc='upper left',
            fontsize=9,
            framealpha=0.9
        )

        self.ax.axis('off')
        self.ax.set_xlim(-0.05, 1.05)
        self.ax.set_ylim(-0.05, 1.05)

                                     
        plt.tight_layout()
        plt.show()

    def show_final(self, pareto_frontier: List[PromptCandidate]):
        """Show final visualization with Pareto frontier highlighted."""
        pareto_ids = {c.id for c in pareto_frontier}

        with self.lock:
            self._update_plot(highlight_ids=pareto_ids)
