"""GEPA - Genetic-Pareto Prompt Optimizer."""

from .analysis import ErrorAnalyzer
from .clients import BaseLLMClient, LLMClient
from .config import Settings, get_settings
from .core import GEPAOptimizer, ParetoSelector, PromptEvaluator, PromptMutator
from .core.evaluator import default_compare_fn
from .models import (
    DatasetEntry,
    OptimizationConfig,
    OptimizationResult,
    PromptCandidate,
    PromptMetrics,
)
from .visualization import FileVisualizer, LiveVisualizer

__version__ = "0.3.0"

__all__ = [
    "GEPAOptimizer",
    "PromptEvaluator",
    "PromptMutator",
    "ParetoSelector",
    "BaseLLMClient",
    "LLMClient",
    "ErrorAnalyzer",
    "FileVisualizer",
    "LiveVisualizer",
    "Settings",
    "get_settings",
    "OptimizationConfig",
    "OptimizationResult",
    "PromptCandidate",
    "PromptMetrics",
    "DatasetEntry",
    "default_compare_fn",
]
