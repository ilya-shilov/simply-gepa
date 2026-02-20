"""Optimization configuration models."""

from typing import Any, Dict, Optional, Set

from pydantic import BaseModel, Field

DEFAULT_CROSSOVER_RATE = 0.3
DEFAULT_CROSSOVER_TEMPERATURE = 0.6
DEFAULT_CROSSOVER_PARENT_POOL_SIZE = 3
MIN_CROSSOVER_RATE = 0.0
MAX_CROSSOVER_RATE = 1.0
MIN_CROSSOVER_TEMPERATURE = 0.0
MAX_CROSSOVER_TEMPERATURE = 1.0
MIN_CROSSOVER_PARENT_POOL_SIZE = 2
MAX_CROSSOVER_PARENT_POOL_SIZE = 20
DEFAULT_PRIORITY_ACCURACY = 2.0
DEFAULT_PRIORITY_FN_REDUCTION = 2.0
DEFAULT_PRIORITY_FP_REDUCTION = 0.5
DEFAULT_PRIORITY_COST_REDUCTION = 0.001

DEFAULT_MUTATION_SYSTEM_PROMPT = (
    "You are an expert in prompt engineering and optimization."
)

DEFAULT_REFLECTION_TEMPLATE = """You are an expert in prompt optimization.

CURRENT PROMPT:
{prompt_text}

EXAMPLES WHERE THE PROMPT FAILED:
{failures_text}

Analyze why the prompt failed:
1. What error patterns do you see?
2. What is missing from the prompt?
3. Which instructions are too vague?

Reply concisely (2-3 points) with specific suggestions for improvement."""

DEFAULT_IMPROVEMENT_TEMPLATE = """You are an expert in prompt engineering.

ORIGINAL PROMPT:
{original_prompt}

ERROR ANALYSIS:
{reflection}

Create an IMPROVED version of the prompt that:
1. Fixes the identified problems
2. Adds clearer instructions
3. Includes examples or clarifications for difficult cases
4. Stays approximately the same length (+-20%)

IMPORTANT: Preserve the structure and style of the original prompt. Improve, don't rewrite from scratch.

Return ONLY the improved prompt text, without additional comments."""

DEFAULT_CROSSOVER_TEMPLATE = """You are an expert in prompt optimization.

PROMPT A:
{prompt_a}

PROMPT B:
{prompt_b}

Create a HYBRID prompt that:
1. Takes the best and most precise formulations from both prompts
2. Removes duplicates and contradictions
3. Preserves the structure and style of the source prompts
4. Stays approximately the same length (+-{length_tolerance}%)

Return ONLY the final prompt text without comments."""


SUPPORTED_PROFILES: Set[str] = {"fast", "balanced", "quality", "advanced"}

PROFILE_PRESETS: Dict[str, Dict[str, Any]] = {
    "fast": {
        "num_generations": 4,
        "population_size": 12,
        "mutation_temperature": 0.9,
        "crossover_enabled": True,
        "crossover_rate": 0.4,
        "crossover_temperature": 0.6,
        "crossover_parent_pool_size": 6,
        "early_stop_enabled": True,
        "early_stop_sample_ratio": 0.2,
        "early_stop_accuracy_threshold": 0.2,
        "adaptive_mutation_enabled": False,
        "priority_accuracy": 2.0,
        "priority_fn_reduction": 2.0,
        "priority_fp_reduction": 0.5,
        "priority_cost_reduction": 0.001,
    },
    "balanced": {
        "num_generations": 8,
        "population_size": 10,
        "mutation_temperature": 0.7,
        "crossover_enabled": True,
        "crossover_rate": 0.35,
        "crossover_temperature": 0.6,
        "crossover_parent_pool_size": 5,
        "early_stop_enabled": True,
        "early_stop_sample_ratio": 0.2,
        "early_stop_accuracy_threshold": 0.2,
        "adaptive_mutation_enabled": False,
        "priority_accuracy": 2.0,
        "priority_fn_reduction": 2.0,
        "priority_fp_reduction": 0.5,
        "priority_cost_reduction": 0.001,
    },
    "quality": {
        "num_generations": 12,
        "population_size": 8,
        "mutation_temperature": 0.5,
        "crossover_enabled": False,
        "crossover_rate": 0.2,
        "crossover_temperature": 0.5,
        "crossover_parent_pool_size": 4,
        "early_stop_enabled": True,
        "early_stop_sample_ratio": 0.15,
        "early_stop_accuracy_threshold": 0.15,
        "adaptive_mutation_enabled": True,
        "adaptive_mutation_start_temperature": 0.9,
        "adaptive_mutation_end_temperature": 0.3,
        "priority_accuracy": 2.0,
        "priority_fn_reduction": 3.0,
        "priority_fp_reduction": 0.5,
        "priority_cost_reduction": 0.001,
    },
    "advanced": {},
}


class OptimizationConfig(BaseModel):
    """GEPA optimization configuration."""

    dataset_path: str
    runs_dir: Optional[str] = None
    criterion_name: str = "default"
    num_generations: int = Field(default=10, ge=1, le=50)
    population_size: int = Field(default=5, ge=2, le=20)
    mutation_temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    adaptive_mutation_enabled: bool = False
    adaptive_mutation_start_temperature: float = Field(default=0.9, ge=0.0, le=1.0)
    adaptive_mutation_end_temperature: float = Field(default=0.3, ge=0.0, le=1.0)
    crossover_enabled: bool = False
    crossover_rate: float = Field(
        default=DEFAULT_CROSSOVER_RATE,
        ge=MIN_CROSSOVER_RATE,
        le=MAX_CROSSOVER_RATE
    )
    crossover_temperature: float = Field(
        default=DEFAULT_CROSSOVER_TEMPERATURE,
        ge=MIN_CROSSOVER_TEMPERATURE,
        le=MAX_CROSSOVER_TEMPERATURE
    )
    crossover_parent_pool_size: int = Field(
        default=DEFAULT_CROSSOVER_PARENT_POOL_SIZE,
        ge=MIN_CROSSOVER_PARENT_POOL_SIZE,
        le=MAX_CROSSOVER_PARENT_POOL_SIZE
    )
    llm_model: str = "default"
    llm_timeout: int = Field(default=60, ge=10, le=300)
    early_stop_enabled: bool = True
    early_stop_sample_ratio: float = Field(default=0.2, ge=0.05, le=0.9)
    early_stop_accuracy_threshold: float = Field(default=0.2, ge=0.0, le=1.0)
    priority_accuracy: float = Field(default=DEFAULT_PRIORITY_ACCURACY, ge=0.0)
    priority_fn_reduction: float = Field(default=DEFAULT_PRIORITY_FN_REDUCTION, ge=0.0)
    priority_fp_reduction: float = Field(default=DEFAULT_PRIORITY_FP_REDUCTION, ge=0.0)
    priority_cost_reduction: float = Field(default=DEFAULT_PRIORITY_COST_REDUCTION, ge=0.0)
    mutation_system_prompt: str = DEFAULT_MUTATION_SYSTEM_PROMPT
    reflection_template: str = DEFAULT_REFLECTION_TEMPLATE
    improvement_template: str = DEFAULT_IMPROVEMENT_TEMPLATE
    crossover_template: str = DEFAULT_CROSSOVER_TEMPLATE

    @classmethod
    def from_profile(
        cls,
        profile: str,
        dataset_path: str,
        **overrides: Any,
    ) -> "OptimizationConfig":
        """Create config from a named profile with optional overrides."""
        if profile not in SUPPORTED_PROFILES:
            raise ValueError(
                f"Unknown profile '{profile}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_PROFILES))}"
            )
        defaults = dict(PROFILE_PRESETS.get(profile, {}))
        defaults["dataset_path"] = dataset_path
        defaults.update(overrides)
        return cls(**defaults)
