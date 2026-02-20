"""Prompt crossover via LLM synthesis of two prompts."""

from loguru import logger

from ..clients import BaseLLMClient
from ..models import OptimizationConfig

LENGTH_TOLERANCE_RATIO = 0.2
PERCENT_FACTOR = 100


class PromptCrossover:
    """Generate hybrid prompts by combining two strong prompts."""

    def __init__(self, llm_client: BaseLLMClient, config: OptimizationConfig):
        """Initialize crossover with LLM client and config templates."""
        self.llm = llm_client
        self.config = config

    def crossover_prompts(self, prompt_a: str, prompt_b: str, temperature: float) -> str:
        """Create a hybrid prompt combining the strongest parts of two prompts."""
        crossover_prompt = self.config.crossover_template.format(
            prompt_a=prompt_a,
            prompt_b=prompt_b,
            length_tolerance=int(LENGTH_TOLERANCE_RATIO * PERCENT_FACTOR)
        )
        response = self.llm.chat_completion(
            messages=[
                {"role": "system", "content": self.config.mutation_system_prompt},
                {"role": "user", "content": crossover_prompt},
            ],
            temperature=temperature,
        )
        result = response.strip()
        logger.debug(f"Crossover prompt ({len(result)} chars)")
        return result
