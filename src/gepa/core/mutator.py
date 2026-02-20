"""Prompt mutation via LLM reflection on failures."""

import json
from typing import Callable, Dict, List, Optional

from loguru import logger

from ..clients import BaseLLMClient
from ..models import DatasetEntry, OptimizationConfig, PromptCandidate

MAX_FAILURES_FOR_REFLECTION = 5
REFLECTION_TEMPERATURE = 0.5

FailureFormatFn = Callable[[DatasetEntry], str]


def default_failure_format_fn(entry: DatasetEntry) -> str:
    """Format failure as JSON representation of input + expected."""
    return (
        f"Input: {json.dumps(entry.input, ensure_ascii=False)}\n"
        f"Expected: {entry.expected}"
    )


class PromptMutator:
    """Generates improved prompts through LLM reflection on failures."""

    def __init__(
        self,
        llm_client: BaseLLMClient,
        config: OptimizationConfig,
        failure_format_fn: Optional[FailureFormatFn] = None
    ):
        """Initialize mutator with LLM client and config templates."""
        self.llm = llm_client
        self.config = config
        self.failure_format_fn = failure_format_fn or default_failure_format_fn

    def mutate_prompt(
        self,
        candidate: PromptCandidate,
        failed_examples: List[DatasetEntry],
        generation: int,
        temperature: float = 0.7
    ) -> str:
        """Generate improved prompt by analyzing failures."""
        improved_prompt, _ = self.mutate_prompt_with_reflection(
            candidate=candidate,
            failed_examples=failed_examples,
            generation=generation,
            temperature=temperature
        )
        return improved_prompt

    def mutate_prompt_with_reflection(
        self,
        candidate: PromptCandidate,
        failed_examples: List[DatasetEntry],
        generation: int,
        temperature: float = 0.7
    ) -> tuple[str, str]:
        """Generate improved prompt and reflection."""
        logger.debug(
            f"Mutating prompt {candidate.id} with {len(failed_examples)} failures"
        )
        reflection = self._reflect_on_failures(
            candidate.text,
            failed_examples[:MAX_FAILURES_FOR_REFLECTION]
        )
        improved_prompt = self._generate_improved_prompt(
            original_prompt=candidate.text,
            reflection=reflection,
            temperature=temperature
        )
        logger.debug(f"Generated improved prompt ({len(improved_prompt)} chars)")
        return improved_prompt, reflection

    def _reflect_on_failures(
        self,
        prompt_text: str,
        failed_examples: List[DatasetEntry]
    ) -> str:
        """Analyze why prompt failed using LLM meta-reasoning."""
        failures_text = self._format_failures(failed_examples)
        reflection_prompt = self.config.reflection_template.format(
            prompt_text=prompt_text,
            failures_text=failures_text
        )

        response = self.llm.chat_completion(
            messages=[
                {"role": "system", "content": self.config.mutation_system_prompt},
                {"role": "user", "content": reflection_prompt}
            ],
            temperature=REFLECTION_TEMPERATURE
        )

        logger.debug(f"Reflection: {response[:200]}...")
        return response

    def _generate_improved_prompt(
        self,
        original_prompt: str,
        reflection: str,
        temperature: float
    ) -> str:
        """Generate improved prompt incorporating reflection insights."""
        mutation_prompt = self.config.improvement_template.format(
            original_prompt=original_prompt,
            reflection=reflection
        )

        improved = self.llm.chat_completion(
            messages=[
                {"role": "system", "content": self.config.mutation_system_prompt},
                {"role": "user", "content": mutation_prompt}
            ],
            temperature=temperature
        )

        return improved.strip()

    def _format_failures(self, failed_examples: List[DatasetEntry]) -> str:
        """Format failures for LLM reflection analysis."""
        failures_parts = []

        for i, entry in enumerate(failed_examples[:MAX_FAILURES_FOR_REFLECTION], 1):
            formatted = self.failure_format_fn(entry)
            failures_parts.append(f"--- Example {i} ---\n{formatted}")

        return "\n\n".join(failures_parts)
