"""Prompt evaluation on dataset."""

import asyncio
import json
import random
import time
from pathlib import Path
from typing import Awaitable, Callable, Dict, List, Optional, Tuple

from loguru import logger

from ..clients import BaseLLMClient
from ..models import DatasetEntry, PromptMetrics

EvalFn = Callable[[str, Dict], Awaitable[str]]
CompareFn = Callable[[str, str], bool]

POSITIVE_LABELS = {"true", "1", "yes"}
NEGATIVE_LABELS = {"false", "0", "no"}


def default_compare_fn(predicted: str, expected: str) -> bool:
    """Compare predicted and expected by exact match (case-insensitive)."""
    return predicted.strip().lower() == expected.strip().lower()


class PromptEvaluator:
    """Evaluates prompt performance on dataset."""

    def __init__(
        self,
        llm_client: BaseLLMClient,
        eval_fn: Optional[EvalFn] = None,
        compare_fn: Optional[CompareFn] = None
    ):
        """Initialize evaluator with optional custom eval and compare functions."""
        self.llm = llm_client
        self.eval_fn = eval_fn
        self.compare_fn = compare_fn or default_compare_fn

    def load_dataset(self, dataset_path: Path) -> List[DatasetEntry]:
        """Load dataset from JSONL file."""
        entries: List[DatasetEntry] = []

        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    entry = DatasetEntry.model_validate(data)
                    entries.append(entry)

        logger.info(f"Loaded {len(entries)} entries from {dataset_path}")
        return entries

    async def evaluate_prompt(
        self,
        prompt_text: str,
        dataset: List[DatasetEntry],
        criterion: str = "default"
    ) -> Tuple[PromptMetrics, List[DatasetEntry]]:
        """Evaluate prompt on dataset."""
        logger.info(f"Evaluating prompt on {len(dataset)} examples...")
        start_time = time.time()

        tasks = [
            self._evaluate_single(prompt_text, entry)
            for entry in dataset
        ]
        results = await asyncio.gather(*tasks)

        total = len(results)
        correct = 0
        false_negatives = 0
        false_positives = 0
        not_enough_count = 0
        latencies: List[float] = []
        failed_examples: List[DatasetEntry] = []

        for entry, predicted, is_correct, latency_ms in results:
            latencies.append(latency_ms)

            if is_correct:
                correct += 1
            else:
                failed_examples.append(entry)
                fn_fp = self._classify_error(predicted, entry.expected)
                if fn_fp == "fn":
                    false_negatives += 1
                elif fn_fp == "fp":
                    false_positives += 1
                else:
                    not_enough_count += 1

        accuracy = correct / total if total > 0 else 0.0
        fn_rate = false_negatives / total if total > 0 else 0.0
        fp_rate = false_positives / total if total > 0 else 0.0
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        cost_tokens = self.llm.count_tokens(prompt_text) if hasattr(self.llm, "count_tokens") else len(prompt_text) // 4

        metrics = PromptMetrics(
            accuracy=accuracy,
            false_negative_rate=fn_rate,
            false_positive_rate=fp_rate,
            cost_tokens=cost_tokens,
            latency_ms=avg_latency,
            total_examples=total,
            correct=correct,
            false_negatives=false_negatives,
            false_positives=false_positives,
            not_enough_count=not_enough_count
        )

        elapsed = time.time() - start_time
        logger.info(f"Evaluation complete in {elapsed:.1f}s: {metrics}")
        return metrics, failed_examples

    async def evaluate_prompt_with_early_stopping(
        self,
        prompt_text: str,
        dataset: List[DatasetEntry],
        criterion: str = "default",
        sample_ratio: float = 0.2,
        accuracy_threshold: float = 0.2
    ) -> Tuple[PromptMetrics, List[DatasetEntry]]:
        """Evaluate prompt with early stopping on a sample."""
        if not dataset:
            return await self.evaluate_prompt(prompt_text, dataset, criterion)

        sample_size = max(1, int(len(dataset) * sample_ratio))
        if sample_size >= len(dataset):
            return await self.evaluate_prompt(prompt_text, dataset, criterion)

        sample_indices = set(random.sample(range(len(dataset)), sample_size))
        sample = [dataset[i] for i in sample_indices]
        remaining = [entry for i, entry in enumerate(dataset) if i not in sample_indices]

        sample_metrics, sample_failed = await self.evaluate_prompt(prompt_text, sample, criterion)
        if sample_metrics.accuracy < accuracy_threshold:
            logger.info(
                f"Early stopping: accuracy {sample_metrics.accuracy:.2%} "
                f"below threshold {accuracy_threshold:.2%}"
            )
            return sample_metrics, sample_failed

        remaining_metrics, remaining_failed = await self.evaluate_prompt(prompt_text, remaining, criterion)
        combined_metrics = self._combine_metrics(sample_metrics, remaining_metrics, prompt_text)
        combined_failed = sample_failed + remaining_failed
        return combined_metrics, combined_failed

    def _classify_error(self, predicted: str, expected: str) -> str:
        """Classify error as false_negative, false_positive, or other."""
        expected_lower = expected.strip().lower()
        predicted_lower = predicted.strip().lower()

        expected_positive = expected_lower in POSITIVE_LABELS
        expected_negative = expected_lower in NEGATIVE_LABELS

        if expected_positive and predicted_lower in NEGATIVE_LABELS:
            return "fn"
        if expected_negative and predicted_lower in POSITIVE_LABELS:
            return "fp"
        if expected_positive or expected_negative:
            return "fn" if expected_positive else "fp"
        return "other"

    def _combine_metrics(
        self,
        first: PromptMetrics,
        second: PromptMetrics,
        prompt_text: str
    ) -> PromptMetrics:
        """Combine metrics from two evaluations."""
        total = first.total_examples + second.total_examples
        correct = first.correct + second.correct
        false_negatives = first.false_negatives + second.false_negatives
        false_positives = first.false_positives + second.false_positives
        not_enough_count = first.not_enough_count + second.not_enough_count

        accuracy = correct / total if total > 0 else 0.0
        fn_rate = false_negatives / total if total > 0 else 0.0
        fp_rate = false_positives / total if total > 0 else 0.0
        latency_ms = (
            (first.latency_ms * first.total_examples + second.latency_ms * second.total_examples) / total
            if total > 0 else 0.0
        )
        cost_tokens = self.llm.count_tokens(prompt_text) if hasattr(self.llm, "count_tokens") else len(prompt_text) // 4

        return PromptMetrics(
            accuracy=accuracy,
            false_negative_rate=fn_rate,
            false_positive_rate=fp_rate,
            cost_tokens=cost_tokens,
            latency_ms=latency_ms,
            total_examples=total,
            correct=correct,
            false_negatives=false_negatives,
            false_positives=false_positives,
            not_enough_count=not_enough_count
        )

    async def _evaluate_single(
        self,
        prompt_text: str,
        entry: DatasetEntry
    ) -> Tuple[DatasetEntry, str, bool, float]:
        """Evaluate prompt on single example."""
        start_time = time.time()

        try:
            if self.eval_fn:
                predicted = await self.eval_fn(prompt_text, entry.input)
            else:
                predicted = await self._default_eval(prompt_text, entry.input)

            latency_ms = (time.time() - start_time) * 1000
            is_correct = self.compare_fn(predicted, entry.expected)
            return entry, predicted, is_correct, latency_ms

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            latency_ms = (time.time() - start_time) * 1000
            return entry, "", False, latency_ms

    async def _default_eval(self, prompt_text: str, input_data: Dict) -> str:
        """Default evaluation: substitute placeholders and send to LLM."""
        formatted_prompt = prompt_text.format(**input_data)
        response = await self.llm.achat_completion(
            messages=[{"role": "user", "content": formatted_prompt}],
            temperature=0.0
        )
        return response.strip()
