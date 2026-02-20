"""Post-optimization error analysis."""

import json
from typing import Dict, List

from loguru import logger

from ..clients import BaseLLMClient
from ..models import DatasetEntry, PromptCandidate

ANALYSIS_TEMPERATURE = 0.3
RECOMMENDATION_TEMPERATURE = 0.5
MAX_ERROR_EXAMPLES = 10
MAX_REPRESENTATIVE_EXAMPLES = 5


class ErrorExample:
    """Single error case from evaluation."""

    def __init__(
        self,
        input_data: Dict,
        expected: str,
        predicted: str,
        criterion: str
    ):
        """Initialize error example with prediction data."""
        self.input_data = input_data
        self.expected = expected
        self.predicted = predicted
        self.criterion = criterion

    @property
    def error_type(self) -> str:
        """Classify error type based on prediction mismatch."""
        expected_lower = self.expected.strip().lower()
        predicted_lower = self.predicted.strip().lower()
        if expected_lower in {"true", "1", "yes"} and predicted_lower in {"false", "0", "no"}:
            return "false_negative"
        if expected_lower in {"false", "0", "no"} and predicted_lower in {"true", "1", "yes"}:
            return "false_positive"
        if expected_lower != predicted_lower:
            return "mismatch"
        return "correct"


class ErrorAnalyzer:
    """Analyzes remaining errors after optimization."""

    def __init__(self, llm_client: BaseLLMClient):
        """Initialize analyzer with LLM client for meta-analysis."""
        self.llm_client = llm_client

    def analyze_errors(
        self,
        candidate: PromptCandidate,
        dataset_path: str,
        criterion_name: str
    ) -> Dict:
        """Analyze errors and generate improvement recommendations."""
        logger.info("Starting post-optimization error analysis...")

        examples = self._load_dataset(dataset_path)
        errors = self._collect_errors(candidate, examples, criterion_name)
        error_stats = self._categorize_errors(errors)
        patterns = self._analyze_patterns(errors, criterion_name)
        recommendations = self._generate_recommendations(
            errors,
            patterns,
            criterion_name
        )

        return {
            "total_errors": len(errors),
            "error_stats": error_stats,
            "patterns": patterns,
            "recommendations": recommendations,
            "example_errors": self._select_representative_errors(errors)
        }

    def _load_dataset(self, dataset_path: str) -> List[Dict]:
        """Load dataset from JSONL file."""
        examples = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))
        return examples

    def _collect_errors(
        self,
        candidate: PromptCandidate,
        examples: List[Dict],
        criterion_name: str
    ) -> List[ErrorExample]:
        """Evaluate candidate and collect all misclassified examples."""
        errors = []
        logger.info(f"Evaluating {len(examples)} examples to collect errors...")

        for example in examples:
            input_data = example.get("input", example.get("record", {}))
            expected = example.get("expected", "")
            if not expected and "expected_validation" in example:
                expected = str(example["expected_validation"].get(criterion_name, ""))

            predicted = self._get_prediction(candidate.text, input_data)

            error_ex = ErrorExample(
                input_data=input_data,
                expected=expected,
                predicted=predicted,
                criterion=criterion_name
            )

            if error_ex.error_type != "correct":
                errors.append(error_ex)

        logger.info(f"Collected {len(errors)} errors out of {len(examples)} examples")
        return errors

    def _get_prediction(self, prompt_text: str, input_data: Dict) -> str:
        """Get model prediction for single input."""
        input_text = json.dumps(input_data, ensure_ascii=False)
        messages = [
            {"role": "system", "content": prompt_text},
            {"role": "user", "content": input_text}
        ]

        response = self.llm_client.chat_completion(messages, temperature=0.0)
        return response.strip()

    def _categorize_errors(self, errors: List[ErrorExample]) -> Dict:
        """Categorize errors by type for statistics."""
        stats: Dict[str, int] = {
            "false_negatives": 0,
            "false_positives": 0,
            "mismatch": 0
        }

        for error in errors:
            error_type = error.error_type
            if error_type == "false_negative":
                stats["false_negatives"] += 1
            elif error_type == "false_positive":
                stats["false_positives"] += 1
            else:
                stats["mismatch"] += 1

        return stats

    def _analyze_patterns(
        self,
        errors: List[ErrorExample],
        criterion_name: str
    ) -> str:
        """Use LLM to identify common error patterns."""
        if not errors:
            return "No errors to analyze"

        error_descriptions = []
        for i, error in enumerate(errors[:MAX_ERROR_EXAMPLES], 1):
            input_summary = json.dumps(error.input_data, ensure_ascii=False)[:200]
            error_descriptions.append(
                f"Error {i} ({error.error_type}):\n"
                f"Expected: {error.expected}\n"
                f"Predicted: {error.predicted}\n"
                f"Input: {input_summary}...\n"
            )

        prompt = (
            f'Analyze the model errors for criterion "{criterion_name}".\n\n'
            f"Here are the errors:\n\n"
            f"{chr(10).join(error_descriptions)}\n"
            f"Find common patterns in these errors. "
            f"What do the cases where the model fails have in common?\n"
            f"Reply concisely, 2-3 sentences."
        )

        messages = [{"role": "user", "content": prompt}]
        response = self.llm_client.chat_completion(
            messages,
            temperature=ANALYSIS_TEMPERATURE
        )
        return response.strip()

    def _generate_recommendations(
        self,
        errors: List[ErrorExample],
        patterns: str,
        criterion_name: str
    ) -> str:
        """Generate actionable recommendations for improvement."""
        if not errors:
            return "Model works perfectly, no recommendations needed"

        prompt = (
            f"Based on the error analysis:\n\n"
            f"Criterion: {criterion_name}\n"
            f"Total errors: {len(errors)}\n"
            f"Error patterns: {patterns}\n\n"
            f"What does the model need to improve accuracy?\n"
            f"Consider:\n"
            f"1. Are external data sources needed?\n"
            f"2. Is additional context needed?\n"
            f"3. Are domain-specific knowledge requirements missing?\n"
            f"4. Other prompt limitations?\n\n"
            f"Give specific recommendations (3-5 points)."
        )

        messages = [{"role": "user", "content": prompt}]
        response = self.llm_client.chat_completion(
            messages,
            temperature=RECOMMENDATION_TEMPERATURE
        )
        return response.strip()

    def _select_representative_errors(
        self,
        errors: List[ErrorExample],
        max_examples: int = MAX_REPRESENTATIVE_EXAMPLES
    ) -> List[Dict]:
        """Select diverse error examples for display."""
        by_type: Dict[str, List[ErrorExample]] = {}
        for error in errors:
            error_type = error.error_type
            if error_type not in by_type:
                by_type[error_type] = []
            by_type[error_type].append(error)

        selected = []
        for error_type, error_list in by_type.items():
            if error_list:
                error = error_list[0]
                input_summary = json.dumps(error.input_data, ensure_ascii=False)[:200]
                selected.append({
                    "error_type": error_type,
                    "expected": error.expected,
                    "predicted": error.predicted,
                    "input_summary": input_summary
                })

        return selected[:max_examples]
