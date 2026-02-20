"""Mutation logging utilities."""

import json
from pathlib import Path
from typing import List, Optional

from ...models import DatasetEntry, PromptCandidate

MUTATION_LOG_FILENAME = "mutation_log.jsonl"
MUTATION_LOG_MAX_FAILURES = 5
MUTATION_LOG_MAX_TEXT_LENGTH = 200


class MutationLogger:
    """Write mutation logs with failure details."""

    def __init__(self, runs_dir: Path):
        """Initialize mutation logger."""
        self.runs_dir = runs_dir
        self.run_id: Optional[str] = None

    def set_run_id(self, run_id: str) -> None:
        """Set current run id."""
        self.run_id = run_id

    def append(
        self,
        candidate: PromptCandidate,
        parent: PromptCandidate,
        generation: int,
        reflection: str,
        failed_examples: List[DatasetEntry],
        mutation_kind: str
    ) -> None:
        """Append mutation log entry."""
        run_dir = self.runs_dir / (self.run_id or "gepa_run")
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / MUTATION_LOG_FILENAME
        failures = [
            self._format_failure(entry)
            for entry in failed_examples[:MUTATION_LOG_MAX_FAILURES]
        ]
        payload = {
            "run_id": self.run_id,
            "generation": generation,
            "mutation_kind": mutation_kind,
            "parent_id": parent.id,
            "candidate_id": candidate.id,
            "parent_accuracy": parent.metrics.accuracy,
            "candidate_accuracy": candidate.metrics.accuracy,
            "reflection": reflection.strip(),
            "top_failures": failures
        }
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _format_failure(self, entry: DatasetEntry) -> dict:
        """Format a failure example for logs."""
        truncated_input = {}
        for key, value in entry.input.items():
            str_value = str(value)
            truncated_input[key] = str_value[:MUTATION_LOG_MAX_TEXT_LENGTH]
        return {
            "input": truncated_input,
            "expected": entry.expected
        }
