"""Convert medical records to GEPA dataset format.

Reads raw medical consultation records and converts them to GEPA's
DatasetEntry JSONL format: {"input": {...}, "expected": "True"/"False"}.

Usage:
    python convert_data.py --input golden_consultations.jsonl \
                           --output data/validation/icd10_matches_diagnosis.jsonl \
                           --criterion icd_code_matches_diagnosis \
                           --dataset-size 50
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

LABEL_TRUE = 1
LABEL_FALSE = 0


def _parse_protocol_fields(protocol_json: str) -> Optional[dict]:
    """Parse protocol JSON string into a flat dict of fields.

    Adapt this function to your data format. This example expects
    a JSON string with medical protocol fields.
    """
    try:
        data = json.loads(protocol_json)
    except (json.JSONDecodeError, TypeError):
        return None

    if isinstance(data, dict):
        return data
    return None


def _convert_entry(entry: dict) -> Optional[dict]:
    """Convert a raw medical record to GEPA DatasetEntry format."""
    protocol_fields = _parse_protocol_fields(entry.get("protocol_json", "{}"))
    if not protocol_fields:
        logging.warning(
            f"Failed to parse entry: {entry.get('protocol_id')}"
        )
        return None

    label_map = {0: "False", 1: "True"}
    expected = label_map.get(entry.get("label"), "False")

    return {
        "input": {
            "record_id": str(entry.get("protocol_id", "")),
            "symptoms": protocol_fields.get("Anamnesis", ""),
            "diagnosis": protocol_fields.get("Diagnosis", ""),
            "treatment": protocol_fields.get("Recommendation", ""),
            "additional_info": protocol_fields.get("AdditionalInfo", ""),
        },
        "expected": expected,
    }


def _load_entries(source_path: Path) -> List[dict]:
    """Load JSONL entries from a file."""
    entries: List[dict] = []
    with open(source_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def _split_by_label(
    entries: Iterable[dict], criterion: str
) -> Tuple[List[dict], List[dict]]:
    """Split entries into positive and negative lists for a criterion."""
    positive: List[dict] = []
    negative: List[dict] = []
    for entry in entries:
        if entry.get("criterion_code") != criterion:
            continue
        if entry.get("label") == LABEL_TRUE:
            positive.append(entry)
        elif entry.get("label") == LABEL_FALSE:
            negative.append(entry)
    return positive, negative


def _select_balanced(
    positive: List[dict], negative: List[dict], target_size: int
) -> List[dict]:
    """Select a balanced subset of entries (equal positive/negative)."""
    target_per_class = max(1, target_size // 2)
    return positive[:target_per_class] + negative[:target_per_class]


def _convert_entries(entries: Iterable[dict]) -> List[dict]:
    """Convert raw entries to GEPA DatasetEntry format."""
    converted: List[dict] = []
    for entry in entries:
        gepa_entry = _convert_entry(entry)
        if gepa_entry:
            converted.append(gepa_entry)
    return converted


def _write_jsonl(path: Path, items: Iterable[dict]) -> None:
    """Write items to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def convert_dataset(
    source_path: Path,
    output_path: Path,
    criterion: str,
    target_size: int,
) -> int:
    """Convert dataset and return number of written entries."""
    entries = _load_entries(source_path)
    positive, negative = _split_by_label(entries, criterion)
    selected = _select_balanced(positive, negative, target_size)
    converted = _convert_entries(selected)
    _write_jsonl(output_path, converted)
    return len(converted)


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Convert medical records to GEPA DatasetEntry format."
    )
    parser.add_argument(
        "--input",
        default="golden_consultations.jsonl",
        help="Path to source JSONL file with labeled medical records",
    )
    parser.add_argument(
        "--output",
        default="data/validation/dataset.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument(
        "--criterion",
        default="icd_code_matches_diagnosis",
        help="Criterion code to filter by",
    )
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=50,
        help="Target dataset size (balanced between positive/negative)",
    )
    return parser.parse_args()


def main() -> None:
    """Run dataset conversion."""
    args = _parse_args()
    count = convert_dataset(
        source_path=Path(args.input),
        output_path=Path(args.output),
        criterion=args.criterion,
        target_size=args.dataset_size,
    )
    print(f"Converted {count} entries to {args.output}")


if __name__ == "__main__":
    main()
