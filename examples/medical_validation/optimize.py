"""Optimize medical validation prompts using GEPA.

This example demonstrates a complex eval_fn that calls an external /validate API.
The complexity comes entirely from the evaluation pipeline (protocol building,
API calls, response parsing), not from GEPA itself.

Prerequisites:
    - A running validation API at http://localhost:8000/validate
    - A local LLM endpoint (SGLang/vLLM) for GEPA mutations
    - Source dataset: evaluation/data/golden_consultations.jsonl

Usage:
    # Single criterion
    CRITERION=icd10_matches_diagnosis python optimize.py

    # All criteria
    python optimize.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import asyncio
import os
import threading
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import httpx
from loguru import logger
from tqdm import tqdm

from gepa import GEPAOptimizer, LLMClient, OptimizationConfig
from gepa.config import Settings as GEPASettings

CRITERIA = [
    "icd10_matches_diagnosis",
    "diagnosis_matches_examination",
    "treatment_matches_diagnosis",
    "examination_plan_matches_diagnosis",
    "medications_prescribed_correctly",
]

CRITERION_NAME_MAP = {
    "icd10_matches_diagnosis": "icd_code_matches_diagnosis",
}

VALUE_MAP = {1: "True", 0: "False", -1: "NotEnough"}

API_URL = "http://localhost:8000/validate"

SGLANG_URL = os.getenv("SGLANG_URL", "http://localhost:8000/v1")
SGLANG_MODEL = os.getenv("SGLANG_MODEL", "default")
DATASET_SIZE = int(os.getenv("GEPA_DATASET_SIZE", "50"))
NUM_GENERATIONS = int(os.getenv("GEPA_NUM_GENERATIONS", "10"))
POPULATION_SIZE = int(os.getenv("GEPA_POPULATION_SIZE", "5"))
LLM_MAX_CONCURRENT = int(os.getenv("LLM_MAX_CONCURRENT", "5"))


def get_dataset_criterion_name(criterion: str) -> str:
    """Get the criterion name as it appears in the dataset."""
    return CRITERION_NAME_MAP.get(criterion, criterion)


def _build_protocol(input_data: Dict) -> Dict:
    """Build Protocol dict from DatasetEntry input."""
    record_id = input_data.get("record_id", "")
    try:
        protocol_uuid = str(uuid.UUID(int=int(record_id)))
    except (ValueError, TypeError):
        try:
            protocol_uuid = str(uuid.UUID(record_id))
        except (ValueError, TypeError):
            protocol_uuid = str(uuid.uuid4())

    return {
        "Id": protocol_uuid,
        "ProtocolType": 0,
        "Anamnesis": input_data.get("symptoms"),
        "Diagnosis": input_data.get("diagnosis"),
        "Recommendation": input_data.get("treatment"),
        "AdditionalInfo": input_data.get("additional_info"),
        "HistoricalAnamnesis": None,
        "ClinicalRecommendations": None,
    }


def _extract_result(response_data: Dict, criterion_name: str) -> str:
    """Extract criterion result string from /validate response."""
    criteria = response_data.get("Criteria", {})
    for key, value in criteria.items():
        if key.lower() == criterion_name.lower().replace("_", ""):
            return VALUE_MAP.get(value.get("Value", -1), "NotEnough")
    logger.error(
        f"Criterion '{criterion_name}' not found in response: "
        f"{list(criteria.keys())}"
    )
    return "NotEnough"


class _EvalProgressBar:
    """Thread-safe tqdm wrapper for eval_fn request tracking."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pbar: Optional[tqdm] = None
        self._completed = 0

    def reset(self, total: int) -> None:
        """Reset progress bar with new total."""
        with self._lock:
            if self._pbar:
                self._pbar.close()
            self._completed = 0
            self._pbar = tqdm(
                total=total,
                desc="Eval requests",
                unit="req",
                bar_format=(
                    "{l_bar}{bar}| {n_fmt}/{total_fmt} "
                    "[{elapsed}<{remaining}, {rate_fmt}]"
                ),
                dynamic_ncols=True,
            )

    def increment(self) -> None:
        """Increment completed count."""
        with self._lock:
            self._completed += 1
            if self._pbar:
                self._pbar.n = self._completed
                self._pbar.refresh()

    def close(self) -> None:
        """Close progress bar."""
        with self._lock:
            if self._pbar:
                self._pbar.close()
                self._pbar = None


_eval_progress = _EvalProgressBar()


def make_eval_fn(criterion_name: str, max_concurrent: int = 5):
    """Create eval_fn for GEPA that calls /validate endpoint.

    This is the core of the complex example. The eval_fn:
    1. Builds a Protocol dict from the flat input_data
    2. POSTs to /validate with the prompt as criterion description
    3. Parses the JSON response and extracts the criterion result
    4. Returns "True", "False", or "NotEnough" as a string

    GEPA compares this return value with `expected` from the dataset.
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def eval_fn(prompt_text: str, input_data: Dict) -> str:
        async with semaphore:
            protocol = _build_protocol(input_data)
            request = {
                "Protocol": protocol,
                "criteria_descriptions": {criterion_name: prompt_text},
            }
            try:
                async with httpx.AsyncClient(timeout=600) as client:
                    resp = await client.post(API_URL, json=request)
                    resp.raise_for_status()
                return _extract_result(resp.json(), criterion_name)
            except Exception as e:
                logger.error(f"eval_fn error: {type(e).__name__}: {e}")
                return "NotEnough"
            finally:
                _eval_progress.increment()

    return eval_fn


def find_latest_incomplete_run(criterion: str) -> Optional[str]:
    """Find the latest incomplete run for a criterion to resume from."""
    runs_dir = Path(f"data/gepa/runs/{criterion}")
    if not runs_dir.exists():
        return None

    run_dirs = sorted(runs_dir.glob("gepa_run_*"), reverse=True)
    for run_dir in run_dirs:
        state_file = run_dir / "state.json"
        if state_file.exists():
            logger.info(f"Found incomplete run: {run_dir}")
            return str(run_dir)

    return None


def _build_gepa_llm_client() -> LLMClient:
    """Create GEPA LLMClient pointing at local LLM endpoint."""
    gepa_settings = GEPASettings(
        api_key="local",
        base_url=SGLANG_URL,
        model=SGLANG_MODEL,
    )
    return LLMClient(gepa_settings)


def _load_baseline_prompt(criterion: str) -> str:
    """Load baseline prompt for a criterion from YAML.

    Adapt this function to your project's prompt storage.
    Here we show the pattern — you would replace this with your
    own prompt loading logic.
    """
    prompt_dir = Path("prompts")
    prompt_file = prompt_dir / f"{criterion}.txt"
    if prompt_file.exists():
        return prompt_file.read_text(encoding="utf-8").strip()
    raise FileNotFoundError(
        f"Baseline prompt not found: {prompt_file}. "
        f"Create it or modify _load_baseline_prompt()."
    )


def optimize_criterion(criterion: str) -> Dict:
    """Optimize prompt for a single validation criterion."""
    logger.info(f"Starting optimization for criterion: {criterion}")

    dataset_criterion = get_dataset_criterion_name(criterion)
    dataset_path = Path(f"data/validation/{criterion}.jsonl")
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}. "
            f"Run convert_data.py first."
        )

    baseline_prompt = _load_baseline_prompt(criterion)
    logger.info(f"Loaded baseline prompt ({len(baseline_prompt)} chars)")

    runs_dir = Path(f"data/gepa/runs/{criterion}")
    runs_dir.mkdir(parents=True, exist_ok=True)

    config = OptimizationConfig(
        dataset_path=str(dataset_path),
        runs_dir=str(runs_dir),
        criterion_name=dataset_criterion,
        num_generations=NUM_GENERATIONS,
        population_size=POPULATION_SIZE,
        mutation_temperature=0.7,
        early_stop_enabled=True,
    )

    resume_from = find_latest_incomplete_run(criterion)
    if resume_from:
        logger.info(f"Resuming from previous run: {resume_from}")

    llm_client = _build_gepa_llm_client()
    eval_fn = make_eval_fn(criterion, max_concurrent=LLM_MAX_CONCURRENT)

    total_evals = (1 + config.num_generations * config.population_size) * DATASET_SIZE
    _eval_progress.reset(total_evals)

    optimizer = GEPAOptimizer(
        llm_client=llm_client,
        config=config,
        eval_fn=eval_fn,
    )

    try:
        result = optimizer.optimize(baseline_prompt, resume_from=resume_from)
    finally:
        _eval_progress.close()

    baseline_acc = result.baseline_metrics.accuracy
    optimized_acc = result.recommended_prompt.metrics.accuracy
    improvement = (optimized_acc - baseline_acc) * 100

    logger.success(
        f"\n{'='*60}\n"
        f"Optimization complete: {criterion}\n"
        f"{'='*60}\n"
        f"Baseline accuracy:  {baseline_acc:.1%}\n"
        f"Optimized accuracy: {optimized_acc:.1%}\n"
        f"Improvement:        {improvement:+.1f}%\n"
        f"FN rate:            {result.recommended_prompt.metrics.false_negative_rate:.1%}\n"
        f"FP rate:            {result.recommended_prompt.metrics.false_positive_rate:.1%}\n"
        f"{'='*60}"
    )

    output_file = Path(f"data/gepa/optimized_{criterion}.txt")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(result.recommended_prompt.text, encoding="utf-8")
    logger.info(f"Optimized prompt saved to: {output_file}")

    return {
        "criterion": criterion,
        "baseline_accuracy": baseline_acc,
        "optimized_accuracy": optimized_acc,
        "improvement": improvement,
        "output_file": str(output_file),
    }


def plot_results(results: List[Dict]) -> None:
    """Plot baseline vs optimized accuracy per criterion."""
    if not results:
        return

    criteria = [r["criterion"] for r in results]
    baseline = [r["baseline_accuracy"] * 100 for r in results]
    optimized = [r["optimized_accuracy"] * 100 for r in results]

    x = range(len(criteria))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(
        [i - width / 2 for i in x], baseline, width,
        label="Baseline", color="#e74c3c",
    )
    ax.bar(
        [i + width / 2 for i in x], optimized, width,
        label="Optimized", color="#2ecc71",
    )

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("GEPA Prompt Optimization Results")
    ax.set_xticks(list(x))
    ax.set_xticklabels(
        [c.replace("_", "\n") for c in criteria], fontsize=8,
    )
    ax.legend()
    ax.set_ylim(0, 100)
    plt.tight_layout()

    output_path = Path("data/gepa/plots/final_results.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    logger.info(f"Results plot saved to: {output_path}")


def main() -> None:
    """Optimize prompts for all validation criteria."""
    logger.info("GEPA Medical Validation Prompt Optimization")

    target_criterion = os.getenv("CRITERION")

    logger.info(
        f"Settings: {NUM_GENERATIONS} generations, "
        f"{POPULATION_SIZE} population, "
        f"{DATASET_SIZE} dataset size"
    )

    if target_criterion:
        criteria_to_run = [target_criterion]
        logger.info(f"Single criterion: {target_criterion}")
    else:
        criteria_to_run = CRITERIA
        logger.info("Running all criteria.")

    results: List[Dict] = []

    for criterion in criteria_to_run:
        logger.info(f"\n\n{'#'*60}\nOptimizing: {criterion}\n{'#'*60}\n")
        try:
            result = optimize_criterion(criterion)
            results.append(result)
        except Exception as e:
            logger.exception(f"Failed to optimize {criterion}: {e}")
            continue

    logger.success("\nAll optimizations complete!")
    for r in results:
        logger.info(
            f"  {r['criterion']}: "
            f"{r['baseline_accuracy']:.1%} -> {r['optimized_accuracy']:.1%} "
            f"(+{r['improvement']:.1f}%)"
        )

    plot_results(results)


if __name__ == "__main__":
    main()
