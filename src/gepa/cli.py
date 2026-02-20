"""Command-line interface for GEPA."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    os.environ["PYTHONIOENCODING"] = "utf-8"

import yaml
from loguru import logger

from .models.config import PROFILE_PRESETS, SUPPORTED_PROFILES

GEPA_YAML = "gepa.yaml"
DEFAULT_PROFILE = "balanced"
REQUIRED_CONFIG_FIELDS = ("prompt", "dataset", "model")

EXAMPLE_CONFIG = """\
# GEPA Configuration
# API key: set GEPA_API_KEY or OPENAI_API_KEY in environment
# For local models (SGLang, vLLM, Ollama) set base_url — no API key needed.

# Required
prompt: prompt.txt
dataset: dataset.jsonl
model: default

# Local model endpoint (uncomment for local inference)
# base_url: http://localhost:8000/v1

# Profile: fast | balanced | quality | advanced
#   fast     - 4 generations, large population, high creativity
#   balanced - 8 generations, moderate settings (default)
#   quality  - 12 generations, adaptive mutation, strict early stop
#   advanced - no presets, you control every parameter
profile: balanced

# Optional overrides (any value below overrides the profile default)
# generations: 8
# population_size: 10
# temperature: 0.7
# criterion: default
# runs_dir: runs
"""

EXAMPLE_DATASET = [
    {"input": {"text": "This product is amazing, I love it!"}, "expected": "positive"},
    {"input": {"text": "Terrible experience, would not recommend."}, "expected": "negative"},
    {"input": {"text": "It works fine, nothing special."}, "expected": "neutral"},
]

EXAMPLE_PROMPT = "Classify the sentiment of the following text as positive, negative, or neutral.\n\nText: {text}\n\nSentiment:"


def main() -> None:
    """GEPA CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="gepa",
        description="GEPA - Genetic-Pareto Prompt Optimizer",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("init", help="Create example project files")

    run_parser = subparsers.add_parser("run", help="Run prompt optimization")
    run_parser.add_argument("--prompt", type=str, help="Path to baseline prompt file")
    run_parser.add_argument("--dataset", type=str, help="Path to dataset JSONL file")
    run_parser.add_argument("--generations", type=int, help="Number of generations")
    run_parser.add_argument("--population-size", type=int, help="Population size")
    run_parser.add_argument("--model", type=str, help="LLM model name")
    run_parser.add_argument(
        "--profile",
        type=str,
        choices=sorted(SUPPORTED_PROFILES),
        help="Optimization profile: fast|balanced|quality|advanced",
    )
    run_parser.add_argument("--base-url", type=str, help="OpenAI-compatible API base URL")
    run_parser.add_argument("--api-key", type=str, help="API key (or set OPENAI_API_KEY)")
    run_parser.add_argument("--criterion", type=str, help="Criterion name")
    run_parser.add_argument("--resume", type=str, help="Resume from a previous run directory")
    run_parser.add_argument("--config", type=str, default=GEPA_YAML, help="Config file path")

    args = parser.parse_args()

    if args.command == "init":
        cmd_init()
    elif args.command == "run":
        cmd_run(args)
    else:
        parser.print_help()


def cmd_init() -> None:
    """Create example GEPA project files."""
    cwd = Path.cwd()

    files = {
        GEPA_YAML: EXAMPLE_CONFIG,
        "prompt.txt": EXAMPLE_PROMPT,
        "dataset.jsonl": "\n".join(
            json.dumps(entry, ensure_ascii=False) for entry in EXAMPLE_DATASET
        ) + "\n",
    }

    for filename, content in files.items():
        filepath = cwd / filename
        if filepath.exists():
            logger.warning(f"Skipped (already exists): {filename}")
            continue
        filepath.write_text(content, encoding="utf-8")
        logger.success(f"Created: {filename}")

    print("\nProject initialized! Next steps:")
    print("  1. Edit gepa.yaml — set model and base_url (for local) or API key (for cloud)")
    print("  2. Edit prompt.txt with your baseline prompt")
    print("  3. Replace dataset.jsonl with your data")
    print("  4. Run: gepa run")


def cmd_run(args: argparse.Namespace) -> None:
    """Run GEPA optimization."""
    from .clients import LLMClient
    from .config import Settings
    from .core import GEPAOptimizer

    config_data = _load_yaml_config(args.config)
    try:
        effective = _merge_config(config_data, args)
        _validate_required_config(effective)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    api_key = (
        effective.get("api_key")
        or os.environ.get("GEPA_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or ""
    )
    base_url = effective.get("base_url")
    if not api_key:
        if base_url:
            api_key = "local"
            logger.info("Using local endpoint without API key.")
        else:
            logger.error("No API key. Set GEPA_API_KEY / OPENAI_API_KEY or use --api-key.")
            sys.exit(1)

    prompt_path = Path(effective["prompt"])
    if not prompt_path.exists():
        logger.error(f"Prompt file not found: {prompt_path}")
        sys.exit(1)
    baseline_prompt = prompt_path.read_text(encoding="utf-8").strip()
    logger.info(f"Loaded prompt from {prompt_path} ({len(baseline_prompt)} chars)")

    dataset_path = Path(effective["dataset"])
    if not dataset_path.exists():
        logger.error(f"Dataset file not found: {dataset_path}")
        sys.exit(1)

    runs_dir = effective.get("runs_dir", "runs")
    Path(runs_dir).mkdir(parents=True, exist_ok=True)

    opt_config = _build_optimization_config(effective, dataset_path, runs_dir)

    settings = Settings(
        api_key=api_key,
        model=effective.get("model", "default"),
        base_url=base_url,
        temperature=effective.get("temperature", 0.7),
    )
    llm_client = LLMClient(settings)

    logger.info(
        f"Starting GEPA: {opt_config.num_generations} generations, "
        f"population={opt_config.population_size}, model={settings.model}"
    )

    optimizer = GEPAOptimizer(llm_client=llm_client, config=opt_config)

    resume_from = effective.get("resume")

    try:
        result = optimizer.optimize(baseline_prompt, resume_from=resume_from)
        _save_results(result, Path(runs_dir))
        _print_summary(result)
    except KeyboardInterrupt:
        logger.warning("Optimization interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        raise


_YAML_TO_CANONICAL = {
    "generations": "num_generations",
    "temperature": "mutation_temperature",
    "criterion": "criterion_name",
}


def _build_optimization_config(
    effective: Dict[str, Any],
    dataset_path: Path,
    runs_dir: str,
) -> "OptimizationConfig":
    """Build OptimizationConfig from effective config dict."""
    from .models import OptimizationConfig

    overrides: Dict[str, Any] = {}
    config_fields = OptimizationConfig.model_fields
    for key, value in effective.items():
        canonical = _YAML_TO_CANONICAL.get(key, key)
        if canonical in config_fields and value is not None:
            overrides[canonical] = value

    overrides.pop("dataset_path", None)
    overrides["runs_dir"] = runs_dir

    profile = effective.get("profile", DEFAULT_PROFILE)
    return OptimizationConfig.from_profile(profile, str(dataset_path), **overrides)


def _load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load YAML config file if it exists."""
    path = Path(config_path)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _merge_config(yaml_data: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Merge config using 3 layers: required + profile + overrides."""
    profile_from_env = os.environ.get("GEPA_PROFILE")
    result = dict(yaml_data)
    cli_overrides = {
        "prompt": args.prompt,
        "dataset": args.dataset,
        "profile": args.profile,
        "generations": args.generations,
        "population_size": args.population_size,
        "model": args.model,
        "base_url": args.base_url,
        "api_key": args.api_key,
        "criterion": args.criterion,
        "resume": args.resume,
    }
    for key, value in cli_overrides.items():
        if value is not None:
            result[key] = value

    profile = (result.get("profile") or profile_from_env or DEFAULT_PROFILE).strip().lower()
    if profile not in SUPPORTED_PROFILES:
        raise ValueError(
            f"Unsupported profile '{profile}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_PROFILES))}"
        )

    # Layer 1: profile defaults
    effective = dict(PROFILE_PRESETS[profile]) if profile != "advanced" else {}
    # Layer 2/3: user overrides from YAML + CLI
    effective.update(result)
    effective["profile"] = profile
    return effective


def _validate_required_config(config: Dict[str, Any]) -> None:
    """Validate required user-facing config fields."""
    missing = [field for field in REQUIRED_CONFIG_FIELDS if not config.get(field)]
    if missing:
        raise ValueError(
            f"Missing required config fields: {', '.join(missing)}. "
            f"Set them in {GEPA_YAML} or pass via CLI."
        )


def _save_results(result: Any, runs_dir: Path) -> None:
    """Save optimization results to disk."""
    run_dir = runs_dir / result.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics_file = run_dir / "metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_id": result.run_id,
                "criterion": result.criterion_name,
                "duration_seconds": result.duration_seconds,
                "converged": result.converged,
                "baseline_metrics": result.baseline_metrics.model_dump(),
                "recommended_metrics": result.recommended_prompt.metrics.model_dump(),
                "pareto_frontier_size": len(result.pareto_frontier),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    prompt_file = run_dir / "recommended_prompt.txt"
    prompt_file.write_text(result.recommended_prompt.text, encoding="utf-8")

    pareto_file = run_dir / "pareto_frontier.yaml"
    with open(pareto_file, "w", encoding="utf-8") as f:
        pareto_data = {
            "frontier": [
                {
                    "id": c.id,
                    "generation": c.generation,
                    "metrics": c.metrics.model_dump(),
                    "prompt": c.text,
                }
                for c in result.pareto_frontier
            ]
        }
        yaml.dump(pareto_data, f, allow_unicode=True, default_flow_style=False)

    if result.error_analysis:
        error_file = run_dir / "error_analysis.json"
        with open(error_file, "w", encoding="utf-8") as f:
            json.dump(result.error_analysis, f, indent=2, ensure_ascii=False)

    logger.success(f"Results saved to: {run_dir}")


def _print_summary(result: Any) -> None:
    """Print optimization summary."""
    baseline_acc = result.baseline_metrics.accuracy
    best_acc = result.recommended_prompt.metrics.accuracy
    improvement = (best_acc - baseline_acc) * 100

    print(f"\n{'='*60}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"Baseline accuracy:  {baseline_acc:.1%}")
    print(f"Best accuracy:      {best_acc:.1%}")
    print(f"Improvement:        {improvement:+.1f}%")
    print(f"FN rate:            {result.recommended_prompt.metrics.false_negative_rate:.1%}")
    print(f"FP rate:            {result.recommended_prompt.metrics.false_positive_rate:.1%}")
    print(f"Duration:           {result.duration_seconds:.0f}s")
    print(f"{'='*60}")

    if result.error_analysis:
        patterns = result.error_analysis.get("patterns", "")
        if patterns:
            print(f"\nError patterns:\n{patterns}")
        recommendations = result.error_analysis.get("recommendations", "")
        if recommendations:
            print(f"\nRecommendations:\n{recommendations}")


def run_optimization() -> None:
    """Legacy entry point — redirects to main."""
    main()
