"""Minimal GEPA example: sentiment classification with zero custom eval code."""

from pathlib import Path

from gepa import GEPAOptimizer, LLMClient, OptimizationConfig
from gepa.config import Settings

PROMPT_FILE = Path(__file__).parent / "prompt.txt"
DATASET_FILE = Path(__file__).parent / "dataset.jsonl"

settings = Settings(
    model="gpt-4o-mini",
)

config = OptimizationConfig.from_profile(
    "fast",
    dataset_path=str(DATASET_FILE),
    runs_dir=str(Path(__file__).parent / "runs"),
)

llm_client = LLMClient(settings)
optimizer = GEPAOptimizer(llm_client=llm_client, config=config)

baseline_prompt = PROMPT_FILE.read_text(encoding="utf-8").strip()
result = optimizer.optimize(baseline_prompt)

print(f"\nBaseline accuracy: {result.baseline_metrics.accuracy:.1%}")
print(f"Best accuracy:     {result.recommended_prompt.metrics.accuracy:.1%}")
print(f"\nOptimized prompt:\n{result.recommended_prompt.text}")
