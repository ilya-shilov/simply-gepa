# GEPA - Genetic-Pareto Prompt Optimizer

Automatically optimize LLM prompts using a genetic algorithm with multi-objective Pareto selection.

GEPA treats prompts as a population: each generation, it analyzes where the current prompt fails, mutates it into improved candidates, and keeps only the Pareto-optimal ones — balancing accuracy, false negative rate, false positive rate, and token cost simultaneously. You bring a baseline prompt and a labeled dataset; GEPA handles the rest.

Works with any OpenAI-compatible API — cloud models (GPT-4o, Claude) or local inference (SGLang, vLLM, Ollama).

## Installation

```bash
pip install gepa            # core (no visualization)
pip install gepa[viz]       # + matplotlib evolution plots
pip install gepa[all]       # + visualization + rich UI
```

## Quick Start (CLI)

Three commands to your first optimized prompt:

```bash
gepa init                   # creates gepa.yaml, prompt.txt, dataset.jsonl
# edit gepa.yaml — set model / base_url / API key
gepa run                    # runs optimization, saves results to runs/
```

`gepa init` generates a working sentiment-classification example. Edit `prompt.txt` and `dataset.jsonl` with your own data, then `gepa run`.

## Quick Start (Python API)

```python
from gepa import GEPAOptimizer, LLMClient, OptimizationConfig
from gepa.config import Settings

settings = Settings(
    api_key="sk-...",           # or set GEPA_API_KEY / OPENAI_API_KEY env var
    model="gpt-4o-mini",
)
# For local models (SGLang, vLLM, Ollama):
# settings = Settings(base_url="http://localhost:8000/v1", model="default")

config = OptimizationConfig.from_profile(
    "balanced",
    dataset_path="dataset.jsonl",
    runs_dir="runs",
)

llm_client = LLMClient(settings)
optimizer = GEPAOptimizer(llm_client=llm_client, config=config)

result = optimizer.optimize("Classify the sentiment as positive, negative, or neutral.\n\nText: {text}\n\nSentiment:")

print(f"Baseline: {result.baseline_metrics.accuracy:.1%}")
print(f"Best:     {result.recommended_prompt.metrics.accuracy:.1%}")
print(result.recommended_prompt.text)
```

## How much code do I need to write?

| Your evaluation pipeline | Custom code you write | Example |
|---|---|---|
| `prompt + input -> LLM -> answer` (exact match) | **0 lines** | [simple_sentiment](examples/simple_sentiment/) |
| Custom comparison logic (fuzzy match, regex) | **~5 lines** (`compare_fn`) | — |
| Custom prompt formatting (multi-message, JSON schema) | **~10 lines** (`eval_fn`) | — |
| Full pipeline (external API, protocol building, response parsing) | **Your pipeline** | [medical_validation](examples/medical_validation/) |

The key insight: **your code complexity = your evaluation pipeline complexity, not GEPA's.**

- If evaluation is "send prompt to LLM, compare answer with expected" — you write nothing. GEPA substitutes `{placeholders}` from `input`, sends to LLM, compares with `expected` via exact match.
- If evaluation is a complex pipeline (external API calls, structured JSON output, multi-step processing) — you describe that pipeline in `eval_fn`. GEPA handles everything else: mutation, crossover, Pareto selection, early stopping, visualization.

### eval_fn signature

```python
async def eval_fn(prompt_text: str, input_data: dict) -> str:
    """
    Receives the current prompt candidate and one dataset entry's input dict.
    Must return a string prediction that will be compared with `expected`.
    """
```

### compare_fn signature

```python
def compare_fn(predicted: str, expected: str) -> bool:
    """Return True if predicted matches expected. Default: case-insensitive exact match."""
```

## Dataset Format

JSONL file, one entry per line. Each entry has `input` (dict with any keys) and `expected` (string):

```jsonl
{"input": {"text": "I love this product!"}, "expected": "positive"}
{"input": {"text": "Terrible experience."}, "expected": "negative"}
{"input": {"text": "It works fine."}, "expected": "neutral"}
```

The keys in `input` must match `{placeholders}` in your prompt (when using default eval). For example, if your prompt contains `{text}` and `{category}`, each input dict must have `"text"` and `"category"` keys.

## Profiles

| Profile | Generations | Population | Mutation temp | Crossover | Early stop | Best for |
|---|---|---|---|---|---|---|
| `fast` | 4 | 12 | 0.9 | On (0.4) | On | Quick experiments, prototyping |
| `balanced` | 8 | 10 | 0.7 | On (0.35) | On | General use (default) |
| `quality` | 12 | 8 | 0.5 | Off | On (strict) | Production prompts |
| `advanced` | — | — | — | — | — | Full manual control, no presets |

Use `advanced` when you want to set every parameter yourself.

## Configuration

### gepa.yaml (CLI)

```yaml
prompt: prompt.txt
dataset: dataset.jsonl
model: gpt-4o-mini
profile: balanced

# Local model endpoint
# base_url: http://localhost:8000/v1

# Override any profile parameter
# generations: 12
# population_size: 8
# temperature: 0.5
# criterion: my_criterion
# runs_dir: runs
```

### Environment variables

| Variable | Description |
|---|---|
| `GEPA_API_KEY` | API key (highest priority) |
| `OPENAI_API_KEY` | API key (fallback) |
| `GEPA_PROFILE` | Default profile name |

For local models (SGLang, vLLM, Ollama), set `base_url` in config — no API key needed.

### Python API

```python
# From profile with overrides
config = OptimizationConfig.from_profile(
    "quality",
    dataset_path="data.jsonl",
    num_generations=20,          # override profile default
    priority_fn_reduction=3.0,   # prioritize reducing false negatives
)

# Full manual control
config = OptimizationConfig(
    dataset_path="data.jsonl",
    num_generations=15,
    population_size=8,
    mutation_temperature=0.6,
    crossover_enabled=True,
    crossover_rate=0.3,
    early_stop_enabled=True,
    early_stop_sample_ratio=0.2,
    early_stop_accuracy_threshold=0.2,
    adaptive_mutation_enabled=True,
    adaptive_mutation_start_temperature=0.9,
    adaptive_mutation_end_temperature=0.3,
)
```

## Hyperparameters Reference

All parameters of `OptimizationConfig`:

### Evolution

| Parameter | Type | Default | Range | Description |
|---|---|---|---|---|
| `num_generations` | int | 10 | 1–50 | Number of generations (search depth) |
| `population_size` | int | 5 | 2–20 | Population size per generation (search width) |
| `mutation_temperature` | float | 0.7 | 0.0–1.0 | LLM creativity for mutations |

### Adaptive Mutation

| Parameter | Type | Default | Range | Description |
|---|---|---|---|---|
| `adaptive_mutation_enabled` | bool | False | — | Enable temperature decay over generations |
| `adaptive_mutation_start_temperature` | float | 0.9 | 0.0–1.0 | Temperature at generation 1 |
| `adaptive_mutation_end_temperature` | float | 0.3 | 0.0–1.0 | Temperature at last generation |

### Crossover

| Parameter | Type | Default | Range | Description |
|---|---|---|---|---|
| `crossover_enabled` | bool | False | — | Enable prompt hybrids |
| `crossover_rate` | float | 0.3 | 0.0–1.0 | Fraction of population created via crossover |
| `crossover_temperature` | float | 0.6 | 0.0–1.0 | LLM creativity when merging prompts |
| `crossover_parent_pool_size` | int | 3 | 2–20 | Top-N candidates to pick parents from |

### Early Stopping & Pareto Priorities

| Parameter | Type | Default | Range | Description |
|---|---|---|---|---|
| `early_stop_enabled` | bool | True | — | Pre-screen candidates on a subset |
| `early_stop_sample_ratio` | float | 0.2 | 0.05–0.9 | Fraction of dataset for quick check |
| `early_stop_accuracy_threshold` | float | 0.2 | 0.0–1.0 | Min accuracy to continue full eval |
| `priority_accuracy` | float | 2.0 | >=0 | Pareto weight for accuracy |
| `priority_fn_reduction` | float | 2.0 | >=0 | Pareto weight for reducing false negatives |
| `priority_fp_reduction` | float | 0.5 | >=0 | Pareto weight for reducing false positives |
| `priority_cost_reduction` | float | 0.001 | >=0 | Pareto weight for reducing prompt length |

### Other

| Parameter | Type | Default | Description |
|---|---|---|---|
| `llm_timeout` | int | 60 | LLM request timeout in seconds (10–300) |
| `mutation_system_prompt` | str | (built-in) | System prompt for mutation LLM |
| `reflection_template` | str | (built-in) | Template for error analysis step |
| `improvement_template` | str | (built-in) | Template for prompt improvement step |
| `crossover_template` | str | (built-in) | Template for crossover step |

## Separate LLM for Mutations

By default, GEPA uses the same LLM for both evaluation and prompt mutation/crossover. If your evaluation model is small or local, you can use a more capable model for mutations:

```python
eval_client = LLMClient(Settings(
    base_url="http://localhost:8000/v1",
    model="default",
))

meta_client = LLMClient(Settings(
    api_key="sk-...",
    model="gpt-4o",
))

optimizer = GEPAOptimizer(
    llm_client=eval_client,       # used for evaluation
    config=config,
    meta_llm_client=meta_client,  # used for mutation, crossover, error analysis
)
```

## How the Algorithm Works

GEPA runs a genetic algorithm where **prompts are the population** and **LLM is the mutation engine**:

```
Generation 0:  Evaluate baseline prompt → PromptMetrics (accuracy, FN, FP, cost)
                                        ↓
Generation 1:  Mutator analyzes failures → LLM generates improved prompts
               Crossover (optional)      → LLM merges best parts of two prompts
               Evaluate all candidates   → PromptMetrics for each
               Pareto selection          → keep non-dominated candidates
                                        ↓
Generation N:  ...repeat...
                                        ↓
Result:        Pareto frontier + recommended prompt + error analysis
```

**Mutation** is a two-step process:
1. **Reflection** — LLM sees the prompt + examples where it failed, analyzes error patterns
2. **Improvement** — LLM rewrites the prompt to fix identified problems

**Pareto selection** keeps candidates that are not dominated on any of 4 metrics. A candidate is dominated if another candidate is better or equal on all metrics and strictly better on at least one. The recommended prompt is chosen by priority: min FN rate → max accuracy → min cost.

## Output & Results

`optimizer.optimize()` returns an `OptimizationResult`:

```python
result.run_id                              # unique run identifier
result.criterion_name                      # criterion name
result.duration_seconds                    # total time
result.converged                           # True if accuracy plateaued

result.baseline_metrics                    # PromptMetrics of original prompt
result.baseline_metrics.accuracy           # 0.0–1.0
result.baseline_metrics.false_negative_rate
result.baseline_metrics.false_positive_rate

result.recommended_prompt                  # best PromptCandidate
result.recommended_prompt.text             # the actual prompt string
result.recommended_prompt.metrics          # PromptMetrics

result.pareto_frontier                     # List[PromptCandidate] — all Pareto-optimal
result.error_analysis                      # Dict with patterns & recommendations (or None)
```

### Files saved to `runs/`

| File | Content |
|---|---|
| `recommended_prompt.txt` | Best prompt — ready to use |
| `metrics.json` | run_id, accuracy, FN/FP rates, duration, converged |
| `pareto_frontier.yaml` | All Pareto-optimal candidates with metrics |
| `error_analysis.json` | Error patterns and recommendations |
| `state.json` | Checkpoint for resuming (auto-saved each generation) |
| `plots/` | PNG evolution graphs (if `gepa[viz]` installed) |

## Resume from Checkpoint

If optimization is interrupted (Ctrl+C, crash, timeout), resume from the last completed generation:

```python
# Python API
result = optimizer.optimize(baseline_prompt, resume_from="runs/gepa_run_20260208_143022")
```

```bash
# CLI
gepa run --resume runs/gepa_run_20260208_143022
```

GEPA auto-saves `state.json` after each generation, so you lose at most one generation of work.

## Visualization

GEPA creates PNG snapshots of the evolution graph after each generation (requires `gepa[viz]`):

```bash
pip install gepa[viz]   # matplotlib + networkx
```

`FileVisualizer` is used by default — no code needed. Plots are saved to `runs/<run_id>/plots/`.

The graph shows:
- **Nodes** = prompt candidates, colored by accuracy (green = best, red = worst)
- **Edges** = parent → child (mutation or crossover)
- **Stars** = Pareto frontier members
- **X-axis** = generation number

To use a custom visualizer, pass it to `GEPAOptimizer`:

```python
from gepa import FileVisualizer

visualizer = FileVisualizer(output_dir="my_plots/")
optimizer = GEPAOptimizer(llm_client=client, config=config, visualizer=visualizer)
```

## Custom LLM Client

GEPA works with any OpenAI-compatible API via the built-in `LLMClient`. For non-OpenAI providers, implement `BaseLLMClient`:

```python
from gepa import BaseLLMClient

class MyLLMClient(BaseLLMClient):
    async def achat_completion(self, messages, temperature=None, max_tokens=None, json_mode=False, **kwargs) -> str:
        # Your async implementation
        ...

    def chat_completion(self, messages, temperature=None, max_tokens=None, json_mode=False, **kwargs) -> str:
        # Your sync implementation
        ...

optimizer = GEPAOptimizer(llm_client=MyLLMClient(), config=config)
```

Both methods must accept `messages: List[Dict[str, str]]` and return a string response.

## Examples

- **[simple_sentiment](examples/simple_sentiment/)** — Zero custom code. Sentiment classification with default eval. ~20 lines of Python.
- **[medical_validation](examples/medical_validation/)** — Full pipeline. Custom eval_fn calling an external API, protocol building, response parsing. ~370 lines of Python — all of which describe the evaluation pipeline, not GEPA.


## License
<img width="131" height="180" alt="68747470733a2f2f692e706f7374696d672e63632f383750316b3139382f3638373437343730336132663266366637303635366537333666373537323633363532653666373236373266373437323631363436353664363137323662373332663666373036" src="https://github.com/user-attachments/assets/a7423392-df8e-4ced-ba90-61d4e40da0f5" />

The class is licensed under the MIT License

Copyright © 2026 Shilov Ilya.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
