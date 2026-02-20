# Simple Sentiment Classification

Minimal GEPA example with **zero custom evaluation code**.

GEPA substitutes `{text}` from `input` into the prompt, sends to LLM, and compares the response with `expected` via case-insensitive exact match.

## Files

| File | Description |
|---|---|
| `prompt.txt` | Baseline prompt with `{text}` placeholder |
| `dataset.jsonl` | 15 labeled examples (positive / negative / neutral) |
| `run.py` | ~20 lines: load config, create optimizer, run |

## Run

```bash
# Set your API key
export OPENAI_API_KEY=sk-...

# Run optimization
python run.py
```

For local models, edit `run.py` and set `base_url`:

```python
settings = Settings(base_url="http://localhost:8000/v1", model="default")
```

## What happens

1. GEPA evaluates the baseline prompt on all 15 examples
2. Mutates the prompt based on failures (adds clarity, examples, structure)
3. Evaluates each mutant, selects Pareto-optimal candidates
4. Repeats for 4 generations
5. Saves the best prompt to `runs/`
