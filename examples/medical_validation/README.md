# Medical Protocol Validation (Complex Example)

Real-world GEPA usage: optimizing prompts for medical protocol validation via an external API.

This example is **~500 lines** of Python. Here's why — and why **none of that complexity comes from GEPA**.

## Why is this example complex?

The evaluation pipeline has many domain-specific steps that have nothing to do with prompt optimization:

| Step | What it does | Lines |
|---|---|---|
| Protocol building | Convert flat `input_data` dict into a nested `Protocol` structure with UUIDs | ~20 |
| API call | POST to `/validate` endpoint with timeout, semaphore concurrency | ~15 |
| Response parsing | Extract criterion result from nested JSON, map int values to strings | ~15 |
| Progress tracking | Thread-safe tqdm wrapper for tracking eval requests | ~30 |
| Dataset conversion | Parse raw medical records, balance labels, convert to GEPA format | ~125 |
| Multi-criterion loop | Run optimization for each of 5 criteria sequentially | ~40 |
| Resume support | Find latest incomplete run and resume from checkpoint | ~15 |
| Results visualization | Plot baseline vs optimized accuracy per criterion | ~40 |

**Total domain-specific code: ~300 lines.** The GEPA integration itself is ~20 lines (same as the simple example).

## The key difference from simple_sentiment

In the simple example, evaluation is: send prompt to LLM, compare answer with expected.

Here, evaluation is: inject the prompt as a criterion description into a system prompt, POST the full protocol to an external `/validate` API, parse the JSON response, extract a specific criterion's value, and map it to `True`/`False`/`NotEnough`. All of this is described in `eval_fn` — GEPA doesn't know or care about these details.

## Files

| File | Lines | Description |
|---|---|---|
| `optimize.py` | ~370 | Main optimization script with custom `eval_fn` |
| `convert_data.py` | ~125 | Convert raw medical records to GEPA dataset format |

## Prerequisites

- A running medical validation API at `http://localhost:8000/validate`
- Source dataset: `evaluation/data/golden_consultations.jsonl`
- A local LLM endpoint (SGLang/vLLM) for GEPA mutations

## Run

```bash
# Single criterion
CRITERION=icd10_matches_diagnosis python optimize.py

# All 5 criteria
python optimize.py
```

## Architecture

```
optimize.py
    |
    |-- make_eval_fn(criterion)     # returns async eval_fn
    |       |
    |       |-- _build_protocol()   # input_data -> Protocol dict
    |       |-- httpx.post()        # call /validate API
    |       |-- _extract_result()   # JSON -> "True"/"False"/"NotEnough"
    |
    |-- GEPAOptimizer(              # GEPA handles the rest
    |       llm_client=...,
    |       config=...,
    |       eval_fn=eval_fn,
    |   )
    |
    |-- optimizer.optimize(baseline_prompt)
```
