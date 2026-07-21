# benchmarker

Find the best sampling parameters (temperature, top_p, top_k, …) for a model
served by `llama-server` (OpenAI-compatible API). It runs a suite of test
prompts, measures speed (tokens/s, time-to-first-token), and helps you rank
configs by a mix of speed and quality — using a **manual judge LLM** for quality
scoring.

## Install

```bash
pip install -e .
```

## Workflow (Simplified)

The tool follows a two-step loop:

### 1. Initialise (once)

```bash
benchmarker init
```

Creates `tests.json` and `params.yaml` in the current directory with sensible
defaults. Edit them to customise your prompts and parameter search space.

### 2. Run a benchmark

```bash
benchmarker run --model my-model --url http://localhost:8080/v1/chat/completions
```

This writes:
- `benchmark_runs/latest/raw_data.json` — raw metrics
- `benchmark_runs/latest/judge_prompt.md` — a self-contained file for the judge

### 3. Judge the results

Copy the **entire** `judge_prompt.md` file into a web LLM (ChatGPT, Claude, etc.).
The prompt includes strict instructions to output a JSON object with:

- `scores` — quality rating per config
- `recommendation` — `"conclude"`, `"refine"`, or `"expand"`
- `confidence` — `"high"`, `"medium"`, or `"low"`
- `reasoning` — justification
- `refinement_hint` — narrower parameter ranges (when recommending "refine")

### 4. Parse the judge's reply

Save the judge's reply to a file (e.g. `reply.txt`) and run:

```bash
benchmarker parse reply.txt
```

The tool then automatically:

| Recommendation | Action |
|---|---|
| **conclude** | Prints the best configuration, judge's reasoning, and scores. |
| **refine** | Narrows the parameter ranges in `params.yaml` and tells you to re-run. |
| **expand** | Suggests broadening the search or adding more tests. |

If `refine`, just run `benchmarker run` again with no changes — `params.yaml` is
already updated with tighter ranges.

### Full cycle example

```bash
# 1. Initialise
benchmarker init

# 2. Run
benchmarker run --model Qwen3.5-4B-Q4_K_M --url http://127.0.0.1:8080/v1/chat/completions

# 3. Copy judge_prompt.md → ChatGPT → save reply as judge_reply.txt

# 4. Parse
benchmarker parse judge_reply.txt
# → If conclude: done
# → If refine: params.yaml is updated, run again
```

## CLI commands

```
benchmarker init     Create default config files (tests.json, params.yaml)
benchmarker run      Run a benchmark for the given model
benchmarker parse    Parse the judge's reply and take action
```

## Configuration file formats

### `tests.json` — test prompts

```json
[
  {
    "id": "coding_chunk",
    "prompt": "Write a function def chunk(seq, n) ...",
    "max_tokens": 2048,
    "repeat": 5,
    "reasoning": false,
    "stop": ["\ndef ", "\nclass "]
  }
]
```

### `params.yaml` — search space

```yaml
optimizer:
  type: grid          # grid | random | bayesian
  budget: 8
parameters:
  - name: temperature
    type: float
    low: 0.0
    high: 1.0
  - name: top_p
    type: float
    low: 0.5
    high: 1.0
static_params:
  chat_template_kwargs:
    enable_thinking: false
```

## Tests

```bash
pytest
```
