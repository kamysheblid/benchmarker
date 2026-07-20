# benchmarker

Find the best sampling parameters (temperature, top_p, top_k, …) for a model
served by `llama-server` (OpenAI-compatible API). It runs a suite of test
prompts, measures speed (tokens/s, time-to-first-token), and helps you rank
configs by a mix of speed and quality.

## Install

```bash
pip install -e .
```

## How it works

1. **Run** a benchmark → saves responses to `benchmark_runs/latest/`.
2. **Judge** the responses with an external LLM (copy `eval_output.md`).
3. **Import** the scores → get a combined speed + quality ranking.

## Usage

Run with defaults (uses the bundled test suite and parameter ranges):

```bash
benchmarker run --model my-model --url http://localhost:8080/v1/chat/completions
```

Point at your own config files:

```bash
benchmarker run --model my-model --tests my_tests.json --params my_params.yaml
```

This writes:
- `benchmark_runs/latest/raw_data.json` — raw metrics
- `benchmark_runs/latest/eval_output.md` — prompts + responses + a rating template

Copy `eval_output.md` into a judge LLM, then paste its JSON scores back and run:

```bash
benchmarker import-scores scores.json --run-dir benchmark_runs/latest
```

Tune the balance between quality and speed (default `0.5`):

```bash
benchmarker import-scores scores.json --weight-quality 0.7
```

## Config files

`tests.json` — a list of prompts:

```json
[
  { "id": "q1", "prompt": "Explain recursion simply.", "max_tokens": 128 },
  { "id": "q2", "prompt": "Write a haiku about the sea." }
]
```

`params.yaml` — the search space and optimizer:

```yaml
optimizer:
  type: grid        # grid | random | bayesian
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
```

## Tests

```bash
pytest
```
