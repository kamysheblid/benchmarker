# benchmarker

Find the best sampling parameters (temperature, top_p, top_k, …) for a model
served by an OpenAI-compatible API. It runs a suite of test prompts, measures
speed (tokens/s, time-to-first-token), and helps you rank configs by a mix of
speed and quality — using a **manual judge LLM** for quality scoring.

## Install

```bash
pip install -e .
```

## Quick Start

```bash
# 1. Initialise benchmarks/ with self-contained YAML files
benchmarker init

# 2. Run a benchmark
benchmarker run --model my-model --url http://localhost:8080/v1/chat/completions

# 3. Copy judge_prompt.md → external LLM → save reply as reply.txt

# 4. Parse the judge's reply
benchmarker parse reply.txt
```

## Key Flags

| Flag | Purpose |
|---|---|
| `--model NAME` | Model name sent to the API |
| `--url URL` | OpenAI-compatible endpoint (default `http://localhost:8080/v1/chat/completions`) |
| `--benchmarks PATH` | Benchmark YAML file or `benchmarks/` directory |
| `--resume` | Resume from last checkpoint |
| `--force` | Discard checkpoint and start fresh |
| `--verbose` | Show retry/backoff details in console |

## Optimizers

- **`grid`** — tries every combination. Exhaustive; only practical for small search spaces.
- **`random`** — samples random configs until `budget` is exhausted. Simple and fast.
- **`bayesian`** — learns from each config and focuses sampling on promising regions. Works well for coarse exploration over wide ranges. `budget` is the max number of configs it will test; start with `12–20`.
- **`baseline_sweep`** — varies one parameter at a time while holding others fixed. Requires a `baseline` config.

For example, with `budget: 16`, `bayesian` might try 16 different temperature/top_p combos. After each one, it updates its model of which regions are fastest and concentrates later samples there — instead of blindly guessing.

## What It Does

- Varies parameters (temperature, top_p, top_k, repetition_penalty, …)
- Measures TTFT, tokens/s, total time on every request
- Saves raw metrics and a judge prompt for external LLM-as-a-judge scoring
- Import judge scores and get a combined speed/quality ranking

For detailed configuration, test formats, logging, and advanced usage, see [DOCS.md](DOCS.md).
