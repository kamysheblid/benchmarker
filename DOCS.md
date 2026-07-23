# DOCS.md — Detailed Documentation

This file contains detailed configuration, workflow internals, and reference material for the `benchmarker` project.

For a quick start, see `README.md`.

## Table of Contents

- [Workflow](#workflow)
  - [1. Initialise (once)](#1-initialise-once)
  - [2. Run a benchmark](#2-run-a-benchmark)
  - [3. Judge the results](#3-judge-the-results)
  - [4. Parse the judge's reply](#4-parse-the-judges-reply)
  - [Full cycle example](#full-cycle-example)
- [CLI commands](#cli-commands)
  - [`benchmarker run` options](#benchmarker-run-options)
- [Configuration file formats](#configuration-file-formats)
  - [`benchmarks/*.yaml` — self-contained benchmark files](#benchmarksyaml--self-contained-benchmark-files)
    - [File format](#file-format)
    - [Block scalar prompts](#block-scalar-prompts)
    - [Auto-generated IDs](#auto-generated-ids)
    - [Backward compatibility](#backward-compatibility)
    - [Migration from old format](#migration-from-old-format)
  - [`params.yaml` — search space (legacy)](#paramsyaml--search-space-legacy)
    - [Optimizer types](#optimizer-types)
    - [Budget](#budget)
    - [Parameter types](#parameter-types)
- [Agent-Specific Benchmarking](#agent-specific-benchmarking)
  - [Agent Benchmarks Structure](#agent-benchmarks-structure)
  - [System Prompts](#system-prompts)
  - [Reasoning Flag by Agent Type](#reasoning-flag-by-agent-type)
  - [Repeat Counts by Agent](#repeat-counts-by-agent)
  - [Per-Agent Judge Criteria](#per-agent-judge-criteria)
  - [Running Agent Benchmarks](#running-agent-benchmarks)
- [Logging & Resilience](#logging--resilience)
  - [Structured logs](#structured-logs)
  - [Verbose console output](#verbose-console-output)
  - [Resume and force](#resume-and-force)
  - [Error reports](#error-reports)
  - [Circuit breaker](#circuit-breaker)
- [Tests](#tests)

## Workflow

The tool follows a two-step loop:

### 1. Initialise (once)

```bash
benchmarker init
```

Creates a `benchmarks/` directory populated with starter self-contained `.yaml` files
(one per category). Each file contains its own `optimizer`, `parameters`, `static_params`,
and `tests` list, so you can customise prompts and search space without a separate
`params.yaml`.

### 2. Run a benchmark

```bash
benchmarker run --model my-model --url http://localhost:8080/v1/chat/completions
```

If no `benchmarks/` directory is found in the current working directory, the tool falls
back to the bundled default test suite (self-contained YAML files packaged with
`benchmarker`).

This writes:
- `benchmark_runs/latest/raw_data.json` — raw metrics
- `benchmark_runs/latest/judge_prompt.md` — a self-contained file for the judge
- `benchmark_runs/latest/run_meta.json` — benchmark source metadata for `parse`

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
| **refine** | Narrows the parameter ranges in the source benchmark YAML and tells you to re-run. |
| **expand** | Suggests broadening the search or adding more tests. |

If `refine`, just run `benchmarker run` again with no changes — the benchmark YAML
is already updated with tighter ranges.

### Full cycle example

```bash
# 1. Initialise
benchmarker init

# 2. Run (all categories)
benchmarker run --model Qwen3.5-4B-Q4_K_M --url http://127.0.0.1:8080/v1/chat/completions

# 3. Copy judge_prompt.md → ChatGPT → save reply as judge_reply.txt

# 4. Parse
benchmarker parse judge_reply.txt
# → If conclude: done
# → If refine: benchmark YAML is updated, run again
```

## CLI commands

```
benchmarker init     Create benchmarks/ directory with self-contained YAML files
benchmarker run      Run a benchmark for the given model
benchmarker parse    Parse the judge's reply and take action
```

### `benchmarker run` options

```
--tests PATH        Path to test suite JSON file or benchmarks/ directory
--categories TEXT   Comma-separated category slugs to load (directory mode only)
--params PATH       Path to parameter search-space YAML
```

## Configuration file formats

### `benchmarks/*.yaml` — self-contained benchmark files

`benchmarker init` creates a `benchmarks/` directory containing self-contained YAML
files, one per starter category.

```
benchmarks/
├── api-integration.yaml
├── bug-fixing.yaml
├── code-generation.yaml
├── comment-generation.yaml
├── general.yaml
├── refactoring.yaml
├── security-vulnerability.yaml
└── test-generation.yaml
```

`init` creates the 8 starter categories shown above. You can add more categories
by creating additional `.yaml` files under `benchmarks/`.

#### File format

Each `.yaml` file is fully self-contained:

```yaml
optimizer:
  type: bayesian
  budget: 40
parameters:
  - name: temperature
    type: float
    low: 0.0
    high: 1.5
    step: 0.2
  - name: top_k
    type: int
    low: 10
    high: 100
    step: 10
static_params:
  chat_template_kwargs:
    enable_thinking: false
tests:
  - id: bugfixing
    prompt: |
      You are a debugging assistant. Explain the bug in the following code...
    max_tokens: 2048
    repeat: 5
    reasoning: true
```

Supported fields:

- `optimizer.type` — `bayesian`, `grid`, `random`, or `baseline_sweep`
- `optimizer.budget` — number of configs to evaluate
- `parameters[]` — search-space parameter specs
- `static_params` — fixed key/value pairs sent with every request
- `tests[]` — list of benchmark prompts
- `tests[].id` (str, optional) — unique identifier. When omitted, the loader
  auto-generates an id using the pattern `<file-stem>-test-<N>`.
- `tests[].prompt` (str, required) — the benchmark prompt. Must not be empty.
- `tests[].system` (str, optional) — system message override.
- `tests[].max_tokens` (positive int, optional) — maximum tokens to generate.
- `tests[].repeat` (int, default 1) — number of times to repeat this test.
- `tests[].reasoning` (bool, optional) — True = encourage chain-of-thought,
  False = discourage, None = default.

#### Block scalar prompts

Use YAML block scalars (`|`) for multi-line prompts so indentation and newlines
are preserved exactly:

```yaml
tests:
  - id: test_generation
    prompt: |
      You are a testing assistant. Generate comprehensive unit tests...

      Function to test:
      def divide_list(items, divisor):
          ...
```

#### Auto-generated IDs

If a test case does not specify an `id`, the loader assigns one automatically:

```yaml
tests:
  - prompt: "A"        # becomes `autoid-test-1`
  - prompt: "B"        # becomes `autoid-test-2`
```

The auto-id pattern is `<file-stem>-test-<N>`, where `<file-stem>` is the YAML
filename without the `.yaml` extension.

#### Backward compatibility

`tests.json` is no longer created by `init`, but the legacy flat-file format is
still supported. You can load an existing `tests.json` with:

```bash
benchmarker run --tests tests.json
```

When using a flat file, omit `--categories` — category filtering is only available
in directory mode.

#### Migration from old format

To migrate from the old per-directory JSON format, combine each category's JSON
files into a single `category.yaml` under `benchmarks/` with the structure shown
above. The `id`, `prompt`, `max_tokens`, `repeat`, and `reasoning` fields map
directly to entries in the `tests` list.

### `params.yaml` — search space (legacy)

```yaml
optimizer:
  type: grid          # grid | random | bayesian | baseline_sweep
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

#### Optimizer types

| Type | How it works | When to use |
|---|---|---|
| **`grid`** | Enumerates every combination of parameter values. Exhaustive. | Small categorical grids where you want to test every point. |
| **`random`** | Samples random combinations until `budget` is exhausted. | Large numeric ranges where grid search would be too expensive. |
| **`bayesian`** | Builds a probabilistic model of the parameter space and focuses sampling on promising regions. Stops after `budget` trials. | Coarse exploration over wide ranges; typically finds the best region with fewer samples than grid. |
| **`baseline_sweep`** | Ablation study: varies one parameter at a time while holding others at a baseline. | Isolating the effect of a single parameter. Requires a `baseline` config. |

#### Budget

`budget` controls how many **parameter configurations** the optimizer will evaluate. It is **not** the number of API requests. Each config is run against every test × repeat count, so the total number of LLM calls is:

```
total_requests = budget × Σ(test.repeat for each test)
```

##### Per-optimizer behavior

| Optimizer | How `budget` is used | What happens when exhausted | Code path |
|---|---|---|---|
| **`grid`** | Ignored completely | `StopIteration` after all combinations are yielded | `GridOptimizer._build_combinations()` uses `itertools.product`; `estimated_steps()` returns `len(self._combos)` |
| **`random`** | Max number of random samples | `StopIteration` when `_count >= budget` | `RandomOptimizer.suggest()` increments `_count` |
| **`bayesian`** | Max number of Optuna trials | `StopIteration` when `_count >= budget` | `BayesianOptimizer.suggest()` calls `study.ask()` then increments `_count` |
| **`baseline_sweep`** | Ignored; uses a `baseline` config instead | `StopIteration` after all ablation configs are yielded | `ControlledOptimizer._build_ablations()` builds baseline + one-parameter-at-a-time combos |
| **`adaptive`** | No budget parameter; exhaustive refined grid | `StopIteration` after all hinted-range combos are yielded | `AdaptiveOptimizer._build_refined()` |

##### Feedback loop: `tell()`

After each config finishes, the runner reports metrics back to the optimizer:

```python
self.optimizer.tell({
    "tokens_per_sec": penalized_speed,  # avg speed × success_rate
    "success_rate": success_rate,
    "coverage": coverage,
})
```

The Bayesian optimizer uses `tokens_per_sec` or `quality` as the Optuna trial value. If both are missing, the trial is marked `FAIL`.

##### Grid size with `step`

When a parameter has a `step`, the number of values is computed as:

```python
n = int(round((high - low) / step)) + 1
```

So `low: 0.0, high: 1.0, step: 0.1` → 11 values → 11 combos for that parameter.

##### Two-phase mode

`TwoPhaseOptimizer` has a separate `phase1_budget`. The runner switches to phase2 when `trial_index >= phase1_budget`. Phase2 uses `AdaptiveOptimizer` with a refinement hint (either from the best coarse config or from auto-eval passing configs). Phase2 is exhaustive over the narrowed range, so it has no budget cap of its own.

##### Practical implications

- If you set `budget: 8` with `grid` and 3 parameters × 3 values each, you still get **27 configs** (budget ignored).
- If you set `budget: 8` with `random` or `bayesian`, you get exactly **8 configs**.
- A single config with 10 tests at `repeat: 5` = 50 API calls per config. With `budget: 8`, that's 400 total calls.
- `estimated_steps()` returns `budget` for random/bayesian, or `len(combos)` for grid/baseline_sweep/adaptive — the UI progress bar uses this.

**Rule of thumb:**
- Use `grid` for 4 or fewer total combinations.
- Use `bayesian` with `budget: 12–20` for coarse exploration over wide ranges.
- Use `random` as a cheap fallback when Optuna is unavailable.

#### Parameter types

Parameters can be defined as:

- **`categorical`** with explicit `choices` — e.g. `choices: [0, 1.0, 2.0]`
- **`float`** with `low`, `high`, and optional `step` — e.g. `low: 0.0, high: 1.0, step: 0.1`
- **`int`** with `low`, `high`, and optional `step` — e.g. `low: 10, high: 100, step: 10`

For categorical parameters, the grid size is the product of all `choices` lengths. For numeric parameters with `step`, the grid size is the product of the number of steps in each range.

## Agent-Specific Benchmarking

The `benchmarker` supports agent-specific benchmarking via the `system` field in test YAML files and per-agent judge criteria.

### Agent Benchmarks Structure

For multi-agent systems like [agent-hive](https://github.com/hung319/agent-hive), create agent-specific benchmark YAML files:

```
benchmarks/
├── hive/                          # Chief Planner & Orchestrator
│   ├── planning.yaml
│   ├── orchestration.yaml
│   └── approval.yaml
├── architect/                     # Feature Architect (planner only)
│   ├── design.yaml
│   ├── interviewing.yaml
│   └── spec-writing.yaml
├── scout/                         # Codebase & External Researcher
│   ├── codebase-exploration.yaml
│   ├── external-research.yaml
│   └── dependency-analysis.yaml
...
```

### System Prompts

Each test YAML file should include a `system` field that mirrors the agent's actual system prompt from source. This ensures the benchmarked model adopts the correct persona, constraints, and output format.

Example:

```yaml
tests:
  - id: forager-impl-001
    system: "You are Forager, an autonomous senior engineer and task execution agent..."
    prompt: "Implement a function in Python..."
    max_tokens: 2048
    repeat: 7
    reasoning: false
```

See `SYSTEM_PROMPTS.md` for verified system prompts from the agent-hive source.

### Reasoning Flag by Agent Type

| Agent Type | Reasoning | Rationale |
|---|---|---|
| **Planning/Design** (Hive, Architect, Scout) | `true` | Needs to show thought process for plans and research |
| **Execution** (Forager, Project-Initializer) | `false` | Should output code directly, no reasoning needed |
| **Review** (Hygienic, Code-Reviewer) | `true` | Needs to explain reasoning for reviews |
| **Location/Analysis** (Codebase-Locator, Codebase-Analyzer, Pattern-Finder) | `false` | Should output structured data directly |
| **Simplification** (Code-Simplifier) | `true` | Should explain simplifications |
| **Orchestration** (Swarm) | `false` | Should output structured delegation plans |

### Repeat Counts by Agent

| Agent | Recommended `repeat` | Rationale |
|---|---|---|
| **hive** | 5–7 | Planning outputs vary; need sufficient samples |
| **architect** | 5 | Design outputs can vary; moderate repeat |
| **swarm** | 5 | Orchestration logic should be consistent |
| **scout** | 5–7 | Research outputs vary based on information retrieval |
| **forager** | 7–10 | Code generation is highly stochastic |
| **hygienic** | 5 | Review outputs should be consistent |
| **code-reviewer** | 5 | Similar to hygienic |
| **code-simplifier** | 5–7 | Refactoring can vary |
| **codebase-analyzer** | 3–5 | Analysis outputs should be consistent |
| **codebase-locator** | 3–5 | Location tasks are deterministic |
| **pattern-finder** | 5 | Pattern identification can vary |
| **project-initializer** | 3 | Scaffolding is deterministic |

### Per-Agent Judge Criteria

Different agents need different evaluation criteria. See `JUDGE_TEMPLATE.md` for the full template with agent-specific scoring rubrics.

Key criteria by agent:

- **Forager**: Code correctness, convention following, verification, minimal changes
- **Hive**: Plan structure, dependency correctness, phase awareness, actionability
- **Scout**: Evidence-based claims, search strategy, no speculation, parallel execution
- **Hygienic**: Documentation vs design focus, four criteria (clarity/verifiability/completeness/big picture), specificity
- **Architect**: Intent classification, self-clearance, AI-slop detection, test strategy
- **Swarm**: Delegation logic, parallelization, verification plan, blocker handling

### Running Agent Benchmarks

```bash
# Run all categories for a specific agent
benchmarker run --model my-model --categories hive/planning

# Run all categories for all agents
benchmarker run --model my-model

# Generate per-agent judge prompts
for agent in hive architect swarm scout forager hygienic; do
  cat JUDGE_TEMPLATE.md | sed "s/{AGENT_NAME}/$agent/g" > runs/$agent/judge_prompt.md
done
```

## Logging & Resilience

### Structured logs

Every benchmark run writes a `benchmarker.log` file inside the run directory:

```text
benchmark_runs/
└── latest/
    ├── benchmarker.log      # detailed logs (DEBUG+)
    ├── checkpoint.json      # resume checkpoint
    ├── error_report.json    # generated on failure (if any)
    ├── raw_data.json
    ├── judge_prompt.md
    └── ...
```

Use `benchmarker.log` for debugging failed runs. Log levels:
- `INFO` — normal progress and state transitions.
- `DEBUG` — detailed retry, backoff, and streaming metrics (written to file always; console only with `--verbose`).
- `ERROR` / `CRITICAL` — endpoint failures, circuit-breaker trips, I/O errors.

### Verbose console output

```bash
benchmarker run --verbose --model my-model --url http://localhost:8080/v1/chat/completions
```

`--verbose` sets the console handler to `DEBUG` so you can see retry loops and backoff delays in real time.

### Resume and force

If a run is interrupted ( Ctrl+C, crash, power loss ), a `checkpoint.json` is left behind.

```bash
# Resume from the last checkpoint (skips already-completed configs)
benchmarker run --resume --model my-model --url http://localhost:8080/v1/chat/completions

# Discard the checkpoint and start fresh
benchmarker run --force --model my-model --url http://localhost:8080/v1/chat/completions
```

If you run `benchmarker run` without either flag and a checkpoint exists:
- In an interactive terminal, you are prompted to resume or start fresh.
- In a non-interactive environment (CI, SSH without TTY), the command aborts with a message instructing you to use `--resume` or `--force`.

### Error reports

When a run ends with errors, an `error_report.json` is saved to the run directory. The report contains:

- Total error count and breakdown by exception type.
- First and last error messages.
- Request context (URL, model, timeout, HTTP status) when available.
- Heuristic recommendations (e.g., "High server error rate—check server health", "Circuit breaker tripped—endpoint may be down").

### Circuit breaker

The runner wraps LLM calls in a circuit breaker. After 5 consecutive `TransientError` / `ServerError` failures, the breaker opens and stops new requests. When the endpoint is healthy again, the breaker automatically transitions to `half_open` after 60 seconds, then back to `closed` on the next successful call.

## Tests

```bash
pytest
```
