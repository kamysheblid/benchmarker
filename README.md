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

Creates a `benchmarks/` directory and `params.yaml` in the current directory with
sensible defaults. The `benchmarks/` directory contains category subdirectories,
each holding one JSON file per prompt. Edit them to customise your prompts and
parameter search space.

### 2. Run a benchmark

```bash
benchmarker run --model my-model --url http://localhost:8080/v1/chat/completions
```

This writes:
- `benchmark_runs/latest/raw_data.json` — raw metrics
- `benchmark_runs/latest/judge_prompt.md` — a self-contained file for the judge

You can limit the run to specific categories with `--categories`:

```bash
benchmarker run --model my-model --categories bug-fixing,code-generation
```

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

# 2. Run (all categories)
benchmarker run --model Qwen3.5-4B-Q4_K_M --url http://127.0.0.1:8080/v1/chat/completions

# Or run only specific categories
benchmarker run --model Qwen3.5-4B-Q4_K_M --categories bug-fixing,refactoring

# 3. Copy judge_prompt.md → ChatGPT → save reply as judge_reply.txt

# 4. Parse
benchmarker parse judge_reply.txt
# → If conclude: done
# → If refine: params.yaml is updated, run again
```

## CLI commands

```
benchmarker init     Create benchmarks/ directory and params.yaml
benchmarker run      Run a benchmark for the given model
benchmarker parse    Parse the judge's reply and take action
```

### `benchmarker run` options

```
--tests PATH        Path to test suite JSON file or benchmarks/ directory
                    (default: benchmarks)
--categories TEXT   Comma-separated category slugs to load (directory mode only)
```

## Configuration file formats

### `benchmarks/` — test prompts

`benchmarker init` creates a `benchmarks/` directory containing category
subdirectories. Each category holds one JSON file per prompt.

```
benchmarks/
├── api-integration/
│   └── 001-fetch-json.json
├── bug-fixing/
│   └── 001-logic-error.json
├── code-generation/
│   ├── 001-chunk-seq.json
│   ├── 002-palindrome.json
│   ├── 003-twosum.json
│   └── 004-bst.json
├── comment-generation/
│   └── 001-complexity.json
├── general/
│   ├── 001-creative.json
│   ├── 002-reasoning.json
│   └── 003-factual.json
├── refactoring/
│   └── 001-type-hints.json
├── security-vulnerability/
│   └── 001-sql-injection.json
└── test-generation/
    └── 001-divide-list.json
```

`init` creates the 8 starter categories shown above. You can add more categories
by creating additional subdirectories under `benchmarks/`.

#### Single-prompt file format

Each `.json` file contains exactly one prompt object:

```json
{
  "id": "bugfixing",
  "prompt": "Explain the bug in the following code, then provide the corrected version...",
  "max_tokens": 2048,
  "repeat": 5,
  "reasoning": true,
  "stop": ["\ndef ", "\nclass "]
}
```

Supported fields:

- `id` (str, required): Unique identifier for the test case.
- `prompt` (str, required): The benchmark prompt. Must not be empty.
- `system` (str, optional): System message override.
- `max_tokens` (positive int, optional): Maximum tokens to generate.
- `repeat` (int, default 1): Number of times to repeat this test.
- `stop` (list[str], optional): Stop sequences.
- `reasoning` (bool, optional): True = encourage chain-of-thought, False = discourage, None = default.

#### `--categories` flag

Use `--categories` to run only a subset of categories:

```bash
benchmarker run --categories bug-fixing,refactoring
```

Validation rules:
- Slugs are matched **exactly** (case-sensitive) against the subdirectory names
  under `benchmarks/`.
- Invalid slugs raise an error listing the valid categories.
- Empty slugs (e.g. `--categories bug-fixing,,refactoring`) are rejected.
- `--categories` is only valid in directory mode. Passing it with a legacy flat
  `tests.json` file raises an error.

#### Backward compatibility

`tests.json` is no longer created by `init`, but the legacy flat-file format is
still supported. You can load an existing `tests.json` with:

```bash
benchmarker run --tests tests.json
```

When using a flat file, omit `--categories` — category filtering is only available
in directory mode.

#### Valid category slugs

| Slug | Description |
|---|---|
| `api-integration` | API integration and data fetching |
| `bug-fixing` | Debugging and logic error correction |
| `code-completion` | Autocomplete and inline suggestions |
| `code-generation` | Writing new functions and classes from descriptions |
| `code-review` | Reviewing code for issues and improvements |
| `code-summarization` | Summarizing code behavior and intent |
| `code-translation` | Translating code between languages |
| `cicd-pipeline-configuration` | CI/CD pipeline setup and configuration |
| `comment-generation` | Explaining code complexity and algorithms |
| `containerization` | Docker and container setup |
| `database-schema-design` | Database schema and migration design |
| `dead-code-elimination` | Removing unused code and imports |
| `dependency-management` | Managing dependencies and versions |
| `documentation-generation` | Generating docs and README content |
| `issue-triage` | Classifying and prioritizing issues |
| `log-analysis` | Analyzing application and system logs |
| `natural-language-to-code` | Converting requirements to implementation |
| `performance-optimization` | Profiling and optimizing code performance |
| `project-boilerplate` | Scaffolding new project structures |
| `refactoring` | Improving code structure and type safety |
| `repository-level-understanding` | Cross-file repo comprehension |
| `security-vulnerability` | Identifying and fixing security issues |
| `shell-script-generation` | Writing shell scripts and automation |
| `sql-query-generation` | Writing and optimizing SQL queries |
| `test-execution-failure-analysis` | Analyzing test failures and flakiness |
| `test-generation` | Generating unit tests and test suites |
| `type-annotation` | Adding and improving type hints |
| `general` | Non-coding prompts (creative, reasoning, factual) |

### `params.yaml` — search space

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
|------|--------------|-------------|
| **`grid`** | Enumerates every combination of parameter values. Exhaustive. | Small categorical grids where you want to test every point. |
| **`random`** | Samples random combinations until `budget` is exhausted. | Large numeric ranges where grid search would be too expensive. |
| **`bayesian`** | Builds a probabilistic model of the parameter space and focuses sampling on promising regions. Stops after `budget` trials. | Coarse exploration over wide ranges; typically finds the best region with fewer samples than grid. |
| **`baseline_sweep`** | Ablation study: varies one parameter at a time while holding others at a baseline. | Isolating the effect of a single parameter. Requires a `baseline` config. |

#### Budget

`budget` controls how many parameter configurations the optimizer will evaluate:

- **`grid`**: `budget` is **ignored**. The optimizer runs every possible combination. With 3 parameters × 3 values each, that’s 27 configs regardless of budget.
- **`random`**: Max samples before stopping. After `budget` configs, the optimizer raises `StopIteration`.
- **`bayesian`**: Max Optuna trials. After `budget` trials, the optimizer raises `StopIteration`.
- **`baseline_sweep`**: Not used; runs the full ablation set.

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

The `benchmarker` supports agent-specific benchmarking via the `system` field in test JSON files and per-agent judge criteria.

### Agent Benchmarks Structure

For multi-agent systems like [agent-hive](https://github.com/hung319/agent-hive), create agent-specific benchmark directories:

```
benchmarks/
├── hive/                          # Chief Planner & Orchestrator
│   ├── planning/
│   ├── orchestration/
│   └── approval/
├── architect/                     # Feature Architect (planner only)
│   ├── design/
│   ├── interviewing/
│   └── spec-writing/
├── swarm/                         # Execution Orchestrator
│   ├── delegation/
│   ├── verification/
│   └── parallel-execution/
├── scout/                         # Codebase & External Researcher
│   ├── codebase-exploration/
│   ├── external-research/
│   └── dependency-analysis/
├── forager/                       # Task Executor
│   ├── implementation/
│   ├── testing/
│   └── worktree-management/
├── hygienic/                      # Quality Reviewer
│   ├── plan-review/
│   └── code-review/
├── code-reviewer/                 # Code Review Specialist
│   └── diff-review/
├── code-simplifier/               # Code Simplification
│   ├── refactoring/
│   └── complexity-reduction/
├── codebase-analyzer/             # Codebase Analysis
│   └── structure/
├── codebase-locator/              # Code Location
│   └── semantic-search/
├── pattern-finder/                # Pattern Discovery
│   └── anti-patterns/
└── project-initializer/           # Project Setup
    └── scaffolding/
```

### System Prompts

Each test JSON should include a `system` field that mirrors the agent's actual system prompt from source. This ensures the benchmarked model adopts the correct persona, constraints, and output format.

Example:

```json
{
  "id": "forager-impl-001",
  "system": "You are Forager, an autonomous senior engineer and task execution agent...",
  "prompt": "Implement a function in Python...",
  "max_tokens": 2048,
  "repeat": 7,
  "reasoning": false
}
```

See `SYSTEM_PROMPTS.md` for verified system prompts from the agent-hive source.

### Reasoning Flag by Agent Type

| Agent Type | Reasoning | Rationale |
|------------|-----------|-----------|
| **Planning/Design** (Hive, Architect, Scout) | `true` | Needs to show thought process for plans and research |
| **Execution** (Forager, Project-Initializer) | `false` | Should output code directly, no reasoning needed |
| **Review** (Hygienic, Code-Reviewer) | `true` | Needs to explain reasoning for reviews |
| **Location/Analysis** (Codebase-Locator, Codebase-Analyzer, Pattern-Finder) | `false` | Should output structured data directly |
| **Simplification** (Code-Simplifier) | `true` | Should explain simplifications |
| **Orchestration** (Swarm) | `false` | Should output structured delegation plans |

### Repeat Counts by Agent

| Agent | Recommended `repeat` | Rationale |
|-------|---------------------|-----------|
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
benchmarker run --model my-model --categories hive/planning,hive/orchestration

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
