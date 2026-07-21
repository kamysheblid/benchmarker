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
