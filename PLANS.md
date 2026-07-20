# PROJECTS.md – LLM Parameter Benchmarker

## Project Overview

A command-line tool to find the optimal sampling parameters for a model running in `llama-server` (OpenAI‑compatible API).  
It sends a suite of test prompts while varying parameters (temperature, top_k, top_p, min_p, repetition_penalty, etc.),  
measures speed metrics (TTFT, tokens/s), and saves all responses into a Markdown file for external LLM‑as‑a‑judge evaluation.  
After the user obtains quality scores from a remote judge, the tool re‑ingests them and computes a combined speed‑quality ranking,  
outputting the best parameter set.

**Core commands**:
```
benchmarker run --model <name> [--tests tests.json] [--params params.yaml] [--url ...]
benchmarker import-scores <scores.json> --run-dir <dir>
```

**Key design choices**:
- Test suite: JSON array of prompts (with optional `system`, `max_tokens`, `repeat`).
- Parameter search space: YAML config, default Bayesian optimisation (optuna) with configurable budget.
- Streaming enabled to capture time‑to‑first‑token.
- Semi‑automated quality: tool generates `eval_output.md` containing all prompts & responses, plus a rating template; user copies it to an external LLM, pastes back a JSON scores file, then imports.
- Test‑driven development (TDD) throughout.

---

## Technology Stack
- **Python 3.11+**
- CLI: `click` (or `argparse` – we’ll use `click` for composability)
- HTTP client: `httpx` (async)
- Configuration validation: `pydantic`
- Optimisation: `optuna` (for Bayesian), plus own grid/random implementations
- Progress display: `rich`
- Testing: `pytest`, `pytest-asyncio`, `pytest-mock`, `respx` (HTTP mocking)
- YAML parsing: `pyyaml` (or `ruamel.yaml`)
- JSON handling: built‑in

---

## Phase 1: Project Scaffolding & Basic CLI

**Goal**: Create a runnable package with a no‑op command that prints “Hello, world”, tested.

### Tasks
1. Set up project structure:
   ```
   benchmarker/
   ├── pyproject.toml
   ├── src/
   │   └── benchmarker/
   │       ├── __init__.py
   │       └── cli.py
   └── tests/
       ├── __init__.py
       └── test_cli.py
   ```
2. Install `pytest`, `click`, `pytest-mock`.
3. Implement `cli.py` with a `main()` entry point using `click.group()` and a `run` subcommand that echoes the model name.
4. Write **unit test**: `test_cli.py::test_run_prints_model` – invoke via `CliRunner` and assert output contains the model name.
5. Wire up `pyproject.toml` with `[project.scripts]`: `benchmarker = "benchmarker.cli:main"`.
6. Verify `pytest` passes.

**Deliverable**: a working CLI that can be executed with `benchmarker run --model test` and prints something.

---

## Phase 2: Configuration Management

**Goal**: Parse the parameter search space YAML and test suite JSON using Pydantic models, with full validation.

### Tasks
1. Define Pydantic models:
   - `ParameterSpec`: name, type (`float`, `int`, `categorical`), low/high (or choices), optional step.
   - `OptimizerConfig`: type (`bayesian`, `grid`, `random`), budget (int).
   - `ParamsConfig`: optimizer section, list of `ParameterSpec`.
   - `TestCase`: id, prompt, optional system, max_tokens, repeat (default 1).
   - `TestSuite`: list of `TestCase`.
2. Implement `benchmarker/config.py` with functions:
   - `load_params(path: Path) -> ParamsConfig` (YAML)
   - `load_tests(path: Path) -> TestSuite` (JSON)
3. Write **unit tests**:
   - Valid YAML file → ParamsConfig object.
   - Invalid YAML (missing optimizer) → raises `ValidationError`.
   - Valid JSON test suite → TestSuite.
   - Test default values (e.g., `repeat` defaults to 1).
4. Integrate with CLI: add `--params` and `--tests` arguments (default `params.yaml`, `tests.json`) that load the configs and print a summary (just to confirm loading, later will be used).

**Deliverable**: config loading and validation fully tested.

---

## Phase 3: Test Suite Loader & Validation

**Goal**: Ensure the test suite loader works with edge cases and can be used later.

### Tasks
1. Extend `load_tests` to handle non‑existent file gracefully (raise `FileNotFoundError` with message).
2. Validate that each test has a unique `id` (deduplication) – add a Pydantic validator.
3. Ensure `max_tokens` is positive integer if provided.
4. **Unit tests**: duplicate IDs, missing prompt field, empty suite.
5. At this stage the CLI’s `run` command will load the configs and print the number of tests loaded.

**Deliverable**: robust test suite loading.

---

## Phase 4: LLM Client (Async Streaming with Metrics)

**Goal**: An async client that sends chat completion requests to `llama-server`, streams the response, and records TTFT, total tokens, total time, and full text.

### Tasks
1. Create `benchmarker/client.py` with `LLMClient` class:
   - Constructor takes `base_url` (default `http://localhost:8080/v1/chat/completions`), `api_key` (optional), timeout.
   - Async method `complete(messages, model, **params) -> CompletionResult` where `CompletionResult` is a Pydantic model:
     - `prompt_tokens: int`
     - `completion_tokens: int`
     - `response_text: str`
     - `ttft: float` (seconds)
     - `total_time: float`
     - `tokens_per_sec: float`
2. Implementation details:
   - Use `httpx.AsyncClient` with `stream=True`.
   - Record `start_time = time.monotonic()`, then iterate chunks.
   - On first chunk with content, record `ttft`.
   - Accumulate text from chunks.
   - At end, extract `usage` from final chunk (or count tokens if not present – but llama.cpp returns usage). Compute `total_time`, `tokens_per_sec`.
3. **Test with `respx`** mock:
   - Mock a streaming response that sends multiple chunks, with `usage` in the last chunk.
   - Verify TTFT, total time, text concatenation, token counts.
   - Test error handling (HTTP 500) – should raise custom `LLMClientError`.
   - Test timeout.
4. Ensure client is reusable and handles sessions properly.

**Deliverable**: fully tested async LLM client with streaming metrics.

---

## Phase 5: Optimizer Abstraction & Implementations

**Goal**: Provide three search strategies (grid, random, bayesian) that yield parameter sets from the search space.

### Tasks
1. Define abstract `BaseOptimizer` with method `suggest() -> dict[str, Any]` and `tell(metrics: dict)` (for bayesian feedback).
2. Implement `GridOptimizer`:
   - Takes list of `ParameterSpec`, constructs all combinations using `itertools.product`.
   - `suggest()` returns next combination, raises `StopIteration` when done.
3. Implement `RandomOptimizer`:
   - Takes parameter specs and a `budget`; `suggest()` samples randomly from the range (uniform for floats/ints, choice for categorical). Tracks count.
4. Implement `BayesianOptimizer` using `optuna`:
   - At init, create an Optuna study (direction = "maximize" for our objective, which will be speed initially).
   - `suggest()` samples hyperparameters using `optuna.trial` API (define suggest_float, suggest_int, suggest_categorical).
   - `tell(metrics)` – create a frozen trial with the objective value (tokens/s or penalized speed) and tell the study.
   - Because we need to run trials sequentially, we’ll use `optuna`’s `enqueue_trial` and `optimize` with a custom callback that yields control. Simpler: we’ll manually manage trials using `study.ask()` and `study.tell()` for explicit control.
5. **Unit tests**:
   - For grid: verify all combos are generated.
   - For random: check budget exhaustion and value ranges.
   - For bayesian: mock Optuna’s `Study` to verify that `ask()` and `tell()` are called correctly, and that parameters are within bounds.
6. Add a factory function `create_optimizer(config: OptimizerConfig, params: list[ParameterSpec]) -> BaseOptimizer`.

**Deliverable**: pluggable optimizers, tested independently.

---

## Phase 6: Runner – Core Orchestration

**Goal**: Combine the LLM client, test suite, and optimizer to execute the benchmark and save raw results.

### Tasks
1. Create `benchmarker/runner.py` with `Runner` class:
   - `__init__(client, test_suite, optimizer, model_name, run_dir)`
   - `async run()` that:
     - Creates `run_dir` if not exist.
     - Iterates over optimizer trials: for each suggestion, run all tests × repeat.
     - For each request, call `client.complete()`, store result in a list.
     - Computes average tokens/s for the config (to feed back to optimizer if Bayesian).
     - Saves incremental progress (raw data) as JSON lines or a list in `raw_data.json`.
   - On completion, save final `raw_data.json` containing all `RunResult` objects.
2. `RunResult` data class:
   - config (dict), test_id, repetition, prompt, response_text, ttft, total_time, tokens_per_sec, completion_tokens, prompt_tokens.
3. Implement **unit tests** with mocked client and a tiny test suite:
   - Verify runner calls client correct number of times (tests × repeat × trials).
   - Verify results are saved to file.
   - Test that Bayesian optimizer receives speed metrics.
   - Test error resilience (client fails once, retries then proceeds; record failure).
4. CLI `run` command now instantiates client, loads configs, creates runner, and awaits `run()`. After run, it prints summary speed ranking.

**Deliverable**: functional benchmark run with speed metrics, raw data saved.

---

## Phase 7: Evaluation File Generation

**Goal**: Produce a Markdown file that contains all prompts and responses, along with a rating template for the judge LLM.

### Tasks
1. Create `benchmarker/eval_file.py` with function `generate_eval_md(run_dir: Path, run_results: list[RunResult])`:
   - For each unique parameter configuration (group by config), write a section.
   - Within each config, list test_id and repetition, prompt, response, separated by horizontal rules.
   - At the end, append rating instructions (see design).
2. Use Jinja2 or simple string formatting to produce the template.
3. **Unit test**: Given a small set of run results, check that the output contains expected headings, prompts, responses, and the rating block.
4. Integrate into runner: after run completes, call `generate_eval_md` and save to `eval_output.md` in run directory.
5. Also print a friendly message telling the user to copy the file content to a judge LLM.

**Deliverable**: eval file generated after each run.

---

## Phase 8: Import Scores & Final Ranking

**Goal**: Accept a JSON scores file (from the judge), merge with raw data, compute final combined score, and print ranking.

### Tasks
1. Define Pydantic model for `JudgeScoreEntry`:
   - config (string key matching printed config), test_id, repetition, scores (dict with "overall" etc).
   - The whole file is a list of such entries.
2. Create `benchmarker/importer.py`:
   - `import_scores(run_dir: Path, scores_path: Path, weight_quality: float = 0.5)`:
     - Load raw data from `raw_data.json`.
     - Load scores JSON, validate.
     - Merge by matching config (serialized), test_id, rep.
     - For each config, compute average overall score and average tokens/s.
     - Normalize both to [0,1] using min‑max across all configs.
     - Calculate `combined = weight_quality * norm_quality + (1 - weight_quality) * norm_speed`.
     - Sort configs by combined descending.
     - Return a list of `FinalRankingItem`.
3. **Unit test**: provide sample raw data and sample scores, verify merging, normalization, ranking.
4. Add `benchmarker import-scores` CLI command:
   - `benchmarker import-scores scores.json --run-dir <dir> [--weight-quality 0.5]`
   - Prints a `rich` table of final ranking (config, quality, speed, combined).
   - Optionally saves CSV.

**Deliverable**: final combined ranking works.

---

## Phase 9: Reporting & Console Output

**Goal**: Enhance CLI with beautiful progress and summary tables using `rich`.

### Tasks
1. In runner, use `rich.progress` to show a progress bar of trials and tests.
2. After run (speed‑only), print a table of top 5 configs sorted by tokens/s.
3. After import‑scores, print a full ranking table with columns: config, avg quality, avg tokens/s, combined score.
4. Add CSV export option to both commands (optional).
5. **Test** (manually or via capturing console output) – we can rely on integration tests later.

**Deliverable**: polished user interface.

---

## Phase 10: Default Test Suite & Example Params

**Goal**: Ship sensible defaults so tool works out of the box.

### Tasks
1. Create `benchmarker/defaults/tests.default.json` containing the four prompt categories we discussed.
2. Create `benchmarker/defaults/params.default.yaml` with the base parameter ranges.
3. Update CLI to fall back to these defaults if `--tests` or `--params` not provided (look in package resources). Use `importlib.resources`.
4. **Test**: invoke without files and confirm defaults are loaded.

**Deliverable**: ready‑to‑use tool.

---

## Phase 11: Integration & End‑to‑End Testing

**Goal**: Validate the full workflow with a real (or mocked) `llama-server`.

### Tasks
1. Write an integration test using `pytest` and `respx` to mock all HTTP calls.
   - Set up a mock server that returns realistic streaming responses.
   - Run `benchmarker run` with a tiny test suite and a grid of 2 parameter sets.
   - Check that `raw_data.json` and `eval_output.md` are created with expected content.
2. Simulate the import flow: run the import command with a crafted `scores.json`, verify ranking output.
3. Test error paths: invalid API key, server returning errors, etc.
4. (Optional) Manual test with a real `llama-server` instance.

**Deliverable**: automated integration tests pass.

---

## How to Use This File for TDD

Each phase:
1. **Read the tasks** – understand what needs to be built.
2. **Write the tests first** – create `tests/test_<module>.py` with failing tests that match the expected behaviour.
3. **Implement** the minimal code to pass tests.
4. **Refactor** while keeping tests green.
5. **Check off** the phase once all tests pass and the feature integrates with the CLI.

By following this PROJECTS.md, an AI coder (or human) can step through each phase, always having clear validation criteria.
