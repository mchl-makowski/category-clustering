# Service Desk Problem clustering

Groups Service Desk Tickets that report the same specific Problem (e.g. "VPN drops after client update to version X", not "VPN") and gives each Problem a shared Polish name. Three Methods are built behind one output format so they can be compared:

- **A**: embeddings + clustering (agglomerative or HDBSCAN), the baseline.
- **B**: coarse clustering + LLM verification, with split/merge until stable.
- **C**: a Ticket-by-Ticket tool-calling agent with lookup and periodic consolidation.

A fixed **Judge** LLM checks every run for Mixed Problems and Split Problems in place of labels. Terms follow [`GLOSSARY.md`](GLOSSARY.md); requirements are in [`TASK.md`](TASK.md); the runtime decision is [ADR-0001](docs/adr/0001-azure-foundry-runtime.md).

Only `Tytuł` and `Opis` reach any model. Emails, phone numbers and PESEL numbers are masked in model input. The output keeps the original text and every original column.

## Setup

Python 3.11.

**Dev machine (uv):**

```bash
uv sync                 # creates .venv with Python 3.11, installs the package + pytest
uv run pytest           # whole pipeline offline on the fixture, Fake backend
```

**Azure Foundry (pip):** `requirements.txt` holds exact versions exported from `uv.lock`. It does not list the package itself, so install that without deps:

```bash
pip install -r requirements.txt
pip install --no-deps -e .
```

After changing dependencies, regenerate the pinned file:

```bash
uv export --no-dev --no-hashes --no-emit-project --format requirements-txt -o requirements.txt
```

### Environment variables (Azure Foundry backend)

Read by a pydantic-settings class (`backend/azure.py`, prefix `FOUNDRY_`), from the environment or a `.env` file in the working directory (`.env` is git-ignored). Never put them in YAML.

| Variable | Required | Meaning |
|---|---|---|
| `FOUNDRY_API_KEY` | yes | API key of the Foundry / Azure OpenAI resource |
| `FOUNDRY_ENDPOINT` | yes | Resource base URL, e.g. `https://<resource>.openai.azure.com/` (or `*.cognitiveservices.azure.com`, `*.services.ai.azure.com`); no `/openai/...` path. A non-https URL or a path fails at startup, an unknown host warns |
| `FOUNDRY_API_VERSION` | no (default `2024-10-21`) | Azure OpenAI API version |
| `FOUNDRY_EMBEDDING_DEPLOYMENT` | no | Overrides `backend.embedding_deployment` |
| `FOUNDRY_CHAT_DEPLOYMENT` | no | Overrides `backend.chat_deployment` |
| `FOUNDRY_JUDGE_DEPLOYMENT` | no | Overrides `judge.deployment` |

Deployment names are not secrets: they default to the YAML values (`backend.*`, `judge.deployment`). A set `FOUNDRY_*_DEPLOYMENT` variable wins over YAML; an explicit `--set` / notebook override wins over the variable. The Fake backend (`backend.kind: fake`) needs no env vars.

## Usage

### CLI

```bash
# Run one Method (writes outputs/yymmdd_hhmm_<method>.xlsx + .txt, Judge included)
problem-clustering run --method a --input export.xlsx --config config.example.yaml
problem-clustering run --method b --input export.xlsx --config config.example.yaml
problem-clustering run --method c --input export.xlsx --config config.example.yaml

# Override any YAML value with a dotted key (repeatable); --no-judge skips the Judge
problem-clustering run --method a --input export.xlsx --config config.example.yaml \
    --set granularity=fine --set method_a.distance_threshold=0.3 --no-judge

# Put the Metrics of several runs side by side (writes outputs/yymmdd_hhmm_comparison.xlsx)
problem-clustering compare outputs/261115_1200_a.xlsx outputs/261115_1210_b.xlsx outputs/261115_1230_c.xlsx

# Console verbosity: -v (DEBUG) or -q (WARNING) override logging.level; --no-progress = logging.progress=off
problem-clustering run --method c --input export.xlsx --config config.example.yaml -v --no-progress

# Offline smoke run on the fixture
uv run problem-clustering run --method c --input tests/fixtures/tickets.xlsx \
    --config config.example.yaml --set backend.kind=fake --set runtime.cache_dir=null
```

With uv, prefix with `uv run`; `python -m problem_clustering ...` works too. Both subcommands take `--out-dir` (default `outputs`). `--config` is optional; without it all defaults below apply.

CLI flags for console output (see `logging` below):

| Flag | Meaning |
|---|---|
| `-v` / `--verbose` | Console at DEBUG (per-Ticket / per-Problem detail); overrides `logging.level` |
| `-q` / `--quiet` | Console at WARNING (retries, content-filter fallbacks, Judge failure); overrides `logging.level` |
| `--no-progress` | `run` only: no progress bars, same as `--set logging.progress=off` |

`-v` and `-q` are mutually exclusive; both subcommands take them.

### Outputs

- `yymmdd_hhmm_<method>.xlsx` with sheets:
  - `Tickets`: original columns unchanged, plus `problem_id` (`A-0001`, `B-…`, `C-…`; `X-0000` "Brak opisu" for Tickets with empty `Tytuł` and `Opis`, which skip every model call), `problem_name`, `confidence` (A/B: cosine similarity to the Problem centroid, 1.0 for one-Ticket Problems; C: value of the Confidence Level) and `confidence_level` (C only: pewne 1.0, wysokie 0.8, średnie 0.6, niskie 0.4, zgadywane 0.2).
  - `Problem Catalogue`: `problem_id`, `problem_name`, `description` (Polish), `ticket_count`, `example_tickets`.
  - `Judge`: one row per Mixed check and per Split pair, with verdict and reason.
  - `Metrics`: Problem count, one-Ticket share, size distribution, purity, split rate, LLM calls, tokens, embedding calls, runtime (for the Method and the Judge separately). No cost estimate.
- `yymmdd_hhmm_<method>.txt`: run log, one `[HH:MM:SS] LEVEL message` line per record, always at DEBUG whatever the console level (config, ~10% progress lines, Problem counts, iterations / consolidations, retries, totals; Judge lines appended). No progress bar frames.
- `compare` writes `yymmdd_hhmm_comparison.xlsx` (sheet `Comparison`: one row per metric, one column per run).

### Configuration (YAML)

Copy [`config.example.yaml`](config.example.yaml) and edit it. Every section rejects unknown keys, so typos fail loudly. Every key can be overridden from the CLI (`--set section.key=value`, values parsed as YAML: `null`, `true`, `0.3`) or the notebook (`OVERRIDES`).

**`granularity`**: the **Granularity Level**, `coarse` | `medium` (default) | `fine`. It sets the Problem definition paragraph in every LLM prompt of the run: naming, Symptom Summary, Method B verification/merge, Method C agent/consolidation and the Judge. The numeric knobs below change the clustering granularity per Method; for a fair comparison, keep the same level across Methods and the Judge.

**`backend`**

| Key | Default | Meaning |
|---|---|---|
| `kind` | `azure` | `azure` (Foundry, env vars above) or `fake` (offline, deterministic) |
| `embedding_deployment` | `text-embedding-3-large` | Embedding deployment name |
| `chat_deployment` | `gpt-4o` | Chat deployment used by the Methods and naming |
| `embed_batch_size` | `64` | Texts per embedding request |
| `embed_max_chars` | `8000` | Each text is cut to this many chars before embedding (`text-embedding-3-large` takes at most 8191 tokens); LLM prompts are not cut |

**`naming`**: shared Polish naming step for every Problem.

| Key | Default | Meaning |
|---|---|---|
| `sample_size` | `8` | Tickets shown to the LLM when naming a Problem |

**`symptom_summary`**: optional **Symptom Summary**. When enabled, an LLM rewrites each non-empty Ticket into one Polish line (symptom + context) and that line, not the raw text, is what the Method groups. Naming and the output still use the original (masked) text.

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `false` | Turn the Symptom Summary on |
| `log_summaries` | `false` | Write each Ticket's summary into the `.txt` log |

**`method_a`**

| Key | Default | Meaning |
|---|---|---|
| `algorithm` | `agglomerative` | `agglomerative` or `hdbscan` |
| `distance_threshold` | `0.35` | Agglomerative cosine distance; lower = finer Problems |
| `linkage` | `average` | `average` / `complete` / `single` |
| `min_cluster_size` | `2` | HDBSCAN only; noise Tickets become one-Ticket Problems |
| `min_samples` | `null` | HDBSCAN only; `null` = `min_cluster_size` |

**`method_b`**

| Key | Default | Meaning |
|---|---|---|
| `coarse_threshold` | `0.6` | Cosine distance of the starting coarse clusters (looser than A) |
| `sample_size` | `10` | Tickets shown to the LLM per verification / merge check |
| `split_step` | `0.1` | A mixed cluster is re-clustered at its threshold minus this |
| `merge_neighbours` | `1` | Nearest-centroid neighbours LLM-checked for a merge, per Problem |
| `max_iterations` | `5` | Stop after this many iterations (or earlier when nothing changes) |
| `linkage` | `average` | Linkage of the agglomerative steps |

**`method_c`**

| Key | Default | Meaning |
|---|---|---|
| `max_tool_calls` | `6` | Per Ticket; then the Ticket gets a new Problem (zgadywane) |
| `search_k` | `5` | Default `k` for `search_similar_tickets` |
| `search_max_k` | `20` | Hard cap on `k` the agent may request |
| `consolidate_every` | `250` | Tickets between consolidations (one also runs at the end) |
| `consolidate_threshold` | `0.5` | Cosine distance between Problem centroids to be merge candidates |
| `consolidate_group_size` | `15` | Max Problems per consolidation request |
| `example_tickets` | `5` | Ticket texts returned by `get_problem` |

**`judge`**: fixed Judge model, independent of the run's chat model.

| Key | Default | Meaning |
|---|---|---|
| `deployment` | `gpt-4o` | Judge chat deployment (keep it fixed across compared runs) |
| `sample_size` | `8` | Tickets per Mixed check |
| `split_sample_size` | `4` | Tickets per Problem in a Split check |
| `neighbours` | `3` | Nearest Problems (by centroid) checked for a Split Problem; pairs deduplicated |
| `max_problems` | `null` | Judge only the N largest Problems (`null` = all) |

**`output`**

| Key | Default | Meaning |
|---|---|---|
| `example_tickets` | `5` | Example Tickets per Problem in the Problem Catalogue |

**`runtime`**: cache, parallelism, retries around any backend.

| Key | Default (`config.example.yaml`) | Meaning |
|---|---|---|
| `cache_dir` | `.cache/problem_clustering` (code default `null`) | Disk cache of embeddings and LLM replies; `null` disables it |
| `max_concurrency` | `8` | Parallel independent model calls |
| `max_retries` | `5` | Retries on rate limit / timeout / 5xx, then the error is raised |
| `backoff_initial_s` | `1.0` | Wait before retry 1 |
| `backoff_multiplier` | `2.0` | Wait multiplier per retry |
| `backoff_max_s` | `60.0` | Max wait between retries |

**`logging`**: console output of `run` / `compare` and the notebook (`setup_logging(config)`). Without `setup_logging` (library use, tests) nothing is printed. Console lines are `HH:MM:SS LEVEL message`, on stderr; they carry row numbers, Problem ids/names, Confidence Levels and counts, never Ticket text (`symptom_summary.log_summaries` lines go to the `.txt` only).

| Key | Default | Meaning |
|---|---|---|
| `level` | `INFO` | Console level: `DEBUG` (per-Ticket / per-Problem detail, config JSON), `INFO` (steps, processed/total, Problem counts), `WARNING` (retries, content-filter fallbacks, Judge failure), `ERROR`. CLI `-v` / `-q` override it |
| `progress` | `auto` | Progress bars for long loops (embeddings, Symptom Summary, Method B verification / merge checks, Method C Tickets, naming, Judge checks): `auto` = a bar when stderr is a terminal or in Jupyter, otherwise none (Foundry jobs, CI); `on`; `off`. Without a bar an INFO processed/total line is logged about every 10%; each loop ends with an INFO count line |

### Content filter

When the Azure content filter blocks a prompt (HTTP 400 `content_filter`) or a reply (`finish_reason=content_filter`, refusal), the run does not stop. The step falls back and logs a WARNING `content filter` line (console and `.txt`): Symptom Summary uses the raw masked text; naming names the Problem after its first Ticket's title; Method B keeps a cluster as one Problem and does not merge; the Method C agent gives the Ticket a new Problem (`zgadywane`) and consolidation skips that group; a Judge check gets verdict `error` and is left out of purity and split rate. Retries happen only in `runtime` (the SDK's own retries are off).

### Cache

With `runtime.cache_dir` set, every embedding (per text) and every chat reply (per full request: messages, tools, response format, model) is stored on disk under a content hash of backend kind, model and input. A rerun or parameter sweep only pays for calls whose input changed; for example, rerunning Method A with another `distance_threshold` reuses all embeddings. The path is relative to the working directory (`.cache/` is git-ignored). Delete the directory to force fresh calls. Usage counters in `Metrics` are counted outside the cache, so a cached rerun reports the same LLM calls/tokens (comparable across Methods); the `.txt` log says how many calls actually reached the model.

### Notebook

[`notebooks/compare.ipynb`](notebooks/compare.ipynb) reproduces the comparison on a given sample. It loads the same YAML, calls only the public API, and:

1. runs Methods A/B/C with the Judge, or loads earlier run workbooks (`WORKBOOKS = {'c': 'outputs/…_c.xlsx'}`),
2. shows the `compare` table and each Method's Judge verdicts,
3. browses each Method's Problems: name, description and their Tickets.

It calls `setup_logging(config)`, so progress shows as a `tqdm` widget (needs `ipywidgets`, installed with the package) and INFO lines appear under each cell. It defaults to the fixture and the Fake backend, so it runs offline top to bottom. On Foundry, edit the first code cell: `INPUT` = your export, `OVERRIDES = {}` (or your overrides), optionally `CONFIG`, `METHODS`, `BROWSE_PROBLEMS`.

```bash
uv run --with jupyter jupyter lab notebooks/compare.ipynb   # dev machine
```

On Foundry, open it in the workspace's Jupyter after the pip install above.

## Layout

- `src/problem_clustering/`: the one shared module. Public API in `__init__.py`: `load_tickets`, `load_config`, `run_method`, `judge`, `judge_safely` (Judge errors go to the run log instead of losing the run), `write_outputs`, `compare`, `setup_logging` (console level + progress bars; `logs.py`); `backend.make_backend`.
- `cli.py`, `notebooks/compare.ipynb`: thin wrappers, no Method logic.
- `tests/`: pytest on the public API with the Fake backend and `tests/fixtures/tickets.xlsx` (~20 hand-written Tickets, regenerated by `make_fixture.py`).
