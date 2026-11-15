# Service Desk ticket classifier: clustering tickets that describe the same problem

## Goal

We have a Service Desk export in Excel. We want to find tickets that describe **the same specific problem** and give them a shared label. That means fine-grained groups like "VPN drops after client update to version X", not broad categories like "VPN".

This is an exploratory task. We'll build three approaches, compare them, and pick a direction based on the results.

## Input data

An Excel file with these columns: `Typ`, `Numer I/SR`, `Zgłoszenie`, `Tytuł`, `Opis`, `Status`, `Grupa odpowiedzialna`.

- Classification uses **only** `Tytuł` (title) and `Opis` (description).
- `Numer I/SR` is the ticket ID. All other columns pass through to the output unchanged.
- No synthetic dataset is built. The development machine has no LLM API access, so the requester runs everything, including evaluation, on the sample Excel in Azure Foundry.

## Approaches to build

**A. Embeddings + clustering (baseline).** Embed title + description and cluster the embeddings, with a parameter that controls granularity.

**B. Clustering + LLM verification.** Start from coarser clusters. Use an LLM on a small sample from each cluster to check whether all its tickets describe the same problem. Split clusters that don't, and optionally merge clusters that turn out to be the same problem. Repeat until the clusters are stable.

**C. Ticket-by-ticket agent with lookup.** Go through the tickets one at a time while building a catalogue of problems. For each ticket, the agent can look up similar tickets that are already classified, then assigns the ticket to an existing problem or creates a new one. The catalogue is consolidated from time to time so duplicate problems get merged.

## Deliverables

- A **CLI script** that runs any of the three methods against the sample Excel.
- A **Jupyter notebook** for exploring and comparing the methods.
- Both sit on top of one shared Python module. The CLI and the notebook are thin wrappers around it.
- Output for each method:
    - the tickets file with the original columns plus `problem_id`, `problem_name` and, where available, a confidence score,
    - a problem catalogue listing each problem's name, description, ticket count and example tickets.
    - Output should be an XLSX files, optionally some logs in the txt file

## Evaluation

There is no hand-labelled sample. Each method is compared through manual review of its Problems plus an LLM judge, on:

- clustering quality: are different problems mixed in one cluster, and is one problem split across several clusters?
- LLM calls, token usage and runtime (no cost estimate).

## Acceptance criteria

- \[ \] All three methods run from the CLI and produce output in the same format.
- \[ \] The notebook reproduces the comparison on a given sample.
- \[ \] Only `Tytuł` and `Opis` are used for classification.
- \[ \] Granularity can be adjusted through parameters.
- \[ \] A README covers setup and usage.

## Out of scope

- Integration with the Service Desk system or live classification.
- Routing tickets to resolver groups.
- Any UI.

## Decided during planning (2026-11-15)

Terms are defined in `GLOSSARY.md`. The runtime and model constraint is in `docs/adr/0001-azure-foundry-runtime.md`.

- **Runtime:** a quick-win script runs on Azure Foundry compute against an uploaded Excel. A working solution matters more than a sophisticated one.
- **Models:** text-embedding-3-large and GPT-4/5-series deployments on Foundry, all configurable. Auth is an API key read from env vars by a pydantic-settings class.
- **Volume:** about 2500 Tickets per run, in Polish only. Method C is affordable at that volume.
- **Problem definition:** the same observable symptom in the same context. One-Ticket Problems are allowed. Tickets with an empty `Tytuł` and `Opis` go to a reserved "Brak opisu" Problem and skip all model calls.
- **Text:** no cleaning. Minimal regex PII masking (email, phone, PESEL) applies to model input only, and output keeps the original text. The Symptom Summary rewrite is optional and switched on by a parameter. When it is on, the summary is what gets grouped.
- **Method A:** agglomerative (cosine distance threshold) or HDBSCAN, configurable. HDBSCAN noise becomes one-Ticket Problems.
- **Method B:** start from coarse clusters (a looser threshold). The LLM checks up to 10 sampled Tickets per cluster. A mixed cluster is re-clustered at a tighter threshold. Merges are LLM-checked between nearest-centroid neighbours. Stop when nothing changes, or after at most 5 iterations.
- **Method C:** a tool-calling agent, working in file order. Tools: `search_similar_tickets(query, k)`, `get_problem(problem_id)`, and a final `assign_to_problem(problem_id, confidence_level)` or `create_problem(name, description, confidence_level)`. After 6 tool calls without a decision, the Ticket gets a new Problem. Consolidation (merging near-duplicate Problems) runs every 250 Tickets and at the end.
- **Granularity:** one shared Granularity Level (coarse / medium / fine) sets the Problem definition in every LLM prompt, the Judge included. Each Method also has its own numeric knobs in YAML.
- **Confidence:** `confidence` is a number for every Method. For A and B it is the cosine similarity to the Problem centroid (1.0 for one-Ticket Problems). The LLM picks one of 5 Confidence Levels: pewne 1.0, wysokie 0.8, średnie 0.6, niskie 0.4, zgadywane 0.2. The label goes in `confidence_level`.
- **Naming:** a shared LLM step names every Problem and writes its description, in Polish.
- **Evaluation:** there are no labels. A fixed Judge model checks every multi-Ticket Problem for a Mixed Problem (up to 8 sampled Tickets). It checks each Problem against its 3 nearest Problems for a Split Problem (pairs deduplicated). An optional `judge.max_problems` cap applies. Reported per Method: Problem count, share of one-Ticket Problems, size distribution, Judge purity and split rate, LLM calls, tokens, runtime. The notebook includes a browser of each Method's Problems. No cost estimate.
- **CLI/config:** YAML config with CLI overrides. Subcommands `run --method a|b|c` and `compare`. The notebook loads the same YAML.
- **Output:** flat files `yymmdd_hhmm_<method>.xlsx` with sheets `Tickets`, `Problem Catalogue`, `Judge` and `Metrics`, plus a `yymmdd_hhmm_<method>.txt` log. `compare` writes `yymmdd_hhmm_comparison.xlsx`. problem_id values carry the Method prefix (`A-0001`), and `X-0000` is "Brak opisu".
- **Engineering:** a pyproject package managed with uv on Python 3.11, with a pinned `requirements.txt` exported for Foundry. Embeddings and LLM replies are cached on disk by content hash. Independent calls run in parallel up to a concurrency limit, with retry/backoff. A fake model backend and a roughly 20-Ticket hand-written fixture let pytest run the whole pipeline offline.
