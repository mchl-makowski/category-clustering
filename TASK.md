# Service Desk ticket classifier: clustering tickets that describe the same problem

## Goal

We have a Service Desk export in Excel. We want to find tickets that describe **the same specific problem** and give them a shared label. That means fine-grained groups like "VPN drops after client update to version X", not broad categories like "VPN".

This is an exploratory task. We'll build three approaches, compare them, and pick a direction based on the results.

## Input data

An Excel file with these columns: `Typ`, `Numer I/SR`, `Zgłoszenie`, `Tytuł`, `Opis`, `Status`, `Grupa odpowiedzialna`.

- Classification uses **only** `Tytuł` (title) and `Opis` (description).
- `Numer I/SR` is the ticket ID. All other columns pass through to the output unchanged.
- Development runs on synthetic data. Evaluation on the real sample is done by the requester.

## Approaches to build

**A. Embeddings + clustering (baseline).** Embed title + description and cluster the embeddings, with a parameter that controls granularity.

**B. Clustering + LLM verification.** Start from coarser clusters. Use an LLM on a small sample from each cluster to check whether all its tickets describe the same problem. Split clusters that don't, and optionally merge clusters that turn out to be the same problem. Repeat until the clusters are stable.

**C. Ticket-by-ticket agent with lookup.** Go through the tickets one at a time while building a catalogue of problems. For each ticket, the agent can look up similar tickets that are already classified, then assigns the ticket to an existing problem or creates a new one. The catalogue is consolidated from time to time so duplicate problems get merged.

## Deliverables

- A **CLI script** that runs any of the three methods against a labelled, syntetic sample.
- A **Jupyter notebook** for exploring and comparing the methods.
- Both sit on top of one shared Python module. The CLI and the notebook are thin wrappers around it.
- Output for each method:
    - the tickets file with the original columns plus `problem_id`, `problem_name` and, where available, a confidence score,
    - a problem catalogue listing each problem's name, description, ticket count and example tickets.
    - Output should be an XLSX files, optionally some logs in the txt file

## Evaluation

Each method is compared against a hand-labelled sample on:

- clustering quality: are different problems mixed in one cluster, and is one problem split across several clusters?
- cost and runtime.

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

## To be decided during planning

These details are left open on purpose and will be settled in the planning phase:

- which embedding model and LLM to use (local or API), and any data constraints,
- the specific clustering algorithms and how to tune them,
- the preprocessing and text-cleaning steps,
- the exact metrics and the size of the evaluation sample,
- the expected data volume, and whether approach C is affordable at that volume,
- the CLI interface and the configuration format.
