# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

Pre-implementation. The only source of truth so far is `TASK.md` (the requirements spec). There is no code, build system, test suite or README yet. Several design choices are left open on purpose for the planning phase (see "Open decisions" below). Don't treat them as settled until they appear in code or docs, and update this file once they are decided.

## What this project is

An exploratory Service Desk ticket classifier. It groups tickets that describe **the same specific problem** (fine-grained, e.g. "VPN drops after client update to version X", not "VPN") and gives each group a shared label. Three approaches are built and compared:

- **A. Embeddings + clustering (baseline).** Embed title + description, then cluster with a granularity parameter.
- **B. Clustering + LLM verification.** Start with coarse clusters. An LLM checks a sample from each cluster to see whether its tickets describe one problem, then clusters are split (and optionally merged) until they stop changing.
- **C. Ticket-by-ticket agent with lookup.** Process tickets one at a time while building a problem catalogue. For each ticket, look up similar tickets that are already classified, then assign it to an existing problem or create a new one. Consolidate the catalogue from time to time to merge duplicates.

## Required architecture

- **One shared Python module** holds all the logic. The **CLI script** and the **Jupyter notebook** are thin wrappers around it, so don't put method logic in either one.
- All three methods must produce **the same output format**, so they can be swapped and compared:
  - Tickets XLSX: the original columns, plus `problem_id`, `problem_name`, and a confidence score where the method provides one.
  - Problem catalogue XLSX: for each problem, its name, description, ticket count and example tickets.
  - Optional `.txt` logs.
- Granularity must be adjustable through parameters for every method.

## Data contract

Input is an Excel export with the columns `Typ`, `Numer I/SR`, `Zgłoszenie`, `Tytuł`, `Opis`, `Status`, `Grupa odpowiedzialna`. Column names are in Polish and contain diacritics, so keep them exact.

- **Only `Tytuł` and `Opis` may be used for classification.** This is an acceptance criterion. Never feed `Grupa odpowiedzialna`, `Typ`, `Status` or any other column into the embeddings, clustering or LLM prompts.
- `Numer I/SR` is the ticket ID. Every other column passes through to the output unchanged.
- Development uses **synthetic, labelled data**. The requester runs the evaluation on real data, so the code must not depend on anything only the synthetic set has, apart from the ground-truth labels used for evaluation.

## Evaluation

Each method is compared against a hand-labelled sample on:
- clustering quality, covering both failure directions: different problems mixed into one cluster (purity/homogeneity), and one problem split across several clusters (completeness),
- cost (LLM/API usage) and runtime.

The notebook must reproduce this comparison on a given sample.

## Out of scope

Integration with the Service Desk system, live classification, routing tickets to resolver groups, and any UI.

## Open decisions (settle during planning)

Embedding model and LLM (local or API), and any data constraints; clustering algorithms and how to tune them; preprocessing and text cleaning; exact metrics and evaluation sample size; expected data volume, and whether method C is affordable at that volume; CLI interface and config format.

## Agent skills

### Issue tracker

Issues are tracked in GitHub Issues for `mmaku/category-clustering` (via `gh`). See `docs/agents/issue-tracker.md`.

### Triage labels

Default vocabulary: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `GLOSSARY.md` plus `docs/adr/` at the repo root. See `docs/agents/domain.md`.

## Communication

- **Reporting:** be extremely concise. Sacrifice grammar for brevity. No narrative.
- **Asking questions:** use the built-in question UI (`AskUserQuestion`), not plain-text questions in chat.
