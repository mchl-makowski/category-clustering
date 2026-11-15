# Service Desk Problem Clustering

Groups Service Desk tickets that describe the same specific problem and gives each group a shared label, so recurring issues become visible.

## Language

**Ticket**:
One row of the Service Desk export, identified by `Numer I/SR`. Only its title (`Tytuł`) and description (`Opis`) carry meaning for classification.
_Avoid_: Incident, request, record, row

**Problem**:
A set of Tickets reporting the same observable symptom in the same context (system, component, version). Root cause is not the criterion, because it is usually unknown when the Ticket is raised. A Problem may hold a single Ticket.
_Avoid_: Category, cluster, issue, topic, noise, unassigned

**Problem Catalogue**:
The list of Problems a run produced, each with a Polish name, a description, its Ticket count and example Tickets.
_Avoid_: Taxonomy, label list

**Symptom Summary**:
An optional LLM rewrite of a Ticket into one normalised line of symptom plus context, used in place of the raw text for grouping.
_Avoid_: Normalised ticket, abstract

**Method**:
One of the three competing ways of turning Tickets into Problems: A (embeddings + clustering), B (clustering + LLM verification), C (Ticket-by-Ticket agent with lookup).
_Avoid_: Approach, algorithm, pipeline

**Granularity Level**:
How narrowly a run defines a Problem: coarse, medium or fine. One level applies to every LLM step of a run, including the Judge.
_Avoid_: Resolution, detail level

**Confidence Level**:
One of five predefined grades (pewne, wysokie, średnie, niskie, zgadywane) an LLM gives for a Ticket's assignment to a Problem.
_Avoid_: Certainty, probability

## Evaluation

**Mixed Problem**:
A Problem whose Tickets report different symptoms or contexts. Failure of purity.
_Avoid_: Impure cluster, over-merge

**Split Problem**:
One real-world Problem spread across several Problems in a run. Failure of completeness.
_Avoid_: Fragmentation, over-split, duplicate

**Judge**:
A fixed LLM, shared by all Methods, that checks a run's Problems for Mixed and Split Problems in place of ground-truth labels.
_Avoid_: Evaluator, grader
