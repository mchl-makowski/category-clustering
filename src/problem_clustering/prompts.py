"""Prompt text shared by every LLM step. All prompts are Polish.

Every LLM step (naming, B verification/merge, C agent/consolidation, Judge)
must start its system message with `problem_definition(level)` so one
Granularity Level controls how narrow a Problem is. Later tickets add their
own prompt builders below.
"""

from .config import GranularityLevel

_BASE = (
    "Problem to zbiór zgłoszeń Service Desk opisujących ten sam obserwowalny objaw "
    "w tym samym kontekście (system, komponent, wersja). Przyczyna źródłowa nie jest kryterium. "
    "Problem może zawierać jedno zgłoszenie."
)

GRANULARITY_PARAGRAPHS: dict[str, str] = {
    "coarse": "Poziom szczegółowości: OGÓLNY. Łącz zgłoszenia o tym samym objawie w tym samym systemie, "
    "nawet jeśli różnią się wersją, lokalizacją lub drobnymi szczegółami.",
    "medium": "Poziom szczegółowości: ŚREDNI. Łącz zgłoszenia o tym samym objawie w tym samym systemie "
    "i komponencie; rozdzielaj, gdy różni się komponent lub wyraźnie inny kontekst.",
    "fine": "Poziom szczegółowości: SZCZEGÓŁOWY. Łącz tylko zgłoszenia o identycznym objawie w identycznym "
    "kontekście (system, komponent, wersja, lokalizacja); każda istotna różnica oznacza osobny Problem.",
}


def problem_definition(level: GranularityLevel) -> str:
    return f"{_BASE}\n{GRANULARITY_PARAGRAPHS[level]}"


def ticket_list(texts: list[str]) -> str:
    """Tickets as '- ' bullet lines, newlines flattened."""
    return "\n".join("- " + " ".join(t.split()) for t in texts)


# --- naming -----------------------------------------------------------------

NAMING_SCHEMA_NAME = "problem_naming"
NAMING_SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "description": {"type": "string"}},
    "required": ["name", "description"],
    "additionalProperties": False,
}


def naming_messages(level: GranularityLevel, texts: list[str]) -> list[dict]:
    system = (
        f"{problem_definition(level)}\n\n"
        "Jesteś analitykiem Service Desk. Otrzymasz zgłoszenia należące do jednego Problemu. "
        "Nadaj Problemowi krótką, konkretną nazwę po polsku (do 10 słów, objaw + kontekst) "
        "oraz opis po polsku (1-2 zdania). Odpowiedz w JSON: {\"name\": ..., \"description\": ...}."
    )
    user = f"Zgłoszenia:\n{ticket_list(texts)}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
