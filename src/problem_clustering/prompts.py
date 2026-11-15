"""All LLM prompt text and JSON reply schemas, one section per LLM step. All prompts are Polish.

Every LLM step (Symptom Summary, naming, B verification/merge, C agent/consolidation, Judge)
must start its system message with `problem_definition(level)` so one
Granularity Level controls how narrow a Problem is. A new LLM step adds its
section here, not in its own module.
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


# --- Symptom Summary ----------------------------------------------------------

SUMMARY_SCHEMA_NAME = "symptom_summary"
SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {"summary": {"type": "string"}},
    "required": ["summary"],
    "additionalProperties": False,
}
TICKET_HEADER = "Zgłoszenie:\n"


def summary_messages(level: GranularityLevel, text: str) -> list[dict]:
    system = (
        f"{problem_definition(level)}\n\n"
        "Jesteś analitykiem Service Desk. Przepisz zgłoszenie na jedną znormalizowaną linię po polsku: "
        "obserwowalny objaw + kontekst (system, komponent, wersja, lokalizacja), bez danych osobowych, "
        "powitań i zbędnych szczegółów. Szczegółowość kontekstu dopasuj do poziomu szczegółowości powyżej. "
        "Odpowiedz w JSON: {\"summary\": ...}."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": TICKET_HEADER + text}]


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


# --- Method B: cluster verification / merge check ---------------------------

VERIFICATION_SCHEMA_NAME = "cluster_verification"
VERIFICATION_SCHEMA = {
    "type": "object",
    "properties": {"one_problem": {"type": "boolean"}, "reason": {"type": "string"}},
    "required": ["one_problem", "reason"],
    "additionalProperties": False,
}


def verification_messages(level: GranularityLevel, texts: list[str]) -> list[dict]:
    system = (
        f"{problem_definition(level)}\n\n"
        "Jesteś analitykiem Service Desk. Otrzymasz próbkę zgłoszeń z jednej grupy. "
        "Oceń, czy wszystkie opisują jeden i ten sam Problem zgodnie z powyższą definicją. "
        "Odpowiedz w JSON: {\"one_problem\": true|false, \"reason\": krótkie uzasadnienie po polsku}."
    )
    user = f"Zgłoszenia:\n{ticket_list(texts)}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# "Are these two samples the same Problem?" reply; shared by the B merge check and the Judge Split check.
SAME_PROBLEM_SCHEMA = {
    "type": "object",
    "properties": {"same_problem": {"type": "boolean"}, "reason": {"type": "string"}},
    "required": ["same_problem", "reason"],
    "additionalProperties": False,
}

MERGE_SCHEMA_NAME = "problem_merge"
MERGE_SCHEMA = SAME_PROBLEM_SCHEMA

MERGE_GROUP_A = "Grupa A:"
MERGE_GROUP_B = "Grupa B:"


def merge_messages(level: GranularityLevel, texts_a: list[str], texts_b: list[str]) -> list[dict]:
    system = (
        f"{problem_definition(level)}\n\n"
        "Jesteś analitykiem Service Desk. Otrzymasz próbki zgłoszeń z dwóch grup (A i B). "
        "Oceń, czy obie grupy opisują ten sam Problem zgodnie z powyższą definicją i powinny zostać połączone. "
        "Odpowiedz w JSON: {\"same_problem\": true|false, \"reason\": krótkie uzasadnienie po polsku}."
    )
    user = f"{MERGE_GROUP_A}\n{ticket_list(texts_a)}\n\n{MERGE_GROUP_B}\n{ticket_list(texts_b)}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# --- Method C: Ticket-by-Ticket agent ----------------------------------------

CONFIDENCE_LEVEL_HELP = "Poziom pewności (confidence_level): pewne, wysokie, średnie, niskie, zgadywane."


def _function(name: str, description: str, properties: dict, required: list[str]) -> dict:
    """One tool in OpenAI/Azure function-calling shape."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


def agent_tools(confidence_levels: list[str], max_k: int) -> list[dict]:
    level = {"type": "string", "enum": confidence_levels, "description": CONFIDENCE_LEVEL_HELP}
    return [
        _function(
            "search_similar_tickets",
            "Wyszukaj najbardziej podobne, już sklasyfikowane zgłoszenia (wyszukiwanie semantyczne). "
            "Zwraca tekst zgłoszenia, problem_id, nazwę Problemu i podobieństwo (0-1).",
            {
                "query": {"type": "string", "description": "Tekst do wyszukania, np. objaw i kontekst."},
                "k": {"type": "integer", "description": f"Liczba wyników (1-{max_k})."},
            },
            ["query", "k"],
        ),
        _function(
            "get_problem",
            "Pobierz szczegóły Problemu: nazwę, opis, liczbę zgłoszeń i przykładowe zgłoszenia.",
            {"problem_id": {"type": "string", "description": "Identyfikator Problemu, np. C-0001."}},
            ["problem_id"],
        ),
        _function(
            "assign_to_problem",
            "DECYZJA KOŃCOWA: przypisz bieżące zgłoszenie do istniejącego Problemu.",
            {"problem_id": {"type": "string", "description": "Identyfikator istniejącego Problemu."},
             "confidence_level": level},
            ["problem_id", "confidence_level"],
        ),
        _function(
            "create_problem",
            "DECYZJA KOŃCOWA: utwórz nowy Problem dla bieżącego zgłoszenia.",
            {"name": {"type": "string", "description": "Krótka nazwa po polsku (do 10 słów, objaw + kontekst)."},
             "description": {"type": "string", "description": "Opis Problemu po polsku (1-2 zdania)."},
             "confidence_level": level},
            ["name", "description", "confidence_level"],
        ),
    ]


def agent_messages(level: GranularityLevel, ticket_text: str, problem_count: int, max_tool_calls: int) -> list[dict]:
    system = (
        f"{problem_definition(level)}\n\n"
        "Jesteś analitykiem Service Desk budującym katalog Problemów. Zgłoszenia przetwarzasz pojedynczo. "
        "Dla bieżącego zgłoszenia znajdź podobne, już sklasyfikowane zgłoszenia (search_similar_tickets), "
        "w razie potrzeby sprawdź Problem (get_problem), a następnie podejmij dokładnie jedną decyzję końcową: "
        "assign_to_problem, jeśli zgłoszenie opisuje ten sam Problem, albo create_problem, jeśli to nowy Problem. "
        f"Masz najwyżej {max_tool_calls} wywołań narzędzi; bez decyzji zgłoszenie dostanie nowy Problem. "
        f"{CONFIDENCE_LEVEL_HELP} "
        f"Liczba Problemów w katalogu: {problem_count}."
    )
    user = f"Bieżące zgłoszenie:\n{ticket_text}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


AGENT_NUDGE = "Wywołaj narzędzie. Zakończ decyzją: assign_to_problem albo create_problem."


# --- Method C: consolidation -------------------------------------------------

CONSOLIDATION_SCHEMA_NAME = "method_c_consolidation"
CONSOLIDATION_SCHEMA = {
    "type": "object",
    "properties": {
        "merges": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "problem_ids": {"type": "array", "items": {"type": "string"}},
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["problem_ids", "name", "description"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["merges"],
    "additionalProperties": False,
}


def consolidation_messages(level: GranularityLevel, problems: list[dict]) -> list[dict]:
    """`problems`: dicts with problem_id, name, description, ticket_count, examples (list of texts)."""
    system = (
        f"{problem_definition(level)}\n\n"
        "Jesteś analitykiem Service Desk porządkującym katalog Problemów. Otrzymasz listę podobnych Problemów. "
        "Wskaż grupy Problemów, które są duplikatami (ten sam Problem według definicji powyżej) i należy je "
        "połączyć. Dla każdej grupy podaj problem_ids (co najmniej 2), nową nazwę i opis po polsku. "
        "Nie łącz Problemów, które się różnią. Jeśli nie ma duplikatów, zwróć pustą listę. "
        "Odpowiedz w JSON: {\"merges\": [{\"problem_ids\": [...], \"name\": ..., \"description\": ...}]}."
    )
    blocks = [
        f"[{p['problem_id']}] {p['name']} ({p['ticket_count']} zgł.)\n"
        f"Opis: {p['description']}\nPrzykłady:\n{ticket_list(p['examples'])}"
        for p in problems
    ]
    user = "Problemy:\n\n" + "\n\n".join(blocks)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# --- Judge: Mixed / Split checks ----------------------------------------------

MIXED_SCHEMA_NAME = "judge_mixed"
MIXED_SCHEMA = {
    "type": "object",
    "properties": {"mixed": {"type": "boolean"}, "reason": {"type": "string"}},
    "required": ["mixed", "reason"],
    "additionalProperties": False,
}
SPLIT_SCHEMA_NAME = "judge_split"
SPLIT_SCHEMA = SAME_PROBLEM_SCHEMA


def mixed_messages(level: GranularityLevel, texts: list[str]) -> list[dict]:
    system = (
        f"{problem_definition(level)}\n\n"
        "Jesteś sędzią oceniającym grupowanie zgłoszeń Service Desk. Otrzymasz próbkę zgłoszeń "
        "przypisanych do jednego Problemu. Oceń, czy wszystkie opisują ten sam Problem, czy grupa "
        "miesza różne Problemy. Odpowiedz w JSON: {\"mixed\": true|false, \"reason\": \"krótkie uzasadnienie po polsku\"}."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": f"Zgłoszenia:\n{ticket_list(texts)}"}]


def split_messages(level: GranularityLevel, texts_a: list[str], texts_b: list[str]) -> list[dict]:
    system = (
        f"{problem_definition(level)}\n\n"
        "Jesteś sędzią oceniającym grupowanie zgłoszeń Service Desk. Otrzymasz próbki zgłoszeń z dwóch "
        "Problemów. Oceń, czy to w rzeczywistości ten sam Problem rozbity na dwie grupy. Odpowiedz w JSON: "
        "{\"same_problem\": true|false, \"reason\": \"krótkie uzasadnienie po polsku\"}."
    )
    user = f"Problem 1:\n{ticket_list(texts_a)}\n\nProblem 2:\n{ticket_list(texts_b)}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
