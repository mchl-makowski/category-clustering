"""Writes tickets.xlsx: ~20 hand-written Polish Tickets (test asset only, no labels).

Run: uv run python tests/fixtures/make_fixture.py
Non-Tytuł/Opis columns hold distinctive values so the data-contract test can
detect any leak into model input.
"""

from pathlib import Path

import pandas as pd

COLUMNS = ["Typ", "Numer I/SR", "Zgłoszenie", "Tytuł", "Opis", "Status", "Grupa odpowiedzialna"]

# (Tytuł, Opis) pairs. None = empty cell.
ROWS = [
    # VPN drops after FortiClient update (near-duplicates)
    ("VPN rozłącza się po aktualizacji FortiClient", "Po aktualizacji klienta FortiClient do wersji 7.2 połączenie VPN zrywa się co kilka minut."),
    ("Zrywa połączenie VPN", "Od aktualizacji FortiClient 7.2 VPN rozłącza się co kilka minut. Kontakt: jan.kowalski@firma.pl"),
    ("FortiClient 7.2 - VPN się rozłącza", "VPN rozłącza się regularnie po aktualizacji FortiClient do 7.2, praca zdalna niemożliwa."),
    ("VPN rozłącza się co kilka minut", "Po wczorajszej aktualizacji FortiClient połączenie VPN rozłącza się. Tel. +48 601 234 567"),
    # Outlook calendar not syncing
    ("Outlook nie synchronizuje kalendarza", "Kalendarz w Outlook nie synchronizuje się z telefonem, spotkania nie pojawiają się."),
    ("Brak synchronizacji kalendarza Outlook", "Spotkania dodane w Outlook nie synchronizują się z kalendarzem na telefonie."),
    ("Kalendarz Outlook nie synchronizuje spotkań", "Outlook kalendarz przestał synchronizować spotkania z telefonem służbowym."),
    # Printer on 3rd floor
    ("Drukarka HP na 3 piętrze nie drukuje", "Drukarka HP na trzecim piętrze przyjmuje zadania drukowania, ale nic nie drukuje."),
    ("Nie drukuje drukarka HP 3 piętro", "Zadania drukowania wiszą w kolejce, drukarka HP na 3 piętrze nie drukuje."),
    ("Drukarka HP piętro 3 - brak wydruku", "Drukarka HP na 3 piętrze nie drukuje dokumentów, kolejka drukowania stoi."),
    # SAP account locked
    ("Zablokowane konto SAP", "Konto SAP zablokowane po kilku błędnych próbach logowania, proszę o odblokowanie. PESEL 85010112345"),
    ("Proszę odblokować konto w SAP", "Nie mogę zalogować się do SAP, konto zablokowane po błędnym haśle."),
    ("SAP - konto zablokowane", "Konto SAP zostało zablokowane, logowanie niemożliwe. Mail: anna.nowak@firma.pl, tel. 22 123 45 67"),
    # Teams camera
    ("Kamera nie działa w Teams", "Podczas spotkań w Teams kamera nie działa, obraz jest czarny."),
    ("Teams - czarny obraz z kamery", "Kamera w Teams pokazuje czarny obraz na spotkaniach, w innych aplikacjach kamera działa."),
    # One-Ticket Problems
    ("Prośba o nowy monitor", "Proszę o przydzielenie drugiego monitora do stanowiska pracy."),
    ("Błąd makra w Excelu", "Makro w arkuszu Excel kończy się błędem 1004 przy zapisie pliku."),
    # Empty Tickets
    (None, None),
    ("   ", None),
    # Another VPN near-duplicate at the end
    ("VPN FortiClient rozłącza", "FortiClient 7.2 po aktualizacji: VPN rozłącza się co kilka minut, tel. 601-234-568"),
]

TYPES = ["Incydent", "Wniosek o usługę"]
STATUSES = ["Zamknięte", "W realizacji", "Rozwiązane"]
GROUPS = ["GR_SIEC_L2", "GR_APLIKACJE_SAP", "GR_STACJE_ROBOCZE"]


def build() -> pd.DataFrame:
    records = []
    for i, (title, desc) in enumerate(ROWS, start=1):
        records.append(
            {
                "Typ": TYPES[i % 2],
                "Numer I/SR": f"INC{i:06d}",
                "Zgłoszenie": f"ZGL-2026-{i:04d}",
                "Tytuł": title,
                "Opis": desc,
                "Status": STATUSES[i % 3],
                "Grupa odpowiedzialna": GROUPS[i % 3],
            }
        )
    return pd.DataFrame(records, columns=COLUMNS)


if __name__ == "__main__":
    out = Path(__file__).with_name("tickets.xlsx")
    build().to_excel(out, index=False)
    print(f"wrote {out}")
