"""Vigência: dia de referência em Brasília e intervalos (spec 101 FR-005; research R5, R8; T015 e T021)."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from ai_scouting.context.periods import Validity, find_period, reference_day
from ai_scouting.context.text import normalize

# --- US2: o dia de referência e a vigência default ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("instante", "dia"),
    [
        (datetime(2026, 10, 7, 12, tzinfo=UTC), date(2026, 10, 7)),
        (datetime(2026, 10, 8, 1, tzinfo=UTC), date(2026, 10, 7)),  # 22h em Brasília: ainda é dia 7
        (datetime(2026, 10, 8, 3, tzinfo=UTC), date(2026, 10, 8)),
    ],
)
def test_dia_de_referencia_em_brasilia(instante: datetime, dia: date) -> None:
    assert reference_day(instante) == dia


def test_vigencia_de_um_dia() -> None:
    dia = date(2026, 10, 7)
    assert Validity.day(dia) == Validity(start=dia, end=dia)


def test_vigencia_invertida_e_recusada() -> None:
    with pytest.raises(ValueError, match="vigência"):
        Validity(start=date(2026, 10, 8), end=date(2026, 10, 7))


# --- US3: períodos citados viram intervalo (clarificação 4) ------------------------------------------------------

REFERENCIA = date(2026, 10, 7)


@pytest.mark.parametrize(
    ("pergunta", "inicio", "fim"),
    [
        ("Qual era a regra em 2023?", date(2023, 1, 1), date(2023, 12, 31)),
        ("O que valia no ano de 2023?", date(2023, 1, 1), date(2023, 12, 31)),
        ("O que mudou durante 2023?", date(2023, 1, 1), date(2023, 12, 31)),
        ("Como era em março de 2023?", date(2023, 3, 1), date(2023, 3, 31)),
        ("E em fevereiro de 2024?", date(2024, 2, 1), date(2024, 2, 29)),
        ("O que valia em 15/03/2023?", date(2023, 3, 15), date(2023, 3, 15)),
        ("O que valia em 15 de março de 2023?", date(2023, 3, 15), date(2023, 3, 15)),
        ("Qual era a regra no ano passado?", date(2025, 1, 1), date(2025, 12, 31)),
        ("Como era em 2022 e em 2023?", date(2022, 1, 1), date(2022, 12, 31)),  # o primeiro vence
    ],
)
def test_historico_periodo(pergunta: str, inicio: date, fim: date) -> None:
    assert find_period(normalize(pergunta), REFERENCIA) == Validity(start=inicio, end=fim)


@pytest.mark.parametrize(
    "pergunta",
    [
        "O que diz a resolução 3/2023?",  # ano solto é número de documento, não período
        "Qual é a regra?",
        "O que valia em 31/02/2023?",  # data impossível
        "",
    ],
)
def test_historico_sem_periodo(pergunta: str) -> None:
    assert find_period(normalize(pergunta), REFERENCIA) is None
