"""Vigência da consulta: o dia de referência e os intervalos de data (spec 101 FR-005; research R5, R8).

Puro: o instante vem da Turn View (`reference_time`, ADR-013), nunca do relógio do sistema.
"""

from __future__ import annotations

import calendar
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Final

BRASILIA: Final = timezone(timedelta(hours=-3), "UTC-03:00")
"""Horário de Brasília. Sem horário de verão desde 2019, o deslocamento fixo é exato e dispensa a base de fusos
(`tzdata` seria dependência além dos contratos — FR-001)."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Validity:
    """Intervalo de vigência, inclusivo nas duas pontas (clarificação 4)."""

    start: date
    end: date

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ValueError(f"vigência invertida: {self.start} depois de {self.end}")

    @classmethod
    def day(cls, value: date) -> Validity:
        return cls(start=value, end=value)


def reference_day(reference_time: datetime) -> date:
    """O dia da requisição em Brasília: o core entrega `reference_time` em UTC, e a vigência de documento brasileiro é
    por dia local — às 22h em Brasília, o dia UTC já é o seguinte."""
    return reference_time.astimezone(BRASILIA).date()


MONTHS: Final = (
    "janeiro",
    "fevereiro",
    "marco",
    "abril",
    "maio",
    "junho",
    "julho",
    "agosto",
    "setembro",
    "outubro",
    "novembro",
    "dezembro",
)
"""Nomes dos meses já normalizados (sem acento: "março" → "marco")."""

_MONTH: Final = "|".join(MONTHS)
_YEAR: Final = r"(19\d{2}|20\d{2})"

_EXACT_NUMERIC: Final = re.compile(rf"\b(?:em|no dia) (\d{{1,2}}) (\d{{1,2}}) {_YEAR}\b")
"""`em 15/03/2023` — a barra vira espaço na normalização."""
_EXACT_WORDS: Final = re.compile(rf"\b(?:em|no dia) (\d{{1,2}}) de ({_MONTH}) de {_YEAR}\b")
_MONTH_YEAR: Final = re.compile(rf"\b(?:em|no mes de|durante) ({_MONTH}) de {_YEAR}\b")
_YEAR_MARKED: Final = re.compile(rf"\b(?:em|no ano de|durante) {_YEAR}\b")
"""Ano só conta como período depois de uma marca temporal: "resolução 3/2023" não é consulta histórica."""
_LAST_YEAR: Final = re.compile(r"\bano passado\b")


def find_period(normalized: str, reference_day: date) -> Validity | None:
    """O primeiro período citado na pergunta (texto já normalizado), ou None.

    Data exata → um dia; mês com ano → o mês; ano com marca → o ano; "ano passado" → o ano anterior ao dia de
    referência. Data impossível não é período. Com dois períodos, vence o primeiro na ordem do texto (research R8).
    """
    found = [candidate for candidate in _candidates(normalized, reference_day) if candidate[1] is not None]
    if not found:
        return None
    return min(found, key=lambda candidate: candidate[0])[1]


def _candidates(text: str, reference_day: date) -> Iterator[tuple[int, Validity | None]]:
    readers: tuple[tuple[re.Pattern[str], Callable[[re.Match[str]], Validity | None]], ...] = (
        (_EXACT_NUMERIC, lambda m: _exact(int(m[3]), int(m[2]), int(m[1]))),
        (_EXACT_WORDS, lambda m: _exact(int(m[3]), MONTHS.index(m[2]) + 1, int(m[1]))),
        (_MONTH_YEAR, lambda m: _month(int(m[2]), MONTHS.index(m[1]) + 1)),
        (_YEAR_MARKED, lambda m: _year(int(m[1]))),
        (_LAST_YEAR, lambda _m: _year(reference_day.year - 1)),
    )
    for pattern, read in readers:
        for match in pattern.finditer(text):
            yield match.start(), read(match)


def _exact(year: int, month: int, day: int) -> Validity | None:
    try:
        return Validity.day(date(year, month, day))
    except ValueError:
        return None


def _month(year: int, month: int) -> Validity:
    last = calendar.monthrange(year, month)[1]
    return Validity(start=date(year, month, 1), end=date(year, month, last))


def _year(year: int) -> Validity:
    return Validity(start=date(year, 1, 1), end=date(year, 12, 31))
