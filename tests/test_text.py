"""Normalização e casamento de termos (spec 101, research R6; T004)."""

from __future__ import annotations

import pytest

from axon_scouting.context.text import contains_term, normalize


@pytest.mark.parametrize(
    ("texto", "esperado"),
    [
        ("Liderança", "lideranca"),
        ("Clã", "cla"),
        ("  Ramo   SÊNIOR!  ", "ramo senior"),
        ("Qual é a regra?\nEm 15/03/2023, talvez.", "qual e a regra em 15 03 2023 talvez"),
        ("", ""),
    ],
)
def test_normalize(texto: str, esperado: str) -> None:
    assert normalize(texto) == esperado


@pytest.mark.parametrize(
    ("texto", "termo", "casa"),
    [
        ("o pioneiro e", "pioneiro", True),
        ("pioneirismo", "pioneiro", False),
        ("no ramo senior", "ramo senior", True),
        ("ramo seniores", "ramo senior", False),
        ("senior", "ramo senior", False),
        ("manual anterior do ramo", "manual anterior", True),
    ],
)
def test_contains_term(texto: str, termo: str, casa: bool) -> None:
    assert contains_term(texto, termo) is casa
