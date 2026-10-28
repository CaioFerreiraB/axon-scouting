"""Filtros da 101 → `IndexFilters` da 102 (spec 103 FR-002, FR-003, FR-009; research R3; T006)."""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from axon_scouting.rag.index.api import IndexFilters
from axon_scouting.rag.retrieval.domain.filters import InvalidFilters, index_filters

CONTRATO = {  # o exemplo de specs/101-scouting-pack/contracts/retrieval-filters.md
    "branch": "pioneiro",
    "topics": ["lideranca"],
    "document_types": ["manual", "programa_educativo"],
    "effective_from": "2023-01-01",
    "effective_until": "2023-12-31",
    "include_superseded": True,
    "previous_version": False,
}
DIA = {"effective_from": "2026-10-07", "effective_until": "2026-10-07"}


def test_exemplo_do_contrato() -> None:
    assert index_filters(CONTRATO) == IndexFilters(
        effective_from=date(2023, 1, 1),
        effective_until=date(2023, 12, 31),
        include_superseded=True,
        previous_version=False,
        branch="pioneiro",
        document_types=("manual", "programa_educativo"),
    )


def test_so_as_datas() -> None:
    assert index_filters(DIA) == IndexFilters(effective_from=date(2026, 10, 7), effective_until=date(2026, 10, 7))


def test_temas_nao_mudam_nada() -> None:
    assert index_filters({**DIA, "topics": ["uniforme", "acampamento"]}) == index_filters(DIA)


@pytest.mark.parametrize(
    ("raw", "chave"),
    [
        (None, None),
        ({"effective_until": "2026-10-07"}, "effective_from"),
        ({"effective_from": "2026-10-07"}, "effective_until"),
        ({**DIA, "effective_from": "2026-13-01"}, "effective_from"),
        ({**DIA, "effective_from": "2026-10-08"}, "effective_from"),  # início depois do fim
        ({**DIA, "include_superseded": "sim"}, "include_superseded"),
        ({**DIA, "previous_version": 1}, "previous_version"),
        ({**DIA, "document_types": "manual"}, "document_types"),
        ({**DIA, "document_types": ["Resolução"]}, "document_types"),  # rótulo, não id da taxonomia
        ({**DIA, "branch": "pioneiros"}, "branch"),
        ({**DIA, "branch": 3}, "branch"),
        ({**DIA, "tema": "uniforme"}, "tema"),
    ],
)
def test_invalidos(raw: Any, chave: str | None) -> None:
    with pytest.raises(InvalidFilters) as erro:
        index_filters(raw)
    assert erro.value.key == chave
    if chave:
        assert chave in str(erro.value)
