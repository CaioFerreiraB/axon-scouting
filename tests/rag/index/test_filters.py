"""Semântica de `IndexFilters`, uma linha por regra (spec 102 research R5; T007)."""

from __future__ import annotations

import dataclasses
from datetime import date

import pytest

from axon_scouting.rag.index.api import DocumentType, IndexedChunk, IndexFilters, ValidityStatus
from axon_scouting.rag.index.domain.filters import matches

HOJE = date(2026, 10, 7)

M2023 = IndexedChunk(
    chunk_id="manual-x-2023#s1@p1",
    text="texto",
    doc_key="manual-x-2023",
    document_id="manual-x",
    version="2023",
    document_type=DocumentType.MANUAL,
    document_title="Manual X",
    jurisdiction="nacional",
    page_start=1,
    page_end=1,
    effective_date=date(2023, 1, 1),
    expiry_date=date(2026, 2, 28),
    superseded_by="manual-x-2026",
    status=ValidityStatus.SUBSTITUIDO,
)
M2026 = dataclasses.replace(
    M2023,
    chunk_id="manual-x-2026#s1@p1",
    doc_key="manual-x-2026",
    version="2026",
    effective_date=date(2026, 3, 1),
    expiry_date=None,
    superseded_by=None,
    status=ValidityStatus.VIGENTE,
)
M2019 = dataclasses.replace(
    M2023,
    chunk_id="manual-x-2019#s1@p1",
    doc_key="manual-x-2019",
    version="2019",
    effective_date=date(2019, 1, 1),
    expiry_date=date(2022, 12, 31),
    superseded_by="manual-x-2023",
    successor_expiry_date=date(2026, 2, 28),
)
RASCUNHO = dataclasses.replace(M2026, chunk_id="manual-x-2027#s1@p1", version="2027", status=ValidityStatus.RASCUNHO)
REVOGADA = dataclasses.replace(
    M2026,
    chunk_id="res-1-1#art1@p1",
    document_type=DocumentType.RESOLUCAO,
    effective_date=date(2024, 5, 10),
    expiry_date=date(2025, 12, 31),
    status=ValidityStatus.REVOGADO,
)
PIONEIRO = dataclasses.replace(M2026, chunk_id="pio-1#s1@p1", branch="pioneiro")
ESCOTEIRO = dataclasses.replace(M2026, chunk_id="esc-1#s1@p1", branch="escoteiro")
POR = dataclasses.replace(M2026, chunk_id="por-1#regra1@p1", document_type=DocumentType.POR)


def dia(d: date, **kw: object) -> IndexFilters:
    return IndexFilters(effective_from=d, effective_until=d, **kw)  # type: ignore[arg-type]


def periodo(inicio: date, fim: date, **kw: object) -> IndexFilters:
    return IndexFilters(effective_from=inicio, effective_until=fim, **kw)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("filtros", "chunk", "esperado"),
    [
        # em vigor no dia (o default da 101): só a versão em vigor
        (dia(HOJE), M2026, True),
        (dia(HOJE), M2023, False),
        (dia(date(2024, 6, 1)), M2023, True),  # data de referência passada: a versão em vigor então
        (dia(date(2024, 6, 1)), M2026, False),  # ainda não começou
        (dia(date(2026, 2, 28)), M2023, True),  # fim de vigência é inclusivo
        (dia(date(2026, 3, 1)), M2023, False),
        # intervalo: em vigor em algum dia; sem include_superseded, ainda em vigor no fim
        (periodo(date(2023, 1, 1), date(2023, 12, 31), include_superseded=True), M2023, True),
        (periodo(date(2023, 1, 1), date(2023, 12, 31)), M2023, True),  # ainda em vigor em 2023-12-31
        (periodo(date(2026, 1, 1), date(2026, 12, 31)), M2023, False),  # substituída dentro do intervalo
        (periodo(date(2026, 1, 1), date(2026, 12, 31), include_superseded=True), M2023, True),
        (periodo(date(2026, 1, 1), date(2026, 12, 31)), M2026, True),
        (periodo(date(2020, 1, 1), date(2020, 12, 31), include_superseded=True), M2023, False),  # nunca em vigor
        (periodo(date(2025, 1, 1), date(2025, 12, 31), include_superseded=True), REVOGADA, True),
        (dia(HOJE, include_superseded=True), REVOGADA, False),
        # versão imediatamente substituída em relação ao fim do intervalo
        (dia(HOJE, previous_version=True), M2023, True),
        (dia(HOJE, previous_version=True), M2026, False),
        (dia(HOJE, previous_version=True), M2019, False),  # o sucessor (2023) não está em vigor hoje
        (dia(date(2024, 6, 1), previous_version=True), M2019, True),
        (dia(date(2024, 6, 1), previous_version=True), M2023, False),
        # rascunho nunca
        (dia(HOJE), RASCUNHO, False),
        (dia(HOJE, include_superseded=True), RASCUNHO, False),
        (periodo(date(2000, 1, 1), date(2100, 1, 1), include_superseded=True), RASCUNHO, False),
        # ramo: o ramo ou geral
        (dia(HOJE, branch="pioneiro"), PIONEIRO, True),
        (dia(HOJE, branch="pioneiro"), M2026, True),
        (dia(HOJE, branch="pioneiro"), ESCOTEIRO, False),
        (dia(HOJE), ESCOTEIRO, True),
        # tipos: vazio = todos
        (dia(HOJE, document_types=(DocumentType.POR,)), POR, True),
        (dia(HOJE, document_types=(DocumentType.POR,)), M2026, False),
        (dia(HOJE, document_types=(DocumentType.POR, DocumentType.MANUAL)), M2026, True),
    ],
)
def test_semantica(filtros: IndexFilters, chunk: IndexedChunk, esperado: bool) -> None:
    assert matches(filtros, chunk) is esperado
