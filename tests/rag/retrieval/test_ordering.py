"""Marcas e chave de ordem: fusão, com desempate por metadados (spec 103 FR-005; research R4, R5; T007)."""

from __future__ import annotations

import dataclasses
from datetime import date

from axon_ai_contracts import Chunk
from axon_ai_contracts.testing import chunk as chunk_cru

from axon_scouting.rag.index.api import DocumentType, IndexedChunk, IndexFilters, ValidityStatus
from axon_scouting.rag.retrieval.domain.ordering import BRANCH_MATCH, FUSION_SCORE, TYPE_MATCH, marks, ordered

DIA = date(2026, 10, 7)
BASE = IndexedChunk(
    chunk_id="manual-x-2026#s1@p1",
    text="texto",
    doc_key="manual-x-2026",
    document_id="manual-x",
    version="2026",
    document_type=DocumentType.MANUAL,
    document_title="Manual X",
    jurisdiction="nacional",
    page_start=1,
    page_end=1,
    effective_date=date(2026, 3, 1),
    status=ValidityStatus.VIGENTE,
)


def test_marcas() -> None:
    pioneiro = dataclasses.replace(BASE, branch="pioneiro")
    geral = BASE
    pedido = IndexFilters(
        effective_from=DIA, effective_until=DIA, branch="pioneiro", document_types=(DocumentType.MANUAL,)
    )
    sem_nada = IndexFilters(effective_from=DIA, effective_until=DIA)
    assert marks(pioneiro, pedido) == {BRANCH_MATCH: True, TYPE_MATCH: True}
    assert marks(geral, pedido) == {BRANCH_MATCH: False, TYPE_MATCH: True}  # geral não casa com o ramo
    assert marks(pioneiro, sem_nada) == {BRANCH_MATCH: False, TYPE_MATCH: False}
    outro_tipo = IndexFilters(effective_from=DIA, effective_until=DIA, document_types=(DocumentType.POR,))
    assert marks(pioneiro, outro_tipo)[TYPE_MATCH] is False


def anotado(chunk_id: str, fusao: float, *, status: str = "vigente", ramo: bool = False, tipo: bool = False) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text="t",
        score=0.5,
        document="Doc",
        status=status,
        metadata={FUSION_SCORE: fusao, BRANCH_MATCH: ramo, TYPE_MATCH: tipo},
    )


def test_chave_de_ordem() -> None:
    entrada = [
        anotado("e", 0.5),
        anotado("d", 0.5, ramo=True),
        anotado("c", 0.5, status="substituido", ramo=True, tipo=True),
        anotado("b", 0.5, ramo=True, tipo=True),
        anotado("a", 1.0, status="substituido"),
        anotado("f", 0.5),
    ]
    assert [c.chunk_id for c in ordered(entrada)] == ["a", "b", "d", "e", "f", "c"]


def test_crus_depois_dos_anotados_na_ordem_de_entrada() -> None:
    crus = [chunk_cru("z"), chunk_cru("y")]
    entrada = [crus[0], anotado("b", 0.3), crus[1], anotado("a", 0.6)]
    assert [c.chunk_id for c in ordered(entrada)] == ["a", "b", "z", "y"]
