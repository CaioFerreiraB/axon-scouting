"""Ordem do retrieval: a da fusão, com empates desfeitos por metadados, sem pesos (spec 103 FR-005, FR-011; R4, R5,
R13).

`(-fusion_score, não vigente, não específico do ramo, não do tipo sugerido, chunk_id)`. Numa busca por faceta (ADR-045),
antes disso vêm a posição do trecho dentro da faceta e a ordem da faceta: o primeiro de cada ramo, depois o segundo.
O retriever grava as marcas nos metadados do `Chunk` — é quem conhece os filtros —, e o reranker ordena a partir delas,
sem depender da ordem de entrada. Chunk sem as marcas (de fora da 103) vai depois, na ordem em que veio.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from axon_ai_contracts import Chunk

from axon_scouting.rag.index.api import DocumentType, IndexedChunk, IndexFilters, ValidityStatus

FUSION_SCORE: Final = "fusion_score"
BRANCH_MATCH: Final = "branch_match"
TYPE_MATCH: Final = "type_match"
FACET: Final = "facet"
FACET_RANK: Final = "facet_rank"
"""Posição do trecho dentro da busca da faceta (0 = o primeiro dela)."""
FACET_ORDER: Final = "facet_order"
"""Posição da faceta na lista pedida."""


def marks(chunk: IndexedChunk, filters: IndexFilters, preferred: Sequence[DocumentType] = ()) -> dict[str, bool]:
    """Específico do ramo pedido (geral não casa) e de um tipo da pergunta: o citado (que já filtrou) ou o sugerido
    pelos temas, que só desempata (ADR-046)."""
    types = (*filters.document_types, *preferred)
    return {
        BRANCH_MATCH: filters.branch is not None and chunk.branch == filters.branch,
        TYPE_MATCH: bool(types) and chunk.document_type in types,
    }


def _annotated(chunk: Chunk) -> bool:
    return isinstance(chunk.metadata.get(FUSION_SCORE), int | float)


def sort_key(chunk: Chunk) -> tuple[int, int, float, bool, bool, bool, str]:
    fusion = chunk.metadata[FUSION_SCORE]
    assert isinstance(fusion, int | float)
    rank, order = chunk.metadata.get(FACET_RANK, 0), chunk.metadata.get(FACET_ORDER, 0)
    return (
        rank if isinstance(rank, int) else 0,
        order if isinstance(order, int) else 0,
        -float(fusion),
        chunk.status != ValidityStatus.VIGENTE.value,
        chunk.metadata.get(BRANCH_MATCH) is not True,
        chunk.metadata.get(TYPE_MATCH) is not True,
        chunk.chunk_id,
    )


def ordered(chunks: Sequence[Chunk]) -> list[Chunk]:
    annotated = sorted((c for c in chunks if _annotated(c)), key=sort_key)
    return annotated + [c for c in chunks if not _annotated(c)]
