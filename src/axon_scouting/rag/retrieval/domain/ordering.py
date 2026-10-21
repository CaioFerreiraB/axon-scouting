"""Ordem do retrieval: a da fusão, com empates desfeitos por metadados, sem pesos (spec 103 FR-005; R4, R5).

`(-fusion_score, não vigente, não específico do ramo, não do tipo sugerido, chunk_id)`. O retriever grava as marcas nos
metadados do `Chunk` — é quem conhece os filtros —, e o reranker ordena a partir delas, sem depender da ordem de
entrada. Chunk sem as marcas (de fora da 103) vai depois, na ordem em que veio.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from axon_ai_contracts import Chunk

from axon_scouting.rag.index.api import IndexedChunk, IndexFilters, ValidityStatus

FUSION_SCORE: Final = "fusion_score"
BRANCH_MATCH: Final = "branch_match"
TYPE_MATCH: Final = "type_match"


def marks(chunk: IndexedChunk, filters: IndexFilters) -> dict[str, bool]:
    """Específico do ramo pedido (geral não casa) e do tipo sugerido pela pergunta."""
    return {
        BRANCH_MATCH: filters.branch is not None and chunk.branch == filters.branch,
        TYPE_MATCH: bool(filters.document_types) and chunk.document_type in filters.document_types,
    }


def _annotated(chunk: Chunk) -> bool:
    return isinstance(chunk.metadata.get(FUSION_SCORE), int | float)


def sort_key(chunk: Chunk) -> tuple[float, bool, bool, bool, str]:
    fusion = chunk.metadata[FUSION_SCORE]
    assert isinstance(fusion, int | float)
    return (
        -float(fusion),
        chunk.status != ValidityStatus.VIGENTE.value,
        chunk.metadata.get(BRANCH_MATCH) is not True,
        chunk.metadata.get(TYPE_MATCH) is not True,
        chunk.chunk_id,
    )


def ordered(chunks: Sequence[Chunk]) -> list[Chunk]:
    annotated = sorted((c for c in chunks if _annotated(c)), key=sort_key)
    return annotated + [c for c in chunks if not _annotated(c)]
