"""Ordem do retrieval: a da fusão, com empates desfeitos por metadados, sem pesos (spec 103 FR-005, FR-011; R4, R5,
R13).

`(-fusion_score, não vigente, não específico do ramo, não do tipo sugerido, não do órgão nomeado, chunk_id)` — o órgão
desde o ADR-049. Numa busca por faceta (ADR-045), antes disso vêm a posição do trecho dentro da faceta e a ordem da
faceta: o primeiro de cada ramo, depois o segundo. As fichas da lista de documentos (ADR-047) vêm antes de tudo, na
ordem da lista; depois, o que a tool de unidades trouxe (ADR-048: as listas de unidades de um tipo e os trechos com
forma de lista), na ordem dela.
O retriever grava as marcas nos metadados do `Chunk` — é quem conhece os filtros —, e o reranker ordena a partir delas,
sem depender da ordem de entrada. Chunk sem as marcas (de fora da 103) vai depois, na ordem em que veio.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Final

from axon_ai_contracts import Chunk

from axon_scouting.rag.index.api import (
    UNIT_GROUP,
    CatalogEntry,
    IndexedChunk,
    IndexFilters,
    ValidityStatus,
)

FUSION_SCORE: Final = "fusion_score"
BRANCH_MATCH: Final = "branch_match"
TYPE_MATCH: Final = "type_match"
BODY_MATCH: Final = "body_match"
FACET: Final = "facet"
FACET_RANK: Final = "facet_rank"
"""Posição do trecho dentro da busca da faceta (0 = o primeiro dela)."""
FACET_ORDER: Final = "facet_order"
"""Posição da faceta na lista pedida."""
CATALOG_RANK: Final = "catalog_rank"
"""Posição da ficha na lista de documentos; só as fichas a têm."""
STRUCTURE_RANK: Final = "structure_rank"
"""Posição no que a tool de unidades devolveu (ADR-048); só o que ela trouxe a tem."""
ALSO_IN: Final = "also_in"
"""Os rótulos dos trechos de texto idêntico que a busca juntou neste (103 FR-016)."""


def marks(
    chunk: IndexedChunk | CatalogEntry,
    filters: IndexFilters,
    preferred: Sequence[str] = (),
    bodies: Sequence[str] = (),
) -> dict[str, bool]:
    """Específico do ramo pedido (geral não casa), de um tipo da pergunta — o citado (que já filtrou) ou o sugerido
    pelos temas, que só desempata (ADR-046) — e de um órgão que ela nomeia (ADR-049)."""
    types = (*filters.document_types, *preferred)
    wanted_bodies = {*filters.bodies, *bodies}
    branches = (
        chunk.effective_branches if isinstance(chunk, IndexedChunk) else ((chunk.branch,) if chunk.branch else ())
    )
    return {
        BRANCH_MATCH: filters.branch is not None and filters.branch in branches,
        TYPE_MATCH: bool(types) and chunk.document_type in types,
        BODY_MATCH: bool(wanted_bodies & set(chunk.bodies)),
    }


def _annotated(chunk: Chunk) -> bool:
    return isinstance(chunk.metadata.get(FUSION_SCORE), int | float)


def sort_key(chunk: Chunk) -> tuple[int, int, int, int, int, int, float, bool, bool, bool, bool, str]:
    fusion = chunk.metadata[FUSION_SCORE]
    assert isinstance(fusion, int | float)
    rank, order = chunk.metadata.get(FACET_RANK, 0), chunk.metadata.get(FACET_ORDER, 0)
    card = chunk.metadata.get(CATALOG_RANK)
    structure = chunk.metadata.get(STRUCTURE_RANK)
    return (
        0 if isinstance(card, int) else 1,
        card if isinstance(card, int) else 0,
        0 if isinstance(structure, int) else 1,
        structure if isinstance(structure, int) else 0,
        rank if isinstance(rank, int) else 0,
        order if isinstance(order, int) else 0,
        -float(fusion),
        chunk.status != ValidityStatus.VIGENTE.value,
        chunk.metadata.get(BRANCH_MATCH) is not True,
        chunk.metadata.get(TYPE_MATCH) is not True,
        chunk.metadata.get(BODY_MATCH) is not True,
        chunk.chunk_id,
    )


def ordered(chunks: Sequence[Chunk]) -> list[Chunk]:
    """A ordem, com cada trecho uma vez: o mesmo trecho trazido por duas tools fica na melhor posição."""
    annotated = sorted((c for c in chunks if _annotated(c)), key=sort_key)
    seen: set[str] = set()
    result: list[Chunk] = []
    for chunk in [*annotated, *(c for c in chunks if not _annotated(c))]:
        if chunk.chunk_id not in seen:
            seen.add(chunk.chunk_id)
            result.append(chunk)
    return result


def by_unit(chunks: Sequence[Chunk], max_tokens: int) -> list[Chunk]:
    """Os irmãos de uma unidade juntos (103 FR-017, ADR-048): o primeiro trecho de cada unidade puxa, para logo depois
    dele, os outros entregues da mesma unidade, na ordem do documento, até `max_tokens` estimados; o que passa do teto
    fica onde estava. O conjunto não muda — só a ordem."""
    result: list[Chunk] = []
    placed: set[str] = set()
    for position, chunk in enumerate(chunks):
        if chunk.chunk_id in placed:
            continue
        group = chunk.metadata.get(UNIT_GROUP)
        if not isinstance(group, str):
            result.append(chunk)
            placed.add(chunk.chunk_id)
            continue
        taken: list[Chunk] = []
        tokens = 0
        for sibling in chunks[position:]:
            if sibling.chunk_id in placed or sibling.metadata.get(UNIT_GROUP) != group:
                continue
            size = math.ceil(len(sibling.text) / 4)
            if taken and tokens + size > max_tokens:
                break
            taken.append(sibling)
            tokens += size
        taken.sort(key=lambda c: (c.page or 0, c.chunk_id))
        result.extend(taken)
        placed.update(c.chunk_id for c in taken)
    return result
