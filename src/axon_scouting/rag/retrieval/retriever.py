"""`ScoutingRetriever` e a tool `search_docs` (spec 103: FR-001–FR-004, FR-006–FR-009, FR-011; research R2–R8, R13).

Busca híbrida no índice da 102 com os filtros que a 101 publicou, traduzidos sem ler relógio. Devolve `Chunk`s da 001
com proveniência completa e `score` = similaridade densa (o número absoluto que o portão do `rag_first` compara), na
ordem da fusão com os empates desfeitos por metadados — as marcas vão nos metadados para o reranker reconstruir a
ordem. Com facetas (`branches`, ADR-045), uma busca por ramo — a pergunta seguida do nome do ramo, com o filtro dele —,
intercaladas: o primeiro de cada ramo, depois o segundo. Erros do índice (`IndexUnavailable`, `IndexMismatch`) passam
sem tradução: são explícitos (FR-007).
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from typing import Final

from axon_ai_contracts import Chunk, JSONValue, NamedRetriever

from axon_scouting.rag.index.api import ChunkIndex, DocumentType, IndexedChunk, IndexFilters
from axon_scouting.rag.retrieval.domain.config import RetrievalConfig
from axon_scouting.rag.retrieval.domain.filters import facets, index_filters, suggested_types
from axon_scouting.rag.retrieval.domain.ordering import FACET, FACET_ORDER, FACET_RANK, FUSION_SCORE, marks, ordered

TOOL_NAME: Final = "search_docs"
TOOL_DESCRIPTION: Final = (
    "Busca trechos dos documentos oficiais dos Escoteiros do Brasil — POR, estatuto, resoluções e manuais. Por padrão, "
    "só os vigentes na data da conversa; numa consulta sobre um período passado, também as versões anteriores, "
    "marcadas como tal. Devolve cada trecho com documento, versão e página."
)
BRANCH_NAMES: Final = {
    "filhotes": "Filhotes",
    "lobinho": "Lobinho",
    "escoteiro": "Escoteiro",
    "senior": "Sênior",
    "pioneiro": "Pioneiro",
}
"""O nome de cada ramo na subbusca da faceta ("… Ramo Sênior")."""


class ScoutingRetriever:
    """Implementa `Retriever` (001) sobre o `ChunkIndex` da 102."""

    def __init__(self, index: ChunkIndex, config: RetrievalConfig | None = None) -> None:
        self._index = index
        self._config = config or RetrievalConfig()

    async def search(
        self, query: str, *, top_k: int | None = None, filters: Mapping[str, JSONValue] | None = None
    ) -> list[Chunk]:
        limit = self._config.limit(top_k)
        index_filters_ = index_filters(filters)
        branches = facets(filters)
        preferred = suggested_types(filters)
        if not branches:
            found = await self._index.search(query, filters=index_filters_, limit=limit)
            return ordered([_to_chunk(chunk, index_filters_, preferred) for chunk in found])
        return ordered(await self._by_facet(query, index_filters_, branches, limit, preferred))

    async def _by_facet(
        self,
        query: str,
        base: IndexFilters,
        branches: tuple[str, ...],
        limit: int,
        preferred: tuple[DocumentType, ...],
    ) -> list[Chunk]:
        """Uma busca por ramo, `limit` repartido; um trecho achado por duas facetas fica só na primeira.

        Cada faceta busca com o `limit` inteiro e guarda os primeiros: o índice corta as listas densa e esparsa no
        limite antes da fusão, e um limite menor mudaria quem fica em primeiro.
        """
        per_facet = max(1, math.ceil(limit / len(branches)))
        seen: set[str] = set()
        chunks: list[Chunk] = []
        for order, branch in enumerate(branches):
            facet_filters = dataclasses.replace(base, branch=branch)
            found = await self._index.search(f"{query} Ramo {BRANCH_NAMES[branch]}", filters=facet_filters, limit=limit)
            kept = [chunk for chunk in found if chunk.chunk_id not in seen][:per_facet]
            for rank, chunk in enumerate(kept):  # a posição conta só o que a faceta guardou: toda faceta tem um 0
                seen.add(chunk.chunk_id)
                marked = _to_chunk(chunk, facet_filters, preferred)
                metadata = {**marked.metadata, FACET: branch, FACET_RANK: rank, FACET_ORDER: order}
                chunks.append(dataclasses.replace(marked, metadata=metadata))
        return ordered(chunks)[:limit]


def search_docs(retriever: ScoutingRetriever, *, collection: str = "scouting") -> NamedRetriever:
    """A tool `scouting.search_docs` (FR-006): só aparece se o grant conceder a coleção (o nome do índice)."""
    return NamedRetriever(
        name=TOOL_NAME, description=TOOL_DESCRIPTION, collections=frozenset({collection}), retriever=retriever
    )


def _to_chunk(chunk: IndexedChunk, filters: IndexFilters, preferred: tuple[DocumentType, ...] = ()) -> Chunk:
    """O `Chunk` da 102 com o score trocado pela similaridade e as marcas de ordem nos metadados (R2, R8)."""
    base = chunk.to_chunk()
    similarity = 0.0 if chunk.similarity is None else min(max(chunk.similarity, 0.0), 1.0)
    metadata: dict[str, JSONValue] = {
        **base.metadata,
        FUSION_SCORE: float(chunk.score or 0.0),
        **marks(chunk, filters, preferred),
    }
    return dataclasses.replace(base, score=similarity, metadata=metadata)
