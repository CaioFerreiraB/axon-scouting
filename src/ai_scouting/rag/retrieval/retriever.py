"""`ScoutingRetriever` e a tool `search_docs` (spec 103: FR-001–FR-004, FR-006–FR-009; research R2–R8).

Busca híbrida no índice da 102 com os filtros que a 101 publicou, traduzidos sem ler relógio. Devolve `Chunk`s da 001
com proveniência completa e `score` = similaridade densa (o número absoluto que o portão do `rag_first` compara), na
ordem da fusão com os empates desfeitos por metadados — as marcas vão nos metadados para o reranker reconstruir a
ordem. Erros do índice (`IndexUnavailable`, `IndexMismatch`) passam sem tradução: são explícitos (FR-007).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Final

from ai_core_contracts import Chunk, JSONValue, NamedRetriever

from ai_scouting.rag.index.api import ChunkIndex, IndexedChunk, IndexFilters
from ai_scouting.rag.retrieval.domain.config import RetrievalConfig
from ai_scouting.rag.retrieval.domain.filters import index_filters
from ai_scouting.rag.retrieval.domain.ordering import FUSION_SCORE, marks, ordered

TOOL_NAME: Final = "search_docs"
TOOL_DESCRIPTION: Final = (
    "Busca trechos dos documentos oficiais dos Escoteiros do Brasil — POR, estatuto, resoluções e manuais. Por padrão, "
    "só os vigentes na data da conversa; numa consulta sobre um período passado, também as versões anteriores, "
    "marcadas como tal. Devolve cada trecho com documento, versão e página."
)


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
        found = await self._index.search(query, filters=index_filters_, limit=limit)
        return ordered([_to_chunk(chunk, index_filters_) for chunk in found])


def search_docs(retriever: ScoutingRetriever, *, collection: str = "scouting") -> NamedRetriever:
    """A tool `scouting.search_docs` (FR-006): só aparece se o grant conceder a coleção (o nome do índice)."""
    return NamedRetriever(
        name=TOOL_NAME, description=TOOL_DESCRIPTION, collections=frozenset({collection}), retriever=retriever
    )


def _to_chunk(chunk: IndexedChunk, filters: IndexFilters) -> Chunk:
    """O `Chunk` da 102 com o score trocado pela similaridade e as marcas de ordem nos metadados (R2, R8)."""
    base = chunk.to_chunk()
    similarity = 0.0 if chunk.similarity is None else min(max(chunk.similarity, 0.0), 1.0)
    metadata: dict[str, JSONValue] = {**base.metadata, FUSION_SCORE: float(chunk.score or 0.0), **marks(chunk, filters)}
    return dataclasses.replace(base, score=similarity, metadata=metadata)
