"""`CatalogRetriever` e a tool `list_documents` (spec 103 FR-013, FR-014; research R16; ADR-047).

Responde ao pedido de lista que a 101 publicou: uma ficha por documento do acervo cujo título casa, com versão, vigência
e o link do original, sob os mesmos filtros de vigência, ramo e tipo da busca por trechos. Sem pedido, não devolve
nada — no `rag_first` as duas tools rodam sempre, e a ficha só aparece quando a pergunta pede a lista. As fichas
levam a marca que as põe à frente dos trechos no `MetadataReranker`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from axon_ai_contracts import Chunk, JSONValue, NamedRetriever

from axon_scouting.rag.index.api import DocumentCatalog
from axon_scouting.rag.retrieval.domain.catalog import card, select
from axon_scouting.rag.retrieval.domain.config import RetrievalConfig
from axon_scouting.rag.retrieval.domain.filters import catalog_request, index_filters, suggested_types

TOOL_NAME: Final = "list_documents"
TOOL_DESCRIPTION: Final = (
    "Lista os documentos do acervo dos Escoteiros do Brasil que uma pergunta de enumeração pede — políticas, manuais, "
    "resoluções, atas —, uma ficha por documento, com versão, vigência e o link do original. Fora de um pedido de "
    "lista, não devolve nada."
)


class CatalogRetriever:
    """Implementa `Retriever` (001) sobre o `DocumentCatalog` da 102."""

    def __init__(self, catalog: DocumentCatalog, config: RetrievalConfig | None = None) -> None:
        self._catalog = catalog
        self._config = config or RetrievalConfig()

    async def search(
        self, query: str, *, top_k: int | None = None, filters: Mapping[str, JSONValue] | None = None
    ) -> list[Chunk]:
        base = index_filters(filters)
        request = catalog_request(filters)
        if request is None:
            return []
        picked = select(await self._catalog.catalog(filters=base), request)
        limit = min(self._config.limit(top_k), self._config.max_cards)
        preferred = suggested_types(filters)
        return [
            card(entry, rank=rank, total=len(picked), score=score, filters=base, preferred=preferred)
            for rank, (entry, score) in enumerate(picked[:limit])
        ]


def list_documents(retriever: CatalogRetriever, *, collection: str = "scouting") -> NamedRetriever:
    """A tool `scouting.list_documents`: a mesma coleção da busca, então o mesmo grant a libera."""
    return NamedRetriever(
        name=TOOL_NAME, description=TOOL_DESCRIPTION, collections=frozenset({collection}), retriever=retriever
    )
