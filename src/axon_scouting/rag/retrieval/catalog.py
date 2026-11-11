"""`CatalogRetriever` e a tool `list_documents` (spec 103 FR-013, FR-014; research R16; ADR-047).

Responde ao pedido de lista que a 101 publicou: uma ficha por documento do acervo cujo título casa, com versão, vigência
e o link do original, sob os mesmos filtros de vigência, ramo e tipo da busca por trechos. Sem pedido, não devolve
nada — no `rag_first` as duas tools rodam sempre, e a ficha só aparece quando a pergunta pede a lista. As fichas
levam a marca que as põe à frente dos trechos no `MetadataReranker`.

Desde o G6 da 106 (research R23), quais documentos podem entrar é a consulta `documentos` do grafo de domínio, pelo
tipo; o catálogo da 102 continua dando a vigência dos filtros, a fração dos qualificadores e o texto da ficha, e a
saída não muda. Sem grafo (um índice sem `graph.jsonl` ao lado), a lista é a do catálogo.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from axon_ai_contracts import Chunk, JSONValue, NamedRetriever

from axon_scouting.rag.graph.api import DomainGraphSource
from axon_scouting.rag.index.api import DocumentCatalog
from axon_scouting.rag.retrieval.domain.catalog import ANY_KIND, card, select
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

    def __init__(
        self,
        catalog: DocumentCatalog,
        config: RetrievalConfig | None = None,
        *,
        graph: DomainGraphSource | None = None,
    ) -> None:
        self._catalog = catalog
        self._config = config or RetrievalConfig()
        self._graph = graph

    async def search(
        self, query: str, *, top_k: int | None = None, filters: Mapping[str, JSONValue] | None = None
    ) -> list[Chunk]:
        base = index_filters(filters)
        request = catalog_request(filters)
        if request is None:
            return []
        entries = await self._catalog.catalog(filters=base)
        if self._graph is not None:  # a fachada (106 G6): o grafo diz quais documentos do tipo existem
            kind = {} if request.kind == ANY_KIND else {"tipo": request.kind}
            allowed = {e.doc_key for e in await self._graph.query("documentos", vigente=False, **kind)}
            if allowed or await self._graph.query("documentos", vigente=False):  # sem grafo, a lista de hoje
                entries = [entry for entry in entries if entry.doc_key in allowed]
        picked = select(entries, request)
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
