"""`UnitsRetriever` e a tool `list_units` (spec 103 FR-015; research R17; ADR-048).

Responde ao pedido de enumeração que a 101 publicou. Com o tipo de unidade, uma ficha por documento com a lista exata
das unidades desse tipo — a árvore da 102, pelos mesmos filtros de vigência, tipo e ramo (o da unidade) da busca —;
sem tipo, os trechos com forma de lista mais próximos da pergunta, que na busca comum perdem da prosa. Sem pedido, não
devolve nada. O que ela traz vai à frente dos trechos no `MetadataReranker`, depois das fichas do acervo.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Final

from axon_ai_contracts import Chunk, JSONValue, NamedRetriever

from axon_scouting.rag.index.api import ChunkIndex, DocumentOutline
from axon_scouting.rag.retrieval.domain.config import RetrievalConfig
from axon_scouting.rag.retrieval.domain.filters import enumeration_request, facets, index_filters
from axon_scouting.rag.retrieval.domain.ordering import STRUCTURE_RANK
from axon_scouting.rag.retrieval.domain.units import by_document, units_card
from axon_scouting.rag.retrieval.retriever import ScoutingRetriever

TOOL_NAME: Final = "list_units"
TOOL_DESCRIPTION: Final = (
    "Lista as unidades dos documentos dos Escoteiros do Brasil que uma pergunta de enumeração pede — especialidades, "
    "insígnias, blocos, artigos, reuniões —, com o caminho e a página de cada uma, ou os trechos que são listas. Fora "
    "de um pedido de enumeração, não devolve nada."
)
MAX_LISTS: Final = 2
"""Trechos com forma de lista quando a pergunta não nomeia um tipo de unidade."""
MAX_OUTLINES: Final = 1
"""Trecho de sumário (os dois primeiros níveis da árvore de um documento), depois das listas. Sem tipo, a rota é uma
aposta e custa no máximo 3 vagas: com 3 listas e 2 sumários, as 5 vagas tiravam da entrega trechos que respondiam
(103 R17)."""
MAX_DOCUMENTS: Final = 3
"""Unidades do tipo em mais documentos que isso: a pergunta não disse de onde listar ("quais são os blocos", sem ramo),
e a tool devolve os trechos com forma de lista."""


class UnitsRetriever:
    """Implementa `Retriever` (001) sobre o `DocumentOutline` e o `ChunkIndex` da 102."""

    def __init__(self, index: ChunkIndex, outline: DocumentOutline, config: RetrievalConfig | None = None) -> None:
        self._index = index
        self._outline = outline
        self._config = config or RetrievalConfig()

    async def search(
        self, query: str, *, top_k: int | None = None, filters: Mapping[str, JSONValue] | None = None
    ) -> list[Chunk]:
        base = index_filters(filters)
        request = enumeration_request(filters)
        if request is None:
            return []
        groups = (
            [] if request.unit_kind is None else by_document(await self._outline.units(request.unit_kind, filters=base))
        )
        if groups and len(groups) <= MAX_DOCUMENTS:
            kinds = {kind.name: kind for kind in await self._outline.unit_kinds()}
            limit = min(self._config.limit(top_k), self._config.max_cards)
            return [
                units_card(
                    document,
                    units,
                    kind=kinds.get(groups[0][1][0].unit.kind),
                    rank=rank,
                    total=len(groups),
                    filters=base,
                )
                for rank, (document, units) in enumerate(groups[:limit])
            ]
        if facets(filters):  # a pergunta já se divide por ramo: as facetas espalham as vagas, e a aposta as tiraria
            return []
        lists = await ScoutingRetriever(self._index, self._config, shape="list").search(
            query, top_k=MAX_LISTS, filters=filters
        )
        outlines = await ScoutingRetriever(self._index, self._config, shape="outline").search(
            query, top_k=MAX_OUTLINES, filters=filters
        )
        return [
            dataclasses.replace(chunk, metadata={**chunk.metadata, STRUCTURE_RANK: rank})
            for rank, chunk in enumerate([*lists[:MAX_LISTS], *outlines[:MAX_OUTLINES]])
        ]


def list_units(retriever: UnitsRetriever, *, collection: str = "scouting") -> NamedRetriever:
    """A tool `scouting.list_units`: a mesma coleção da busca, então o mesmo grant a libera."""
    return NamedRetriever(
        name=TOOL_NAME, description=TOOL_DESCRIPTION, collections=frozenset({collection}), retriever=retriever
    )
