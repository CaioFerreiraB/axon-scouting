"""`SummariesRetriever` e a tool `search_summaries` (spec 103; ADR-062).

Responde ao pedido de síntese que a 101 publicou (`synthesis`): os resumos das atas — linha do tempo por órgão e ano e
fichas das reuniões — que a 102 montou das fichas conferidas, escolhidos por abrangência, anos e termos da pergunta
(`domain/summaries.py`). Sem pedido, não devolve nada: no `rag_first` todas as tools de recuperação rodam, e o resumo
só aparece na pergunta de síntese. O que ela traz vai à frente dos trechos no `MetadataReranker`, e os trechos das atas
continuam vindo pela busca comum, para a resposta citar a fonte.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from axon_ai_contracts import Chunk, JSONValue, NamedRetriever

from axon_scouting.rag.index.api import MeetingSummaries, SummaryEntry, ValidityStatus
from axon_scouting.rag.retrieval.domain.filters import index_filters, synthesis_request
from axon_scouting.rag.retrieval.domain.ordering import BODY_MATCH, BRANCH_MATCH, FUSION_SCORE, SUMMARY_RANK, TYPE_MATCH
from axon_scouting.rag.retrieval.domain.summaries import select

TOOL_NAME: Final = "search_summaries"
TOOL_DESCRIPTION: Final = (
    "Busca os resumos das atas dos órgãos dos Escoteiros do Brasil — a linha do tempo de cada órgão por ano e a ficha "
    "de cada reunião, com as deliberações, as eleições e os números e a página da ata de cada item —, para perguntas "
    "sobre o que aconteceu ao longo do tempo. Fora de uma pergunta de síntese, não devolve nada."
)
SECTIONS: Final = {"ficha": "ficha da reunião (resumo da ata)", "linha_do_tempo": "linha do tempo (resumo das atas)"}


class SummariesRetriever:
    """Implementa `Retriever` (001) sobre `MeetingSummaries` da 102."""

    def __init__(self, source: MeetingSummaries) -> None:
        self._source = source

    async def search(
        self, query: str, *, top_k: int | None = None, filters: Mapping[str, JSONValue] | None = None
    ) -> list[Chunk]:
        index_filters(filters)  # o contrato de filtros vale igual nas quatro tools
        request = synthesis_request(filters)
        if request is None:
            return []
        picked = select(await self._source.summaries(), request, query)
        if top_k is not None:
            picked = picked[: max(top_k, 0)]
        return [summary_chunk(entry, rank=rank) for rank, entry in enumerate(picked)]


def summary_chunk(entry: SummaryEntry, *, rank: int) -> Chunk:
    """O resumo como `Chunk` da 001: ata e página quando é a ficha de uma reunião; a linha do tempo cita as atas no
    próprio texto, linha a linha."""
    metadata: dict[str, JSONValue] = {
        "document_type": "ata",
        "summary_kind": entry.kind,
        "doc_keys": list(entry.doc_keys),
        "year": entry.year,
        "source_url": entry.source_url,
        FUSION_SCORE: 0.0,
        SUMMARY_RANK: rank,
        BRANCH_MATCH: False,
        TYPE_MATCH: False,
        BODY_MATCH: False,
    }
    return Chunk(
        chunk_id=entry.summary_id,
        text=entry.text,
        score=0.0,
        document=entry.title,
        version=None,
        status=ValidityStatus.VIGENTE.value,
        section=SECTIONS.get(entry.kind, entry.kind),
        page=entry.page,
        metadata=metadata,
    )


def search_summaries(retriever: SummariesRetriever, *, collection: str = "scouting") -> NamedRetriever:
    """A tool `scouting.search_summaries`: a mesma coleção da busca, então o mesmo grant a libera."""
    return NamedRetriever(
        name=TOOL_NAME, description=TOOL_DESCRIPTION, collections=frozenset({collection}), retriever=retriever
    )
