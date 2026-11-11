"""`ScoutingRetriever` e a tool `search_docs` (spec 103: FR-001–FR-004, FR-006–FR-009, FR-011; research R2–R8, R13).

Busca híbrida no índice da 102 com os filtros que a 101 publicou, traduzidos sem ler relógio. Devolve `Chunk`s da 001
com proveniência completa e `score` = similaridade densa (o número absoluto que o portão do `rag_first` compara), na
ordem da fusão com os empates desfeitos por metadados — as marcas vão nos metadados para o reranker reconstruir a
ordem. Com facetas (`branches`, ADR-045), uma busca por ramo — a pergunta seguida do nome do ramo, com o filtro dele —,
intercaladas: o primeiro de cada ramo, depois o segundo. Trechos de texto idêntico (o mesmo bloco repetido em cada
insígnia) ficam um só, com os rótulos dos outros nos metadados (FR-016, ADR-048). Erros do índice (`IndexUnavailable`,
`IndexMismatch`) passam sem tradução: são explícitos (FR-007).

Desde o ADR-063, refeito pelo ADR-064 (103 FR-022): o plano que a 101 publicou (`plan`) roda no grafo de domínio
(106), e os trechos da evidência — do Estatuto ao regulamento regional — entram à frente da busca, até 3 por documento
(os que mais casam com a pergunta) e 3 documentos por consulta. É o que garante a regra de cima mesmo quando a pergunta
fala de um caso e a norma fala em abstrato. A busca híbrida roda sempre; sem plano, nada vem à frente.

Desde o ADR-061: na pergunta de seguimento, uma busca com a âncora antes da pergunta e outra só com a pergunta,
juntas pela melhor posição de cada trecho — a âncora traz o assunto, e a pergunta sozinha protege de uma âncora
errada. E a busca filtrada por órgão ou abrangência que volta vazia é refeita sem esses dois filtros, com a marca
`relaxed` nos trechos.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from typing import Final

from axon_ai_contracts import Chunk, JSONValue, NamedRetriever

from axon_scouting.rag.graph.api import DomainGraphSource, Evidence, QueryPlan
from axon_scouting.rag.index.api import ChunkIndex, IndexedChunk, IndexFilters, Shape
from axon_scouting.rag.retrieval.domain.catalog import BRANCH_NAMES
from axon_scouting.rag.retrieval.domain.config import RetrievalConfig
from axon_scouting.rag.retrieval.domain.filters import (
    anchor,
    facets,
    index_filters,
    plan,
    suggested_bodies,
    suggested_types,
)
from axon_scouting.rag.retrieval.domain.ordering import (
    ALSO_IN,
    FACET,
    FACET_ORDER,
    FACET_RANK,
    FUSION_SCORE,
    GRAPH_QUERY,
    GRAPH_RANK,
    RELAXED,
    ROLE,
    marks,
    ordered,
)
from axon_scouting.rag.retrieval.domain.summaries import RECENCY
from axon_scouting.rag.retrieval.domain.terms import bag, idf, normalize, query_terms, score

GRAPH_PER_DOCUMENT: Final = 3
"""Trechos por documento da evidência do grafo, escolhidos pela pergunta (ADR-063; clarify da 106). Os documentos por
consulta (3) são cortados pelo executor da 106."""
GRAPH_DOCUMENTS: Final = 3
"""Documentos por consulta do plano (ADR-063; clarify da 106): os primeiros pelo papel, pelo nível e pela relevância."""
GRAPH_DEPTH: Final = 200
"""Profundidade da busca que ordena os trechos de cada unidade do grafo: o trecho que a busca híbrida põe mais acima
vence; o BM25 da pergunta desempata e ordena os que ela não alcança (106 R19)."""
GRAPH_TOTAL: Final = 8
"""Vagas do grafo à frente da busca (clarify da 106; G2), repartidas por grupo do plano sem passar do total: a busca
híbrida fica com o resto. Dentro do grupo, os trechos se alternam entre os documentos (o melhor de cada, depois o
segundo)."""

TOOL_NAME: Final = "search_docs"
TOOL_DESCRIPTION: Final = (
    "Busca trechos dos documentos oficiais dos Escoteiros do Brasil — POR, estatuto, resoluções e manuais. Por padrão, "
    "só os vigentes na data da conversa; numa consulta sobre um período passado, também as versões anteriores, "
    "marcadas como tal. Devolve cada trecho com documento, versão e página."
)


class ScoutingRetriever:
    """Implementa `Retriever` (001) sobre o `ChunkIndex` da 102."""

    def __init__(
        self,
        index: ChunkIndex,
        config: RetrievalConfig | None = None,
        *,
        shape: Shape | None = None,
        graph: DomainGraphSource | None = None,
    ) -> None:
        self._index = index
        self._config = config or RetrievalConfig()
        self._shape = shape
        self._graph = graph
        """O grafo de domínio, que roda o plano da pergunta (ADR-064); sem grafo, a busca é só a do índice."""
        """Só trechos dessa forma (a tool de unidades busca as listas, ADR-048)."""

    async def search(
        self, query: str, *, top_k: int | None = None, filters: Mapping[str, JSONValue] | None = None
    ) -> list[Chunk]:
        limit = self._config.limit(top_k)
        base = dataclasses.replace(index_filters(filters), shape=self._shape)
        found = await self._search(query, anchor(filters), base, limit, filters)
        if found or not (base.bodies or base.scopes):
            return found
        relaxed = dataclasses.replace(base, bodies=(), scopes=())
        return [
            dataclasses.replace(chunk, metadata={**chunk.metadata, RELAXED: True})
            for chunk in await self._search(query, anchor(filters), relaxed, limit, filters)
        ]

    async def _search(
        self,
        query: str,
        anchor_: str | None,
        index_filters_: IndexFilters,
        limit: int,
        filters: Mapping[str, JSONValue] | None,
    ) -> list[Chunk]:
        branches = facets(filters)
        preferred = (suggested_types(filters), suggested_bodies(filters))
        if branches:
            text = query if anchor_ is None else f"{anchor_} {query}"
            anchored = await self._by_plan(query, plan(filters), index_filters_, preferred)
            room = max(len(branches), limit - len(anchored))  # as facetas repartem o que sobra (106 R17)
            found_ = await self._by_facet(text, index_filters_, branches, room, preferred)
            return without_repeats(ordered([*anchored, *found_]))[:limit]
        queries = [query] if anchor_ is None else [f"{anchor_} {query}", query]
        best: dict[str, IndexedChunk] = {}
        for text in queries:
            for chunk in await self._unique(text, index_filters_, limit, preferred):
                kept = best.get(chunk.chunk_id)
                if kept is None or (chunk.score or 0.0) > (kept.score or 0.0):
                    best[chunk.chunk_id] = chunk
        found = [_to_chunk(chunk, index_filters_, *preferred) for chunk in best.values()]
        anchored = await self._by_plan(query, plan(filters), index_filters_, preferred)
        return without_repeats(ordered([*anchored, *found]))[:limit]

    async def _by_plan(
        self,
        query: str,
        plan_: QueryPlan | None,
        filters: IndexFilters,
        preferred: tuple[tuple[str, ...], tuple[str, ...]],
    ) -> list[Chunk]:
        """Os trechos da evidência do plano (ADR-064; 103 FR-022). Em cada grupo:
        - os documentos vêm na ordem do papel e do nível (a norma de cima primeiro, como o executor os deu) e, no mesmo
          papel e nível, pela relevância à pergunta — a nota do melhor trecho, num BM25 sobre os trechos do grupo;
        - ficam os `GRAPH_DOCUMENTS` primeiros, com até `GRAPH_PER_DOCUMENT` trechos cada, os que mais casam;
        - os trechos se alternam entre os documentos (o melhor de cada, depois o segundo), até a parte do grupo nas
          `GRAPH_TOTAL` vagas.
        A tool de unidades (`shape`) não usa o grafo."""
        if self._graph is None or plan_ is None or self._shape is not None:
            return []
        groups = [group for group in await self._graph.run(plan_) if group.evidence]
        if not groups:
            return []
        wanted = [c for group in groups for e in group.evidence for c in e.chunk_ids]
        chunks = {chunk.chunk_id: chunk for chunk in await self._index.get(list(dict.fromkeys(wanted)))}
        terms = query_terms(query)
        shares = _shares(len(groups))
        unbounded = dataclasses.replace(filters, branch=None, bodies=(), scopes=(), document_types=())
        ranks: dict[str, dict[str, int]] = {}
        result: list[tuple[str, str, IndexedChunk]] = []
        seen: set[str] = set()
        for group, share in zip(groups, shares, strict=True):
            text = _group_text(query, group.entity)
            if text not in ranks:  # a ordem da busca híbrida, funda, para escolher dentro das unidades (106 R19)
                found = await self._index.search(text, filters=unbounded, limit=GRAPH_DEPTH)
                ranks[text] = {chunk.chunk_id: position for position, chunk in enumerate(found)}
            # "As três últimas": na linha do tempo, as atas mais recentes primeiro, e não as mais relevantes (106 G4).
            timeline = group.query == "linha_do_tempo"
            recent = timeline and bool(RECENCY & set(normalize(query).split()))
            best = _documents(group.evidence, chunks, terms, ranks[text], recent=recent, timeline=timeline)
            taken = 0
            for turn in range(GRAPH_PER_DOCUMENT):
                for role, picked in best:
                    if taken < share and turn < len(picked) and picked[turn].chunk_id not in seen:
                        seen.add(picked[turn].chunk_id)
                        result.append((group.query, role, picked[turn]))
                        taken += 1
        return [
            dataclasses.replace(
                marked := _to_chunk(chunk, filters, *preferred),
                metadata={**marked.metadata, GRAPH_RANK: rank, GRAPH_QUERY: query_name, ROLE: role},
            )
            for rank, (query_name, role, chunk) in enumerate(result)
        ]

    async def _unique(
        self,
        query: str,
        filters: IndexFilters,
        limit: int,
        preferred: tuple[tuple[str, ...], tuple[str, ...]],
    ) -> list[IndexedChunk]:
        """Os achados de uma consulta; quando a junção de textos idênticos libera vagas, a mesma busca um pouco mais
        funda as preenche."""
        found = await self._index.search(query, filters=filters, limit=limit)
        unique = without_repeats(ordered([_to_chunk(chunk, filters, *preferred) for chunk in found]))
        if len(unique) < len(found):
            found = await self._index.search(query, filters=filters, limit=limit + len(found) - len(unique))
        return found

    async def _by_facet(
        self,
        query: str,
        base: IndexFilters,
        branches: tuple[str, ...],
        limit: int,
        preferred: tuple[tuple[str, ...], tuple[str, ...]],
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
                marked = _to_chunk(chunk, facet_filters, *preferred)
                metadata = {**marked.metadata, FACET: branch, FACET_RANK: rank, FACET_ORDER: order}
                chunks.append(dataclasses.replace(marked, metadata=metadata))
        return ordered(chunks)[:limit]


def _shares(groups: int) -> list[int]:
    """As `GRAPH_TOTAL` vagas repartidas entre os grupos, sem passar do total: as sobras vão aos primeiros, e cada grupo
    tem ao menos uma (com mais grupos que vagas, o total passa a ser o número de grupos)."""
    base, extra = divmod(GRAPH_TOTAL, groups)
    return [max(1, base + (1 if index < extra else 0)) for index in range(groups)]


def _group_text(query: str, entity: str | None) -> str:
    """A pergunta e, no grupo de um ramo (`para_cada`), o nome dele — como a busca por faceta (ADR-045)."""
    if entity and entity.startswith("ramo:") and entity[5:] in BRANCH_NAMES:
        return f"{query} Ramo {BRANCH_NAMES[entity[5:]]}"
    return query


def _documents(
    evidence: Sequence[Evidence],
    chunks: Mapping[str, IndexedChunk],
    terms: list[str],
    ranks: Mapping[str, int] | None = None,
    *,
    recent: bool = False,
    timeline: bool = False,
) -> list[tuple[str, list[IndexedChunk]]]:
    """Os `GRAPH_DOCUMENTS` documentos da evidência, com os `GRAPH_PER_DOCUMENT` trechos que mais casam com a pergunta
    em cada um, o melhor primeiro: na ordem do papel e do nível que o executor deu e, no mesmo papel e nível, pela nota
    do melhor trecho. A nota é a posição na busca híbrida (`ranks`) e, para os trechos que ela não alcança, o BM25
    sobre os trechos do grupo. Com `recent`, no mesmo papel e nível, o documento mais recente vem antes. Na linha do
    tempo (`timeline`), a evidência vem por data, e não por papel e nível: as atas ficam numa faixa só, e as cópias de
    uma reunião (a mesma data) valem uma (106 R21)."""
    tiers: dict[str, tuple[int, list[str], str]] = {}
    latest: dict[str, int] = {}
    for item in evidence:
        if item.date is not None:
            latest[item.doc_key] = max(latest.get(item.doc_key, 0), item.date.toordinal())
    rank = 0
    last: tuple[str, str | None, str] | None = None
    for item in evidence:
        tier = ("", None, "") if timeline else (item.role, item.scope, item.edge.type)  # DE_AREA vem depois
        if tier != last:
            rank, last = rank + 1, tier
        if item.doc_key not in tiers:
            tiers[item.doc_key] = (rank, [], item.role)
        tiers[item.doc_key][1].extend(item.chunk_ids)
    present = {c: chunks[c] for _, ids, _ in tiers.values() for c in ids if c in chunks}
    counts = {chunk_id: bag(chunk.text) for chunk_id, chunk in present.items()}
    weights = idf(list(counts.values()), terms)
    scored: list[tuple[int, tuple[int, int, float], str, str, list[IndexedChunk]]] = []
    for doc_key, (tier_rank, ids, role) in tiers.items():
        unique = [c for c in dict.fromkeys(ids) if c in present]
        if not unique:
            continue
        notes = {c: ((ranks or {}).get(c, GRAPH_DEPTH), -score(counts[c], terms, weights)) for c in unique}
        picked = [present[c] for c in sorted(unique, key=lambda c: notes[c])[:GRAPH_PER_DOCUMENT]]
        order = -latest.get(doc_key, 0) if recent else 0
        scored.append((tier_rank, (order, *min(notes.values())), doc_key, role, picked))
    scored.sort(key=lambda item: (item[0], item[1]))
    if timeline:  # as cópias de uma reunião (a registrada, a do CNPJ da filial) contam como uma ata
        meetings: set[int] = set()
        kept: list[tuple[int, tuple[int, int, float], str, str, list[IndexedChunk]]] = []
        for candidate in scored:
            when = latest.get(candidate[2])
            if when is None or when not in meetings:
                kept.append(candidate)
                if when is not None:
                    meetings.add(when)
        scored = kept
    return [(role, picked) for _, _, _, role, picked in scored[:GRAPH_DOCUMENTS]]


def _best(ids: list[str], chunks: Mapping[str, IndexedChunk], terms: list[str], limit: int) -> list[IndexedChunk]:
    """Os `limit` trechos que mais casam com a pergunta (BM25 entre eles), na ordem em que vieram."""
    present = [chunks[i] for i in dict.fromkeys(ids) if i in chunks]
    counts = {chunk.chunk_id: bag(chunk.text) for chunk in present}
    weights = idf(list(counts.values()), terms)
    best = sorted(present, key=lambda c: -score(counts[c.chunk_id], terms, weights))[:limit]
    return sorted(best, key=lambda c: ids.index(c.chunk_id))


def search_docs(retriever: ScoutingRetriever, *, collection: str = "scouting") -> NamedRetriever:
    """A tool `scouting.search_docs` (FR-006): só aparece se o grant conceder a coleção (o nome do índice)."""
    return NamedRetriever(
        name=TOOL_NAME, description=TOOL_DESCRIPTION, collections=frozenset({collection}), retriever=retriever
    )


def without_repeats(chunks: list[Chunk]) -> list[Chunk]:
    """Um trecho por corpo: o de melhor posição fica, com os rótulos dos de texto idêntico em `also_in` (FR-016)."""
    kept: dict[str, Chunk] = {}
    also: dict[str, list[str]] = {}
    for chunk in chunks:
        body = " ".join(chunk.text.split("\n", 1)[-1].split()).casefold()
        if body in kept:
            also[body].append(chunk.text.split("\n", 1)[0])
            continue
        kept[body] = chunk
        also[body] = []
    return [
        dataclasses.replace(chunk, metadata={**chunk.metadata, ALSO_IN: also[body]}) if also[body] else chunk
        for body, chunk in kept.items()
    ]


def _to_chunk(
    chunk: IndexedChunk, filters: IndexFilters, preferred: tuple[str, ...] = (), bodies: tuple[str, ...] = ()
) -> Chunk:
    """O `Chunk` da 102 com o score trocado pela similaridade e as marcas de ordem nos metadados (R2, R8)."""
    base = chunk.to_chunk()
    similarity = 0.0 if chunk.similarity is None else min(max(chunk.similarity, 0.0), 1.0)
    metadata: dict[str, JSONValue] = {
        **base.metadata,
        FUSION_SCORE: float(chunk.score or 0.0),
        **marks(chunk, filters, preferred, bodies),
    }
    return dataclasses.replace(base, score=similarity, metadata=metadata)
