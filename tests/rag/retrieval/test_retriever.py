"""`ScoutingRetriever` sobre o índice em memória (spec 103: US1 T009, US2 T012, US3 T014, US5 T018)."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

import pytest
from axon_ai_contracts import Chunk, NamedRetriever, Retriever

from axon_scouting.rag.index.adapters.bm25 import Bm25Encoder
from axon_scouting.rag.index.adapters.memory import HashingEmbedder, MemoryIndex, MemoryStorage
from axon_scouting.rag.index.api import IndexMismatch, IndexUnavailable
from axon_scouting.rag.retrieval.domain.filters import InvalidFilters
from axon_scouting.rag.retrieval.domain.ordering import BRANCH_MATCH, FUSION_SCORE, TYPE_MATCH, sort_key
from axon_scouting.rag.retrieval.retriever import ScoutingRetriever, search_docs

from .corpus import filtros, indice

HOJE = filtros()


def buscar(query: str, raw: Mapping[str, Any] | None = HOJE, *, top_k: int | None = None, **kw: Any) -> list[Chunk]:
    async def go() -> list[Chunk]:
        retriever = ScoutingRetriever(await indice(**kw))
        return list(await retriever.search(query, top_k=top_k, filters=raw))

    return asyncio.run(go())


# --- US1 -------------------------------------------------------------------------------------------------------


def test_us1_termo_exato_pelo_esparso() -> None:
    """US1 cen. 1: o trecho está entre os candidatos, primeiro na busca esparsa (fusão ≥ 1/2)."""
    achados = {c.chunk_id: c for c in buscar("o que diz a Regra 33")}
    assert "por-2025#regra33@p38" in achados
    assert achados["por-2025#regra33@p38"].metadata[FUSION_SCORE] >= 0.5


def test_proveniencia_score_e_marcas() -> None:
    for chunk in buscar("acampamento da tropa", top_k=10):
        assert chunk.chunk_id and chunk.document and chunk.version and chunk.status and chunk.page
        assert 0.0 <= chunk.score <= 1.0  # similaridade densa (clarificação 1)
        assert isinstance(chunk.metadata[FUSION_SCORE], float)
        assert isinstance(chunk.metadata[BRANCH_MATCH], bool) and isinstance(chunk.metadata[TYPE_MATCH], bool)
        assert "document_id" in chunk.metadata and "effective_date" in chunk.metadata
    (regra,) = [c for c in buscar("o que diz a Regra 33") if c.chunk_id == "por-2025#regra33@p38"]
    assert (regra.document, regra.section, regra.page) == ("POR", "Regra 033", 38)


def test_score_e_a_similaridade_da_102() -> None:
    async def go() -> None:
        index = await indice()
        brutos = await index.search("acampamento da tropa", filters=_f(HOJE), limit=10)
        devolvidos = await ScoutingRetriever(index).search("acampamento da tropa", top_k=10, filters=HOJE)
        por_id = {c.chunk_id: c for c in brutos}
        for chunk in devolvidos:
            similaridade = por_id[chunk.chunk_id].similarity
            assert similaridade is not None
            assert chunk.score == pytest.approx(min(max(similaridade, 0.0), 1.0))
            assert chunk.metadata[FUSION_SCORE] == por_id[chunk.chunk_id].score

    asyncio.run(go())


def test_ordem_da_chave() -> None:
    devolvidos = buscar("acampamento da tropa", top_k=10)
    assert devolvidos == sorted(devolvidos, key=sort_key)


def test_limites() -> None:
    assert len(buscar("acampamento tropa uniforme regra promessa grupo")) <= 5  # sem top_k: default_limit
    assert len(buscar("acampamento tropa uniforme regra promessa grupo", top_k=50)) <= 30
    with pytest.raises(ValueError):
        buscar("acampamento", top_k=0)


def test_vazio_e_erros() -> None:
    assert buscar("de que o a") == []
    with pytest.raises(InvalidFilters):
        buscar("acampamento", None)

    async def sem_build() -> None:
        index = MemoryIndex("scouting", HashingEmbedder(), Bm25Encoder(), storage=MemoryStorage())
        await ScoutingRetriever(index).search("acampamento", filters=HOJE)

    with pytest.raises(IndexUnavailable):  # FR-007: erro explícito, nunca vazio
        asyncio.run(sem_build())

    async def divergente() -> None:
        index = await indice()
        assert isinstance(index, MemoryIndex)
        outro = MemoryIndex("scouting", HashingEmbedder(fingerprint="outro"), Bm25Encoder(), storage=index.storage)
        await ScoutingRetriever(outro).search("acampamento", filters=HOJE)

    with pytest.raises(IndexMismatch):
        asyncio.run(divergente())


def test_tool_search_docs() -> None:
    async def go() -> None:
        retriever = ScoutingRetriever(await indice())
        assert isinstance(retriever, Retriever)
        tool = search_docs(retriever, collection="scouting")
        assert isinstance(tool, NamedRetriever)
        assert (tool.name, tool.collections, tool.retriever) == ("search_docs", frozenset({"scouting"}), retriever)
        assert "Escoteiros do Brasil" in tool.description

    asyncio.run(go())


# --- US2: só o vigente, por default -----------------------------------------------------------------------------


def documentos(achados: list[Chunk]) -> set[str]:
    return {str(c.metadata["document_id"]) + "-" + str(c.version) for c in achados}


def test_us2_default_so_o_vigente() -> None:
    achados = buscar("acampamento da tropa", top_k=30)
    assert "manual-x-2026" in documentos(achados)
    assert "manual-x-2023" not in documentos(achados)  # substituída
    assert "manual-x-2027" not in documentos(achados)  # rascunho


# --- US3: data passada --------------------------------------------------------------------------------------------

PERIODO_2023 = {"effective_from": "2023-01-01", "effective_until": "2023-12-31"}


def test_us3_periodo_passado_com_status_atual() -> None:
    achados = buscar("acampamento da tropa", {**PERIODO_2023, "include_superseded": True}, top_k=30)
    (antigo,) = [c for c in achados if c.metadata["document_id"] == "manual-x"]
    assert (antigo.version, antigo.status) == ("2023", "substituido")
    assert antigo.metadata["superseded_by"] == "manual-x-2026"
    # sem include_superseded a 2023 ainda vem: estava em vigor no fim do período
    assert "manual-x-2023" in documentos(buscar("acampamento da tropa", PERIODO_2023, top_k=30))


def test_us3_versao_anterior() -> None:
    achados = buscar("acampamento da tropa", filtros(previous_version=True), top_k=30)
    assert documentos(achados) == {"manual-x-2023"}


# --- US5: filtros de contexto ------------------------------------------------------------------------------------


def test_us5_ramo_restringe_e_marca() -> None:
    achados = buscar("acampamento da tropa", filtros(branch="pioneiro"), top_k=30)
    assert "manual-esc-1" not in documentos(achados)
    assert "manual-pio-1" in documentos(achados) and "manual-x-2026" in documentos(achados)  # o ramo e os gerais
    for chunk in achados:
        assert chunk.metadata[BRANCH_MATCH] is (chunk.metadata["branch"] == "pioneiro")
    sem_ramo = buscar("acampamento da tropa", top_k=30)
    assert {"manual-esc-1", "manual-pio-1"} <= documentos(sem_ramo)


def test_us5_tipos_restringem_e_marcam() -> None:
    achados = buscar("regra uniforme grupo promessa", filtros(document_types=["por"]), top_k=30)
    assert achados and {c.metadata["document_type"] for c in achados} == {"por"}
    assert all(c.metadata[TYPE_MATCH] is True for c in achados)


def test_us5_temas_nao_mudam_nada() -> None:
    sem = buscar("acampamento da tropa", top_k=30)
    com = buscar("acampamento da tropa", filtros(topics=["acampamento", "uniforme"]), top_k=30)
    assert [c.chunk_id for c in com] == [c.chunk_id for c in sem]


def test_nenhum_resultado_apos_filtros_e_vazio() -> None:
    assert buscar("acampamento", filtros(branch="lobinho", document_types=["estatuto"]), top_k=30) == []


def _f(raw: Mapping[str, Any]) -> Any:
    from axon_scouting.rag.retrieval.domain.filters import index_filters

    return index_filters(raw)


# --- facetas (FR-011, R13; ADR-045) --------------------------------------------------------------------------------


def test_facetas_intercalam_os_ramos_sem_repetir() -> None:
    """Uma busca por ramo: o primeiro de cada faceta antes do segundo de qualquer uma; trecho geral só uma vez."""
    achados = buscar("acampamento da tropa", filtros(branches=["escoteiro", "pioneiro"]), top_k=6)
    ids = [chunk.chunk_id for chunk in achados]
    assert len(ids) == len(set(ids)) <= 6
    primeiros = [(chunk.metadata["facet"], chunk.metadata["facet_rank"]) for chunk in achados[:2]]
    assert sorted(primeiros) == [("escoteiro", 0), ("pioneiro", 0)]
    assert [chunk.metadata["facet_rank"] for chunk in achados] == sorted(
        chunk.metadata["facet_rank"] for chunk in achados
    )
    for chunk in achados:
        assert chunk.metadata["branch"] in (None, chunk.metadata["facet"])
        assert chunk.metadata[BRANCH_MATCH] is (chunk.metadata["branch"] == chunk.metadata["facet"])


def test_a_ordem_das_facetas_sobrevive_ao_reranker() -> None:
    from axon_scouting.rag.retrieval.reranker import MetadataReranker

    achados = buscar("acampamento da tropa", filtros(branches=["escoteiro", "pioneiro"]), top_k=6)
    reordenados = asyncio.run(MetadataReranker().rerank("acampamento da tropa", list(reversed(achados)), top_k=2))
    assert sorted(chunk.metadata["facet"] for chunk in reordenados) == ["escoteiro", "pioneiro"]


@pytest.mark.parametrize(
    "branches",
    [["escoteiro"], ["escoteiro", "escoteiro"], ["escoteiro", "patrulha"], "escoteiro"],
    ids=["um-so", "repetido", "desconhecido", "texto"],
)
def test_facetas_invalidas(branches: object) -> None:
    with pytest.raises(InvalidFilters, match="branches"):
        buscar("acampamento", filtros(branches=branches))


def test_branch_e_branches_juntos_e_erro() -> None:
    with pytest.raises(InvalidFilters, match="branches"):
        buscar("acampamento", filtros(branch="pioneiro", branches=["escoteiro", "pioneiro"]))


def test_trecho_de_outra_faceta_nao_tira_a_posicao_zero() -> None:
    """O trecho geral que a primeira faceta já levou não conta na segunda: ela também começa em 0."""
    achados = buscar("acampamento da tropa", filtros(branches=["escoteiro", "pioneiro"]), top_k=30)
    for faceta in ("escoteiro", "pioneiro"):
        posicoes = sorted(chunk.metadata["facet_rank"] for chunk in achados if chunk.metadata["facet"] == faceta)
        assert posicoes == list(range(len(posicoes)))
