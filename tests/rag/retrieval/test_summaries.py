"""A busca de resumos das atas para a pergunta de síntese (ADR-062): seleção, tool, filtro e artefato."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from axon_ai_contracts import Chunk, NamedRetriever, Retriever

from axon_scouting.rag.index.adapters.summaries import ArtifactSummaries, read_summaries, summary_texts, write_summaries
from axon_scouting.rag.index.api import SummaryEntry
from axon_scouting.rag.retrieval.domain.filters import InvalidFilters
from axon_scouting.rag.retrieval.domain.ordering import SUMMARY_RANK, ordered
from axon_scouting.rag.retrieval.domain.summaries import (
    EVENT_CARDS,
    MAX_SUMMARIES,
    PER_GROUP,
    SynthesisRequest,
    select,
    select_with_timeline,
)
from axon_scouting.rag.retrieval.summaries import SummariesRetriever, search_summaries

from .corpus import filtros


def resumo(
    summary_id: str,
    year: int,
    text: str,
    *,
    kind: str = "linha_do_tempo",
    scope: str = "regional",
    bodies: tuple[str, ...] = ("assembleias", "atas"),
) -> SummaryEntry:
    return SummaryEntry(
        summary_id=summary_id,
        kind=kind,
        title=f"Linha do tempo · Assembleias · {scope} · {year}" if kind == "linha_do_tempo" else f"Ficha · {year}",
        text=text,
        year=year,
        meeting_date=date(year, 3, 1) if kind == "ficha" else None,
        doc_keys=(f"ata-{year}",),
        bodies=bodies,
        scope=scope,
        page=1 if kind == "ficha" else None,
        source_url=None,
    )


RESUMOS = [
    resumo(
        "linha-do-tempo-assembleias-regional#2016",
        2016,
        "eleição para Diretoria Regional: Jabs Costa (Diretor Presidente)",
    ),
    resumo(
        "linha-do-tempo-assembleias-regional#2019",
        2019,
        "eleição para Diretoria Regional: Alexandre Banchi (Diretor-Presidente)",
    ),
    resumo(
        "linha-do-tempo-assembleias-regional#2022",
        2022,
        "eleição para Diretoria Regional: Rodrigo Freitas (Diretor Presidente)",
    ),
    resumo(
        "linha-do-tempo-assembleias-regional#2010", 2010, "eleição para Diretoria Regional: Fulano (Diretor Presidente)"
    ),
    resumo("linha-do-tempo-assembleias-nacional#2019", 2019, "eleição para o CAN: Beltrano", scope="nacional"),
    resumo(
        "ata-2019#ficha~1",
        2019,
        "Reunião ordinária de 01/03/2019, Campinas/SP. 549 delegados credenciados",
        kind="ficha",
    ),
    resumo("ata-2019#ficha~2", 2019, "eleição: Alexandre Banchi (Diretor-Presidente) delegados", kind="ficha"),
    resumo("ata-2019#ficha~3", 2019, "eleição: suplentes da Comissão Fiscal delegados", kind="ficha"),
]
DESDE_2016 = SynthesisRequest(year_from=2016, year_until=2026, scopes=("regional",), bodies=())


def test_janela_de_anos_e_abrangencia() -> None:
    ids = [e.summary_id for e in select(RESUMOS, DESDE_2016, "Quem foram os diretores-presidentes eleitos desde 2016?")]
    assert "linha-do-tempo-assembleias-regional#2010" not in ids  # fora do período
    assert "linha-do-tempo-assembleias-nacional#2019" not in ids  # outra abrangência
    linhas = [i for i in ids if i.startswith("linha-do-tempo")]
    assert linhas == [f"linha-do-tempo-assembleias-regional#{ano}" for ano in (2016, 2019, 2022)]  # em ordem de ano
    assert ids.index(linhas[-1]) < min(ids.index(i) for i in ids if "#ficha" in i)  # linhas antes das fichas


def test_orgao_nomeado_em_camadas() -> None:
    """Os resumos do órgão; sem nenhum, os que falam dele; sem nenhum, todos."""
    conselho = resumo("ata-cc#ficha", 2019, "o conselho decidiu", kind="ficha", bodies=("conselho_consultivo", "atas"))
    todos = [*RESUMOS, conselho]
    do_conselho = SynthesisRequest(year_from=None, year_until=None, scopes=(), bodies=("conselho_consultivo",))
    assert [e.summary_id for e in select(todos, do_conselho, "o que o conselho decidiu")] == ["ata-cc#ficha"]
    da_diretoria = SynthesisRequest(year_from=None, year_until=None, scopes=(), bodies=("diretoria_regional",))
    achados = select(todos, da_diretoria, "eleição da diretoria regional")
    assert achados and all("Diretoria Regional" in e.text for e in achados)  # as atas da Assembleia que falam dela
    sem_ninguem = SynthesisRequest(year_from=None, year_until=None, scopes=(), bodies=("comissao_fiscal",))
    assert select(todos, sem_ninguem, "delegados credenciados")


def test_plural_casa_o_singular() -> None:
    """ "diretores-presidentes eleitos" acha "Diretor-Presidente" e "eleição" não pesa sozinha."""
    pedido = SynthesisRequest(year_from=None, year_until=None, scopes=(), bodies=())
    assert select(RESUMOS, pedido, "diretores presidentes")


def test_ano_unico_abre_uma_janela_para_a_frente() -> None:
    pedido = SynthesisRequest(year_from=2019, year_until=2019, scopes=("regional",), bodies=())
    anos = {e.year for e in select(RESUMOS, pedido, "eleição da diretoria regional em 2019")}
    assert anos <= set(range(2018, 2023)) and 2022 in anos


def test_no_maximo_duas_partes_por_ficha_e_onde_prefere_a_abertura() -> None:
    pedido = SynthesisRequest(year_from=2019, year_until=2019, scopes=("regional",), bodies=())
    fichas = [
        e.summary_id
        for e in select(RESUMOS, pedido, "onde foi a reunião dos delegados em 2019?")
        if "#ficha" in e.summary_id
    ]
    assert len(fichas) == PER_GROUP and fichas[0] == "ata-2019#ficha~1"


def test_teto_de_resumos() -> None:
    muitos = [resumo(f"ata-{i}#ficha", 2019, "delegados credenciados", kind="ficha") for i in range(30)]
    pedido = SynthesisRequest(year_from=None, year_until=None, scopes=(), bodies=())
    assert len(select(muitos, pedido, "delegados credenciados")) == MAX_SUMMARIES


def test_sem_termo_da_pergunta_nada_vem() -> None:
    pedido = SynthesisRequest(year_from=None, year_until=None, scopes=(), bodies=())
    assert select(RESUMOS, pedido, "uniforme da alcateia") == []


class Fonte:
    async def summaries(self) -> list[SummaryEntry]:
        return list(RESUMOS)


def buscar(raw: Mapping[str, Any]) -> list[Chunk]:
    return asyncio.run(SummariesRetriever(Fonte()).search("Quem foram os diretores-presidentes eleitos?", filters=raw))


def test_tool_so_responde_a_sintese() -> None:
    assert buscar(filtros()) == []
    pedido = {"year_from": 2016, "year_until": 2026, "scopes": ["regional"], "bodies": ["diretoria_regional"]}
    achados = buscar(filtros(synthesis=pedido))
    assert achados and [c.metadata[SUMMARY_RANK] for c in achados] == list(range(len(achados)))
    assert all(c.status == "vigente" and c.metadata["document_type"] == "ata" for c in achados)
    assert ordered(achados) == achados  # a ordem da tool sobrevive ao reranker


def test_tool_e_retriever() -> None:
    retriever = SummariesRetriever(Fonte())
    assert isinstance(retriever, Retriever)
    tool = search_summaries(retriever)
    assert isinstance(tool, NamedRetriever) and tool.name == "search_summaries"
    assert tool.collections == frozenset({"scouting"})


@pytest.mark.parametrize(
    "pedido",
    [
        "2016",
        {"year_from": "2016"},
        {"year_from": 2020, "year_until": 2016},
        {"scopes": ["estadual"]},
        {"bodies": ["Diretoria Regional"]},
    ],
)
def test_pedido_invalido(pedido: object) -> None:
    with pytest.raises(InvalidFilters) as erro:
        buscar(filtros(synthesis=pedido))
    assert erro.value.key in ("synthesis", "bodies")


def test_artefato_ida_e_volta(tmp_path: Path) -> None:
    write_summaries(tmp_path / "summaries.jsonl", RESUMOS)
    assert read_summaries(tmp_path / "summaries.jsonl") == RESUMOS
    assert asyncio.run(ArtifactSummaries(tmp_path).summaries()) == RESUMOS
    textos = summary_texts(tmp_path)
    assert textos["ata-2019#ficha~1"].startswith("[Ficha · 2019]\n")


def test_artefato_sem_resumos(tmp_path: Path) -> None:
    assert asyncio.run(ArtifactSummaries(tmp_path).summaries()) == []
    assert summary_texts(tmp_path) == {}


def test_resumo_invalido() -> None:
    with pytest.raises(ValueError):
        resumo("Linha Do Tempo/2019", 2019, "x")
    with pytest.raises(ValueError):
        resumo("ata#ficha", 2019, "x", kind="resumo")


# --- 106 G4: a fachada da linha do tempo ------------------------------------------------------------------------------


def test_sem_eventos_a_fachada_e_a_selecao_de_hoje() -> None:
    query = "Quem foram os diretores-presidentes eleitos?"
    assert select_with_timeline(RESUMOS, DESDE_2016, query, []) == select(RESUMOS, DESDE_2016, query)


def test_as_fichas_das_atas_da_linha_do_tempo_vao_a_frente() -> None:
    antiga = resumo("ata-2010#ficha~2", 2010, "eleição: Fulano (Diretor Presidente) mandato", kind="ficha")
    query = "Quem foram os diretores-presidentes eleitos?"
    picked = select_with_timeline([*RESUMOS, antiga], DESDE_2016, query, ["ata-2010"])
    assert picked[0] == antiga  # fora da janela de anos do pedido, mas é onde a linha do tempo achou o evento
    rest = [e for e in select([*RESUMOS, antiga], DESDE_2016, query) if e != antiga]
    assert picked[1:] == rest[: MAX_SUMMARIES - 1]
    assert EVENT_CARDS < MAX_SUMMARIES


class Grafo:
    def __init__(self) -> None:
        self.planos: list[object] = []

    async def aliases(self) -> dict[str, tuple[str, ...]]:
        return {}

    async def query(self, name: str, /, **args: object) -> list[object]:
        return []

    async def run(self, plan: object) -> tuple[object, ...]:
        from axon_scouting.rag.graph.api import EdgeRef, EventRef, Evidence, FrontGroup

        self.planos.append(plan)
        evento = Evidence(
            query="linha_do_tempo",
            target="tipo_orgao:diretoria_regional",
            doc_key="ata-2019",
            unit_id=None,
            chunk_ids=("ata-2019#p2",),
            role="registro",
            scope="regional",
            date=date(2019, 3, 1),
            edge=EdgeRef(type="DE", source="ata-2019#1@p2", origin="regra"),
            event=EventRef(type="eleicao", text="Eleição para Diretoria Regional", page=2),
        )
        return (FrontGroup(step=0, query="linha_do_tempo", entity=None, evidence=(evento,)),)


def test_a_tool_roda_a_linha_do_tempo_do_plano() -> None:
    grafo = Grafo()
    pedido = {"year_from": 2016, "year_until": 2026, "scopes": ["regional"], "bodies": []}
    plano = {
        "version": 1,
        "forma": "simples",
        "consultas": [
            {"q": "linha_do_tempo", "args": {"alvo": "tipo_orgao:diretoria_regional", "eventos": ["eleicao"]}}
        ],
    }
    achados = asyncio.run(
        SummariesRetriever(Fonte(), graph=grafo).search(  # type: ignore[arg-type]
            "Quem foram os diretores-presidentes eleitos?", filters=filtros(synthesis=pedido, plan=plano)
        )
    )
    assert len(grafo.planos) == 1
    assert achados[0].chunk_id.startswith("ata-2019#ficha")  # a ficha da ata do evento, à frente
    # Sem linha do tempo no plano, o grafo não é consultado.
    asyncio.run(SummariesRetriever(Fonte(), graph=grafo).search("x", filters=filtros(synthesis=pedido)))  # type: ignore[arg-type]
    assert len(grafo.planos) == 1
