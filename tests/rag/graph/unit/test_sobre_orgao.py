"""`sobre` restrito a órgão e o executor no G1 (spec 106 T013): a ordem e os cortes do ADR-063."""

from __future__ import annotations

from axon_scouting.rag.graph.api import QueryPlan, QueryStep
from axon_scouting.rag.graph.domain.executor import executar
from axon_scouting.rag.graph.domain.model import from_records
from axon_scouting.rag.graph.domain.queries import ligados, sobre

from ..grafo_sintetico import registros

G = from_records(registros())


def test_a_ordem_de_cima_para_baixo_por_documento() -> None:
    docs = [e.doc_key for e in sobre(G, ["tipo_orgao:diretoria_regional"])]
    assert docs == ["mundial-2020", "estatuto-2024", "regulamento-sp-2022", "local-2026"]


def test_a_ordem_da_mencao_vem_primeiro() -> None:
    docs = [
        (e.target, e.doc_key) for e in sobre(G, ["tipo_orgao:assembleia_regional", "tipo_orgao:diretoria_regional"])
    ]
    assert docs[0] == ("tipo_orgao:assembleia_regional", "estatuto-2024")
    assert docs[1][0] == "tipo_orgao:diretoria_regional"


def test_o_executor_entrega_todos_os_documentos_na_ordem() -> None:
    """O corte em 3 documentos é da 103, que os ordena também pela relevância à pergunta (106 R19)."""
    plano = QueryPlan(
        consultas=(
            QueryStep(
                q="sobre",
                args={
                    "entidades": ["tipo_orgao:assembleia_regional", "tipo_orgao:diretoria_regional"],
                    "papeis": ["norma"],
                },
            ),
        )
    )
    (grupo,) = executar(G, plano)
    assert [e.doc_key for e in grupo.evidence] == [
        "estatuto-2024",
        "mundial-2020",
        "estatuto-2024",
        "regulamento-sp-2022",
        "local-2026",
    ]


def test_papeis_filtram() -> None:
    assert sobre(G, ["tipo_orgao:diretoria_regional"], papeis=["orientacao"]) == []


def test_governanca_so_em_ligados() -> None:
    plano = QueryPlan(consultas=(QueryStep(q="ligados", args={"alvo": "tipo_orgao:diretoria_regional"}),))
    assert executar(G, plano)[0].evidence == ()
    (relacao,) = ligados(G, "tipo_orgao:diretoria_regional")
    assert relacao.edge.type == "ELEGE" and relacao.chunk_ids == ("estatuto-2024#a24@p17",)


def test_entidade_inexistente_volta_vazia() -> None:
    assert sobre(G, ["tipo_orgao:nao_existe"]) == []
    assert sobre(G, ["tipo_orgao:orfao"]) == []
