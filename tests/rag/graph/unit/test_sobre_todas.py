"""`sobre` com `modo="todas"`, a seção pelo ramo e as formas do executor (spec 106 G2, T038)."""

from __future__ import annotations

from typing import Any

from axon_scouting.rag.graph.api import QueryPlan, QueryStep
from axon_scouting.rag.graph.domain.executor import executar
from axon_scouting.rag.graph.domain.model import from_records
from axon_scouting.rag.graph.domain.queries import sobre


def _records() -> list[dict[str, Any]]:
    def entity(id_: str) -> dict[str, Any]:
        return {
            "kind": "entity",
            "id": id_,
            "type": id_.split(":")[0],
            "name": id_,
            "terms": [],
            "source": "x",
            "origin": "curado",
        }

    def unit(unit_id: str, ancestors: list[str]) -> dict[str, Any]:
        return {
            "kind": "unit",
            "unit_id": unit_id,
            "doc_key": unit_id.split("#")[0],
            "unit_kind": "secao",
            "title": unit_id,
            "chunk_ids": [f"{unit_id}@c"],
            "ancestors": ancestors,
        }

    def sobre_(unit_id: str, to: str) -> dict[str, Any]:
        return {"kind": "edge", "type": "SOBRE", "from": unit_id, "to": to, "source": unit_id, "origin": "regra"}

    return [
        {"kind": "meta", "version": 2, "chunking": "x"},
        *(entity(e) for e in ("ramo:lobinho", "ramo:senior", "secao:alcateia", "conceito:lema", "conceito:matilha")),
        {
            "kind": "version",
            "doc_key": "por",
            "document_id": "por",
            "type": "por",
            "role": "norma",
            "status": "vigente",
            "scope": "nacional",
        },
        {
            "kind": "version",
            "doc_key": "manual",
            "document_id": "manual",
            "type": "manual",
            "role": "orientacao",
            "status": "vigente",
            "scope": "nacional",
        },
        unit("por#cap9", []),
        unit("por#r64", ["por#cap9"]),
        unit("por#r68", ["por#cap9"]),
        unit("por#cap11", []),
        unit("manual#documento", []),
        unit("manual#lema", ["manual#documento"]),
        unit("manual#lema-lobinho", ["manual#documento"]),
        sobre_("por#cap9", "ramo:lobinho"),
        sobre_("por#r64", "conceito:lema"),
        sobre_("por#r68", "conceito:matilha"),
        sobre_("por#cap11", "ramo:senior"),
        sobre_("manual#documento", "ramo:lobinho"),
        sobre_("manual#lema", "conceito:lema"),
        sobre_("manual#lema-lobinho", "conceito:lema"),
        sobre_("manual#lema-lobinho", "secao:alcateia"),
        {
            "kind": "edge",
            "type": "DE_RAMO",
            "from": "secao:alcateia",
            "to": "ramo:lobinho",
            "source": "x",
            "origin": "curado",
        },
    ]


G = from_records(_records())


def test_todas_pelo_caminho_e_a_secao_vale_pelo_ramo() -> None:
    units = [e.unit_id for e in sobre(G, ["ramo:lobinho", "conceito:lema"], modo="todas")]
    assert "por#r64" in units  # a regra do lema, dentro do capítulo do Ramo Lobinho
    assert "manual#lema-lobinho" in units  # a alcateia vale pelo Ramo Lobinho
    assert "manual#lema" not in units  # só pelo documento inteiro, e há uma unidade mais forte no mesmo documento
    assert "por#r68" not in units


def test_a_secao_entra_no_ramo() -> None:
    assert "manual#lema-lobinho" in [e.unit_id for e in sobre(G, ["ramo:lobinho"])]


def test_para_cada_um_grupo_por_entidade() -> None:
    plano = QueryPlan(
        forma="para_cada",
        para_cada=("ramo:lobinho", "ramo:senior"),
        consultas=(QueryStep(q="sobre", args={"entidades": ["$cada", "conceito:lema"], "modo": "todas"}),),
    )
    lobinho, senior = executar(G, plano)
    assert lobinho.entity == "ramo:lobinho" and {e.unit_id for e in lobinho.evidence} >= {"por#r64"}
    # sem unidade do Sênior com o lema, cai para o ramo sozinho
    assert senior.entity == "ramo:senior" and [e.unit_id for e in senior.evidence] == ["por#cap11"]


def test_intersecao_filtra_pelos_documentos_das_seguintes() -> None:
    plano = QueryPlan(
        forma="intersecao",
        consultas=(
            QueryStep(q="sobre", args={"entidades": ["conceito:lema"]}),
            QueryStep(q="sobre", args={"entidades": ["conceito:matilha"]}),
        ),
    )
    (grupo,) = executar(G, plano)
    assert {e.doc_key for e in grupo.evidence} == {"por"}
