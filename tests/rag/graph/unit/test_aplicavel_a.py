"""`aplicavel_a`, a instituição real no `sobre` e a interseção do executor (spec 106 G3, US3; research R20)."""

from __future__ import annotations

from itertools import pairwise
from typing import Any

from axon_scouting.rag.graph.api import QueryPlan, QueryStep
from axon_scouting.rag.graph.domain.executor import executar
from axon_scouting.rag.graph.domain.model import from_records
from axon_scouting.rag.graph.domain.queries import aplicavel_a, sobre

CHAIN = ("inst:n1", "inst:n2", "inst:n3", "inst:n4", "inst:n5", "inst:n6", "inst:n7")
"""Uma cadeia de 7 instituições, n1 embaixo: `aplicavel_a` sobe só 5 níveis."""


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

    def version(doc_key: str, scope: str, role: str = "norma") -> dict[str, Any]:
        return {
            "kind": "version",
            "doc_key": doc_key,
            "document_id": doc_key,
            "type": "estatuto",
            "role": role,
            "status": "vigente",
            "scope": scope,
        }

    def unit(unit_id: str) -> dict[str, Any]:
        return {
            "kind": "unit",
            "unit_id": unit_id,
            "doc_key": unit_id.split("#")[0],
            "unit_kind": "secao",
            "title": unit_id,
            "chunk_ids": [f"{unit_id}@c"],
        }

    def edge(kind: str, from_: str, to: str) -> dict[str, Any]:
        return {"kind": "edge", "type": kind, "from": from_, "to": to, "source": "x", "origin": "curado"}

    return [
        {"kind": "meta", "version": 2, "chunking": "x"},
        *(entity(e) for e in ("tipo_inst:ueb", "tipo_inst:regiao", "tipo_inst:grupo", "tipo_inst:outro_grupo")),
        *(entity(e) for e in ("tipo_orgao:assembleia_de_grupo", "tipo_orgao:diretoria_regional")),
        *(entity(e) for e in ("inst:ueb", "inst:regiao_sp", "inst:ge", "inst:outro_ge", *CHAIN)),
        version("estatuto", "nacional"),
        version("regulamento-sp", "regional"),
        version("estatuto-ge", "local"),
        version("estatuto-outro-ge", "local"),
        version("ata-sp", "regional", role="registro"),
        *(version(f"doc-{i}", "nacional") for i in CHAIN),
        unit("estatuto#grupo"),
        unit("estatuto#assembleia-de-grupo"),
        unit("estatuto#diretoria-regional"),
        unit("estatuto-ge#documento"),
        unit("estatuto-outro-ge#documento"),
        edge("SOBRE", "estatuto#grupo", "tipo_inst:grupo"),
        edge("SOBRE", "estatuto#assembleia-de-grupo", "tipo_orgao:assembleia_de_grupo"),
        edge("SOBRE", "estatuto#diretoria-regional", "tipo_orgao:diretoria_regional"),
        edge("SOBRE", "estatuto-ge#documento", "inst:ge"),
        edge("SOBRE", "estatuto-outro-ge#documento", "inst:outro_ge"),
        edge("INSTANCIA_DE", "inst:ge", "tipo_inst:grupo"),
        edge("INSTANCIA_DE", "inst:outro_ge", "tipo_inst:grupo"),
        edge("INSTANCIA_DE", "inst:regiao_sp", "tipo_inst:regiao"),
        edge("INSTANCIA_DE", "inst:ueb", "tipo_inst:ueb"),
        edge("PARTE_DE", "tipo_orgao:assembleia_de_grupo", "tipo_inst:grupo"),
        edge("PARTE_DE", "tipo_orgao:diretoria_regional", "tipo_inst:regiao"),
        edge("PARTE_DE", "inst:ge", "inst:regiao_sp"),
        edge("PARTE_DE", "inst:outro_ge", "inst:regiao_sp"),
        edge("PARTE_DE", "inst:regiao_sp", "inst:ueb"),
        *(edge("PARTE_DE", low, high) for low, high in pairwise(CHAIN)),
        edge("APLICA_SE_A", "estatuto", "inst:ueb"),
        edge("APLICA_SE_A", "regulamento-sp", "inst:regiao_sp"),
        edge("APLICA_SE_A", "ata-sp", "inst:regiao_sp"),
        edge("APLICA_SE_A", "estatuto-ge", "inst:ge"),
        edge("APLICA_SE_A", "estatuto-outro-ge", "inst:outro_ge"),
        *(edge("APLICA_SE_A", f"doc-{i}", i) for i in CHAIN),
    ]


G = from_records(_records())


def test_vale_para_o_ge_o_que_vale_para_ele_e_para_as_de_cima_de_cima_para_baixo() -> None:
    assert [e.doc_key for e in aplicavel_a(G, "inst:ge")] == ["estatuto", "regulamento-sp", "estatuto-ge"]


def test_o_outro_grupo_nao_entra() -> None:
    assert "estatuto-outro-ge" not in {e.doc_key for e in aplicavel_a(G, "inst:ge")}


def test_so_as_normas_por_padrao() -> None:
    assert "ata-sp" not in {e.doc_key for e in aplicavel_a(G, "inst:regiao_sp")}
    assert "ata-sp" in {e.doc_key for e in aplicavel_a(G, "inst:regiao_sp", papeis=("registro",))}


def test_sobe_ate_5_niveis() -> None:
    assert {e.doc_key for e in aplicavel_a(G, "inst:n1")} == {f"doc-{i}" for i in CHAIN[:6]}


def test_instituicao_fora_do_grafo_nao_tem_documentos() -> None:
    assert aplicavel_a(G, "inst:inexistente") == []


def test_a_instituicao_real_traz_o_documento_dela_o_tipo_e_as_partes_do_tipo() -> None:
    units = {e.unit_id for e in sobre(G, ["inst:ge"])}
    assert units == {"estatuto-ge#documento", "estatuto#grupo", "estatuto#assembleia-de-grupo"}


def test_o_tipo_citado_fica_so_com_as_unidades_dele() -> None:
    """A menção do tipo ("grupo escoteiro") não traz as partes: o comportamento do G1."""
    assert [e.unit_id for e in sobre(G, ["tipo_inst:grupo"])] == ["estatuto#grupo"]


def test_a_intersecao_corta_o_que_nao_vale_para_a_instituicao() -> None:
    plan = QueryPlan(
        forma="intersecao",
        consultas=(
            QueryStep(q="sobre", args={"entidades": ["tipo_inst:grupo", "inst:outro_ge"]}),
            QueryStep(q="aplicavel_a", args={"instituicao": "inst:ge"}),
        ),
    )
    (group,) = executar(G, plan)
    assert group.query == "sobre"
    assert {e.doc_key for e in group.evidence} == {"estatuto"}  # o documento do outro grupo saiu
