"""`ligados` com as ligações entre documentos e `documentos` por emissor (spec 106 G5, US5; research R22)."""

from __future__ import annotations

from typing import Any

from axon_scouting.rag.graph.api import QueryPlan, QueryStep
from axon_scouting.rag.graph.domain.executor import executar
from axon_scouting.rag.graph.domain.model import from_records
from axon_scouting.rag.graph.domain.queries import documentos, ligados

RULE = "por-2025#u/regra-183"
SEM_ASSUNTO = "sp-2024-07-1"


def _records() -> list[dict[str, Any]]:
    def entity(id_: str) -> dict[str, Any]:
        kind = id_.split(":")[0]
        return {"kind": "entity", "id": id_, "type": kind, "name": id_, "terms": [], "source": "x", "origin": "curado"}

    def version(doc_key: str, document_id: str, scope: str, status: str = "vigente", **extra: Any) -> dict[str, Any]:
        return {
            "kind": "version",
            "doc_key": doc_key,
            "document_id": document_id,
            "type": extra.pop("type", "resolucao"),
            "role": extra.pop("role", "norma"),
            "status": status,
            "scope": scope,
            "chunk_ids": [f"{doc_key}#preambulo@p1", f"{doc_key}#art1@p1"],
            **extra,
        }

    def unit(unit_id: str, doc_key: str) -> dict[str, Any]:
        return {
            "kind": "unit",
            "unit_id": unit_id,
            "doc_key": doc_key,
            "unit_kind": "regra",
            "title": unit_id,
            "chunk_ids": [f"{unit_id}@c"],
        }

    def edge(kind: str, from_: str, to: str, source: str = "x") -> dict[str, Any]:
        return {"kind": "edge", "type": kind, "from": from_, "to": to, "source": source, "origin": "regra"}

    return [
        {"kind": "meta", "version": 2, "chunking": "x"},
        *(
            entity(e)
            for e in (
                "tipo_orgao:diretoria_regional",
                "orgao:dr_sp",
                "inst:regiao_sp",
                "conceito:distintivo",
                "conceito:formacao",
                "doc:can-2021-11",
            )
        ),
        version("por-2025", "por", "nacional", type="por"),
        version("can-2021-11-1", "can-2021-11", "nacional"),
        version("can-2019-09-1", "can-2019-09", "nacional", status="revogado"),
        version("can-2022-10-1", "can-2022-10", "nacional"),
        version("in-2024-02-1", "in-2024-02", "regional"),
        version(SEM_ASSUNTO, "sp-2024-07", "regional"),
        version("oficio-dr-1", "oficio-dr", "regional", type="oficio", role="comunicacao"),
        unit(RULE, "por-2025"),
        unit("in-2024-02-1#documento", "in-2024-02-1"),
        edge("CONTEM", "por-2025", RULE),
        edge("CONTEM", "in-2024-02-1", "in-2024-02-1#documento"),
        edge("SOBRE", RULE, "conceito:distintivo"),
        edge("SOBRE", "in-2024-02-1#documento", "conceito:formacao"),
        edge("ALTERA", "can-2021-11-1", RULE, "can-2021-11-1#art1@p1"),
        edge("REVOGA", "can-2022-10-1", "can-2019-09-1", "can-2022-10-1#art5@p1"),
        edge("INSTANCIA_DE", "orgao:dr_sp", "tipo_orgao:diretoria_regional"),
        edge("PARTE_DE", "orgao:dr_sp", "inst:regiao_sp"),
        *(edge("EMITIDO_POR", d, "orgao:dr_sp") for d in ("in-2024-02-1", SEM_ASSUNTO, "oficio-dr-1")),
    ]


G = from_records(_records())


def test_ligados_da_regra_traz_a_resolucao_com_o_trecho_que_altera_primeiro() -> None:
    (found,) = ligados(G, RULE, relacoes=["ALTERA", "REVOGA"])
    assert (found.target, found.doc_key, found.unit_id, found.edge.type) == (
        "can-2021-11-1",
        "can-2021-11-1",
        None,
        "ALTERA",
    )
    assert found.chunk_ids == ("can-2021-11-1#art1@p1", "can-2021-11-1#preambulo@p1")


def test_ligados_do_documento_pelo_numero_traz_a_regra_alterada() -> None:
    (found,) = ligados(G, "doc:can-2021-11")
    assert (found.doc_key, found.unit_id, found.chunk_ids) == ("por-2025", RULE, (f"{RULE}@c",))


def test_ligados_de_uma_entidade_vai_pelas_unidades_sobre_ela() -> None:
    assert [e.doc_key for e in ligados(G, "conceito:distintivo", relacoes=["ALTERA"])] == ["can-2021-11-1"]


def test_ligados_dos_dois_lados_da_revogacao() -> None:
    assert [e.target for e in ligados(G, "can-2019-09-1")] == ["can-2022-10-1"]
    assert [e.target for e in ligados(G, "can-2022-10-1")] == ["can-2019-09-1"]


def test_documentos_pelo_tipo_de_orgao_e_pela_instituicao() -> None:
    by_type = documentos(G, emissor="tipo_orgao:diretoria_regional")
    assert [e.doc_key for e in by_type] == ["in-2024-02-1", SEM_ASSUNTO, "oficio-dr-1"]  # a norma, depois o ofício
    assert by_type[0].chunk_ids == ("in-2024-02-1#preambulo@p1", "in-2024-02-1#art1@p1") and by_type[0].unit_id is None
    assert [e.doc_key for e in documentos(G, emissor="inst:regiao_sp", tipo="oficio")] == ["oficio-dr-1"]
    assert [e.doc_key for e in documentos(G, tipo="resolucao", vigente=False) if e.doc_key.startswith("can-2019")] == [
        "can-2019-09-1"
    ]
    assert not [e for e in documentos(G, tipo="resolucao") if e.doc_key == "can-2019-09-1"]  # revogada fica fora


def test_o_que_o_orgao_publicou_sobre_o_assunto() -> None:
    plano = QueryPlan(
        forma="intersecao",
        consultas=(
            QueryStep(q="sobre", args={"entidades": ["conceito:formacao"]}),
            QueryStep(q="documentos", args={"emissor": "tipo_orgao:diretoria_regional"}),
        ),
    )
    (grupo,) = executar(G, plano)
    assert [e.unit_id for e in grupo.evidence] == ["in-2024-02-1#documento"]
