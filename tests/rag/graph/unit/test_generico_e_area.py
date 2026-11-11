"""O termo genérico (`E_UM`) e o salto pela área (`REGE` + `DE_AREA`) no `sobre` (spec 106 G2; research R19)."""

from __future__ import annotations

from typing import Any

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

    def version(doc_key: str) -> dict[str, Any]:
        return {
            "kind": "version",
            "doc_key": doc_key,
            "document_id": doc_key,
            "type": "politica",
            "role": "norma",
            "status": "vigente",
            "scope": "nacional",
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
        return {"kind": "edge", "type": kind, "from": from_, "to": to, "source": "x", "origin": "regra"}

    return [
        {"kind": "meta", "version": 2, "chunking": "x"},
        *(entity(e) for e in ("conceito:adulto", "cargo:escotista", "cargo:dirigente", "area:espacos_seguros")),
        *(version(d) for d in ("por", "seguros")),
        unit("por#r18"),
        unit("por#r123"),
        unit("seguros#documento"),
        edge("SOBRE", "por#r18", "cargo:escotista"),
        edge("SOBRE", "por#r123", "conceito:adulto"),
        edge("E_UM", "cargo:escotista", "conceito:adulto"),
        edge("E_UM", "cargo:dirigente", "conceito:adulto"),
        edge("REGE", "area:espacos_seguros", "conceito:adulto"),
        edge("DE_AREA", "seguros", "area:espacos_seguros"),
    ]


G = from_records(_records())


def test_o_que_vale_para_os_adultos_vale_para_o_escotista() -> None:
    units = [e.unit_id for e in sobre(G, ["cargo:escotista"])]
    assert units[0] == "por#r18"  # as diretas primeiro
    assert "por#r123" in units  # o termo genérico depois
    assert units[-1] == "seguros#documento"  # e, por último, o documento da área que rege os adultos


def test_a_unidade_sobre_escotistas_trata_de_adultos() -> None:
    assert [e.unit_id for e in sobre(G, ["conceito:adulto", "cargo:escotista"], modo="todas")] == ["por#r18"]


def test_o_salto_pela_area_nao_entra_no_todas() -> None:
    units = [e.unit_id for e in sobre(G, ["conceito:adulto", "cargo:dirigente"], modo="todas")]
    assert "seguros#documento" not in units
