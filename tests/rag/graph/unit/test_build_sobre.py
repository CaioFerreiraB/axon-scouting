"""As regras do `SOBRE` de programa e adultos (spec 106 G2, T036; research R6)."""

from __future__ import annotations

from typing import Any

from axon_scouting.rag.graph.domain.build.sobre import rule_headings, sobre_graph

ALIASES = {
    "ramo:escoteiro": ("ramo escoteiro",),
    "conceito:patrulha": ("patrulha", "patrulhas"),
    "conceito:lema": ("lema",),
    "cargo:escotista": ("escotista", "escotistas"),
}


def _chunk(chunk_id: str, doc_key: str, path: list[tuple[str, str, str]], text: str = "") -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "doc_key": doc_key,
        "text": text,
        "path": [{"unit_id": u, "kind": k, "title": t} for u, k, t in path],
    }


CAP = ("por#u/cap10", "capitulo", "Capítulo 10 – Ramo Escoteiro")
CHUNKS = [
    _chunk(
        "por#r80",
        "por",
        [CAP, ("por#u/cap10/r80", "regra", "Regra 080")],
        "REGRA 080 PATRULHA NO RAMO ESCOTEIRO\nI - ...",
    ),
    _chunk("por#r76", "por", [CAP, ("por#u/cap10/r76", "regra", "Regra 076")], "REGRA 076 LEMA DO RAMO ESCOTEIRO\n..."),
    _chunk("guia#s1", "guia", [("guia#u/s1", "secao", "O lema")]),
    _chunk("velho#s1", "velho", [("velho#u/s1", "secao", "Escotistas")]),
    _chunk("ata#s1", "ata", [("ata#u/s1", "item_de_pauta", "Escotistas")]),
]
VERSIONS = {
    "por": {"role": "norma", "status": "vigente"},
    "guia": {"role": "orientacao", "status": "vigente"},
    "velho": {"role": "norma", "status": "revogado"},
    "ata": {"role": "registro", "status": "vigente"},
}


def test_o_titulo_da_regra_vem_do_texto() -> None:
    assert rule_headings(CHUNKS) == {
        "por#u/cap10/r80": "PATRULHA NO RAMO ESCOTEIRO",
        "por#u/cap10/r76": "LEMA DO RAMO ESCOTEIRO",
    }


def test_a_unidade_mais_funda_que_nomeia_a_entidade() -> None:
    graph = sobre_graph(CHUNKS, VERSIONS, ALIASES, {"guia": "escoteiro"})
    edges = set(graph.edges)
    assert ("por#u/cap10/r80", "conceito:patrulha") in edges
    assert ("por#u/cap10/r80", "ramo:escoteiro") in edges  # a regra nomeia o ramo: ela, e não o capítulo
    assert ("por#u/cap10/r76", "conceito:lema") in edges
    assert ("guia#documento", "ramo:escoteiro") in edges  # o documento de um ramo é sobre o ramo
    assert ("guia#u/s1", "conceito:lema") in edges
    assert graph.units["guia#u/s1"]["ancestors"] == ["guia#documento"]
    assert graph.units["por#u/cap10/r80"]["ancestors"] == ["por#u/cap10"]


def test_so_normas_e_orientacoes_vigentes() -> None:
    edges = set(sobre_graph(CHUNKS, VERSIONS, ALIASES, {}).edges)
    assert not any(unit.startswith(("velho", "ata")) for unit, _ in edges)
