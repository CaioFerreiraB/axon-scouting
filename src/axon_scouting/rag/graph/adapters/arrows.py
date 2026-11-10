"""O grafo de domínio no formato do arrows.app (`nodes`, `relationships`, `style`), o mesmo de
`corpus/Organização Escoteira.revisado.json`, para o mantenedor ver o grafo real onde desenhou o planejado (spec 106
FR-018, research R16)."""

from __future__ import annotations

from typing import Any, Final

from axon_scouting.rag.graph.domain.model import DomainGraph

COLORS: Final = {
    "tipo_inst": "#4C8EDA",
    "tipo_orgao": "#4C8EDA",
    "inst": "#57C7E3",
    "orgao": "#57C7E3",
    "cargo": "#F79767",
    "ramo": "#8DCC93",
    "secao": "#8DCC93",
    "area": "#C990C0",
    "conceito": "#C990C0",
    "doc": "#D9C8AE",
    "version": "#D9C8AE",
    "unit": "#ECB5C9",
}
STYLE: Final = {"font-family": "Nunito Sans", "background-color": "#F2F2F2", "node-color": "#4C8EDA", "radius": 75}


def neighbourhood(graph: DomainGraph, entity: str | None, hops: int, *, units: bool = True) -> set[str]:
    """Sem entidade, só as entidades (as arestas entre elas cabem no editor); com ela, o entorno até `hops` arestas."""
    if entity is None:
        return set(graph.entities)
    keep = {entity}
    for _ in range(hops):
        keep |= {e.to for e in graph.edges if e.from_ in keep} | {e.from_ for e in graph.edges if e.to in keep}
    return keep if units else {node for node in keep if node in graph.entities}


def to_arrows(graph: DomainGraph, keep: set[str]) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    ids: dict[str, str] = {}
    for index, node in enumerate(sorted(keep)):
        ids[node] = f"n{index}"
        if node in graph.entities:
            entity = graph.entities[node]
            caption, label = entity.name, entity.type
            props = {"id": node, "fonte": entity.source, "origem": entity.origin}
        elif node in graph.units:
            unit = graph.units[node]
            caption, label = unit.title, "unit"
            props = {"id": node, "documento": unit.doc_key, "trechos": str(len(unit.chunk_ids))}
        else:
            version = graph.versions.get(node)
            caption, label = node, "version"
            props = {"id": node, "papel": version.role if version else "", "tipo": version.type if version else ""}
        nodes.append(
            {
                "id": ids[node],
                "position": {"x": (index % 12) * 220, "y": (index // 12) * 180},
                "caption": caption,
                "labels": [label],
                "properties": props,
                "style": {"node-color": COLORS.get(label, "#A5ABB6")},
            }
        )
    relationships = [
        {
            "id": f"r{index}",
            "fromId": ids[edge.from_],
            "toId": ids[edge.to],
            "type": edge.type if edge.mode is None else f"{edge.type} ({edge.mode})",
            "properties": {"fonte": edge.source, "origem": edge.origin},
            "style": {},
        }
        for index, edge in enumerate(e for e in graph.edges if e.from_ in ids and e.to in ids)
    ]
    return {"style": dict(STYLE), "nodes": nodes, "relationships": relationships}
