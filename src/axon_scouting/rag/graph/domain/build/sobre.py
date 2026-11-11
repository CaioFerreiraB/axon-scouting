"""A aresta `SOBRE` para as entidades de programa e adultos (spec 106 G2, FR-006; research R6).

Por regra, sem modelo, nos documentos vigentes de papel `norma` e `orientacao`:
1. o título de uma unidade nomeia a entidade (o nome mais longo, pelo `reconhecer`): a unidade mais funda que a nomeia,
   em cada caminho, leva a aresta; nas regras do POR, o título é o cabeçalho do texto ("REGRA 080 PATRULHA NO RAMO
   ESCOTEIRO"), porque a árvore só guarda "Regra 080";
2. o documento de um ramo (o `branch` do catálogo, como os guias e os manuais dos ramos) é, inteiro, sobre o ramo.
Os órgãos seguem a regra do ADR-063 (`organs.py`), que não muda. Puro.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from axon_scouting.rag.graph.api import reconhecer
from axon_scouting.rag.graph.domain.build.organs import normalize

ROLES_WITH_SOBRE: Final = ("norma", "orientacao")
RULE_HEADING: Final = re.compile(r"REGRA\s+(\d{3})\s+([^\n|]+)")


@dataclass(slots=True)
class SobreGraph:
    units: dict[str, dict[str, Any]] = field(default_factory=dict)
    """unit_id → {doc_key, unit_kind, title, chunk_ids, ancestors}."""
    edges: list[tuple[str, str]] = field(default_factory=list)
    """(unit_id, entidade), sem repetição, na ordem do acervo."""


def rule_headings(chunks: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """`unit_id` de cada regra do POR → o cabeçalho dela no texto ("PATRULHA NO RAMO ESCOTEIRO")."""
    found: dict[str, str] = {}
    for chunk in chunks:
        path = chunk.get("path") or []
        if not path or path[-1]["kind"] != "regra" or path[-1]["unit_id"] in found:
            continue
        number = re.search(r"(\d{3})", path[-1].get("title") or "")
        for match in RULE_HEADING.finditer(chunk["text"]):
            if number is None or match.group(1) == number.group(1):
                found[path[-1]["unit_id"]] = match.group(2).strip()
                break
    return found


def sobre_graph(
    chunks: Sequence[Mapping[str, Any]],
    versions: Mapping[str, Mapping[str, Any]],
    aliases: Mapping[str, tuple[str, ...]],
    branch_of: Mapping[str, str],
) -> SobreGraph:
    """`versions`: doc_key → {role, status}; `branch_of`: doc_key → ramo do documento inteiro (catálogo)."""
    headings = rule_headings(chunks)
    eligible = [
        c
        for c in chunks
        if (v := versions.get(c["doc_key"])) and v["role"] in ROLES_WITH_SOBRE and v["status"] == "vigente"
    ]
    below: dict[str, list[str]] = {}
    kinds: dict[str, tuple[str, str, str]] = {}
    for chunk in eligible:
        for unit in chunk.get("path") or []:
            below.setdefault(unit["unit_id"], []).append(chunk["chunk_id"])
            kinds[unit["unit_id"]] = (chunk["doc_key"], unit["kind"], unit.get("title") or "")
        below.setdefault(f"{chunk['doc_key']}#documento", []).append(chunk["chunk_id"])
    result = SobreGraph()
    seen: set[tuple[str, str]] = set()

    def add(unit_id: str, entity: str, ancestors: list[str]) -> None:
        if (unit_id, entity) in seen:
            return
        seen.add((unit_id, entity))
        result.edges.append((unit_id, entity))
        if unit_id not in result.units:
            doc_key, kind, title = kinds.get(unit_id, (unit_id.split("#", 1)[0], "documento", ""))
            result.units[unit_id] = {
                "doc_key": doc_key,
                "unit_kind": kind,
                "title": headings.get(unit_id, title) if kind == "regra" else title,
                "chunk_ids": list(dict.fromkeys(below.get(unit_id, []))),
                "ancestors": ancestors,
            }

    for chunk in eligible:
        doc_unit = f"{chunk['doc_key']}#documento"
        branch = branch_of.get(chunk["doc_key"])
        if branch and f"ramo:{branch}" in aliases:
            add(doc_unit, f"ramo:{branch}", [])
        path = chunk.get("path") or []
        assigned: set[str] = set()
        for depth in range(len(path) - 1, -1, -1):
            unit = path[depth]
            title = (
                headings.get(unit["unit_id"], unit.get("title") or "")
                if unit["kind"] == "regra"
                else unit.get("title") or ""
            )
            for entity in reconhecer(normalize(title), aliases):
                if entity not in assigned:
                    assigned.add(entity)
                    ancestors = [p["unit_id"] for p in path[:depth]] + ([doc_unit] if branch else [])
                    add(unit["unit_id"], entity, ancestors)
    return result
