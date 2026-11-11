"""`DomainGraph`: o grafo de domínio em memória, montado uma vez a partir dos registros do `graph.jsonl` v2 (spec 106,
research R10–R11; contracts/graph-jsonl-v2.md). Dicionários imutáveis depois da carga; puro: sem I/O.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date
from types import MappingProxyType
from typing import Any, Final, cast

from axon_scouting.rag.graph.api import ORIGINS, EventType, Origin, Role

GRAPH_VERSION: Final = 2
GOVERNANCE: Final = frozenset({"ELEGE", "COMPOE", "INTEGRA", "CONVOCA", "FISCALIZA", "APRECIA_RECURSO_DE", "DIRIGE"})
"""As relações de governança: no grafo, com a fonte, e fora da busca automática (FR-007, decisão 4 do mantenedor)."""
LINKS: Final = ("ALTERA", "REVOGA", "SUBSTITUI", "CITA")
"""As ligações entre documentos e unidades (G5): só `ligados` as segue."""

Record = Mapping[str, Any]


@dataclass(frozen=True, slots=True, kw_only=True)
class EntityRec:
    id: str
    type: str
    name: str
    terms: tuple[str, ...]
    aliases: tuple[str, ...]
    source: str
    origin: Origin
    level: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class VersionRec:
    doc_key: str
    document_id: str
    type: str
    role: Role
    status: str
    scope: str | None = None
    date: date | None = None
    bodies: tuple[str, ...] = ()
    chunk_ids: tuple[str, ...] = ()
    """Os trechos da versão (G5): o que `documentos` e `ligados` entregam quando a evidência é o documento inteiro."""
    ementa: str | None = None
    """A ementa da resolução, lida por regra na primeira página (G5)."""


@dataclass(frozen=True, slots=True, kw_only=True)
class UnitRec:
    unit_id: str
    doc_key: str
    unit_kind: str
    title: str
    chunk_ids: tuple[str, ...]
    branches: tuple[str, ...] = ()
    parent: str | None = None
    ancestors: tuple[str, ...] = ()
    """As unidades acima desta (da raiz para baixo) e, no documento de um ramo, o documento inteiro (G2, `todas`)."""


@dataclass(frozen=True, slots=True, kw_only=True)
class EventRec:
    """Um item conferido da ficha de uma ata (G4): a reunião, uma deliberação ou uma eleição, com a página e os
    trechos da ata nela."""

    id: str
    doc_key: str
    date: date | None
    type: EventType
    text: str
    page: int | None
    chunk_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class EdgeRec:
    type: str
    from_: str
    to: str
    source: str
    origin: Origin
    mode: str | None = None
    checked: bool = False
    attrs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class DomainGraph:
    """As entidades na ordem do catálogo (ela desempata o reconhecimento), as versões, as unidades, as arestas na ordem
    do arquivo e os índices por ponta."""

    meta: Mapping[str, Any]
    entities: Mapping[str, EntityRec]
    versions: Mapping[str, VersionRec]
    units: Mapping[str, UnitRec]
    edges: tuple[EdgeRec, ...]
    events: Mapping[str, EventRec]
    by_to: Mapping[tuple[str, str], tuple[EdgeRec, ...]]
    by_from: Mapping[tuple[str, str], tuple[EdgeRec, ...]]

    @classmethod
    def empty(cls) -> DomainGraph:
        return from_records(())

    def aliases(self) -> dict[str, tuple[str, ...]]:
        return {entity.id: entity.terms for entity in self.entities.values() if entity.terms}


def from_records(records: Iterable[Record]) -> DomainGraph:
    """Monta o grafo; campo obrigatório ausente, origem desconhecida ou ponta inexistente → `ValueError` que nomeia a
    linha. `kind` desconhecido é ignorado (compatibilidade dentro da versão 2)."""
    meta: dict[str, Any] = {}
    entities: dict[str, EntityRec] = {}
    versions: dict[str, VersionRec] = {}
    units: dict[str, UnitRec] = {}
    events: dict[str, EventRec] = {}
    edges: list[EdgeRec] = []
    for line, raw in enumerate(records, start=1):
        kind = raw.get("kind")
        try:
            if kind == "meta":
                if raw["version"] != GRAPH_VERSION:
                    raise ValueError(f"versão {raw['version']!r}; esperada {GRAPH_VERSION}")
                meta = dict(raw)
            elif kind == "entity":
                entity = EntityRec(
                    id=str(raw["id"]),
                    type=str(raw["type"]),
                    name=str(raw["name"]),
                    terms=tuple(str(t) for t in raw["terms"]),
                    aliases=tuple(str(a) for a in raw.get("aliases") or ()),
                    source=str(raw["source"]),
                    origin=_origin(raw["origin"]),
                    level=raw.get("level"),
                )
                if entity.id in entities:
                    raise ValueError(f"entidade repetida: {entity.id}")
                entities[entity.id] = entity
            elif kind == "version":
                versions[str(raw["doc_key"])] = VersionRec(
                    doc_key=str(raw["doc_key"]),
                    document_id=str(raw["document_id"]),
                    type=str(raw["type"]),
                    role=cast(Role, raw["role"]),
                    status=str(raw["status"]),
                    scope=raw.get("scope"),
                    date=date.fromisoformat(raw["date"]) if raw.get("date") else None,
                    bodies=tuple(raw.get("bodies") or ()),
                    chunk_ids=tuple(str(c) for c in raw.get("chunk_ids") or ()),
                    ementa=raw.get("ementa"),
                )
            elif kind == "unit":
                units[str(raw["unit_id"])] = UnitRec(
                    unit_id=str(raw["unit_id"]),
                    doc_key=str(raw["doc_key"]),
                    unit_kind=str(raw["unit_kind"]),
                    title=str(raw["title"]),
                    chunk_ids=tuple(str(c) for c in raw["chunk_ids"]),
                    branches=tuple(raw.get("branches") or ()),
                    parent=raw.get("parent"),
                    ancestors=tuple(raw.get("ancestors") or ()),
                )
            elif kind == "event":
                event = EventRec(
                    id=str(raw["id"]),
                    doc_key=str(raw["doc_key"]),
                    date=date.fromisoformat(raw["date"]) if raw.get("date") else None,
                    type=cast(EventType, raw["type"]),
                    text=str(raw["text"]),
                    page=raw.get("page"),
                    chunk_ids=tuple(str(c) for c in raw.get("chunk_ids") or ()),
                )
                events[event.id] = event
            elif kind == "edge":
                edges.append(
                    EdgeRec(
                        type=str(raw["type"]),
                        from_=str(raw["from"]),
                        to=str(raw["to"]),
                        source=str(raw["source"]),
                        origin=_origin(raw["origin"]),
                        mode=raw.get("mode"),
                        checked=bool(raw.get("checked", False)),
                        attrs=MappingProxyType(dict(raw.get("attrs") or {})),
                    )
                )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"graph.jsonl, linha {line} ({kind}): {error}") from error
    known = set(entities) | set(versions) | set(units) | set(events)
    for edge in edges:
        for end in (edge.from_, edge.to):
            if end not in known:
                raise ValueError(f"aresta {edge.type} {edge.from_} → {edge.to}: ponta inexistente {end}")
    by_to: dict[tuple[str, str], list[EdgeRec]] = {}
    by_from: dict[tuple[str, str], list[EdgeRec]] = {}
    for edge in edges:
        by_to.setdefault((edge.type, edge.to), []).append(edge)
        by_from.setdefault((edge.type, edge.from_), []).append(edge)
    return DomainGraph(
        meta=MappingProxyType(meta),
        entities=MappingProxyType(entities),
        versions=MappingProxyType(versions),
        units=MappingProxyType(units),
        edges=tuple(edges),
        events=MappingProxyType(events),
        by_to=MappingProxyType({key: tuple(value) for key, value in by_to.items()}),
        by_from=MappingProxyType({key: tuple(value) for key, value in by_from.items()}),
    )


def _origin(value: object) -> Origin:
    if value not in ORIGINS:
        raise ValueError(f"origem desconhecida: {value!r}")
    return value
