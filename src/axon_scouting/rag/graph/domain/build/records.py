"""Os registros do `graph.jsonl` v2 a partir das fontes (spec 106 T016, research R2, R10; contracts/graph-jsonl-v2.md).

No G1, o grafo é o do ADR-063 no formato novo:
- `entity`: os tipos de órgão (`tipo_orgao:`) e de instituição (`tipo_inst:`) com os mesmos termos do `organ_terms`, e
  os cargos (`cargo:`) sem termos — só para as pontas das relações de governança;
- `version`: todo documento do catálogo, com o papel (`ROLES`);
- `unit` e `edge` `CONTEM`/`SOBRE` (modo pelo papel): as seções que disciplinam cada órgão;
- `edge` de governança: as relações do `organizacao.yaml` com fonte conferida e presente no índice.
A fonte de uma entidade curada é a entrada do catálogo (`organizacao.yaml:<seção>/<id>`), como o grafo de hoje: o órgão
cujo `instituido_em` está "a verificar" continua reconhecível (research R18). Puro.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from axon_scouting.rag.graph.api import MODE_OF, ROLES
from axon_scouting.rag.graph.domain.build.organs import (
    VERIFY,
    ChunkPlace,
    cargo_terms,
    organs_from_mapping,
    relations_from_mapping,
    sections,
)
from axon_scouting.rag.graph.domain.model import GRAPH_VERSION
from axon_scouting.rag.graph.domain.roles import missing_roles

ORGANIZATION: Final = "organizacao.yaml"


@dataclass(slots=True)
class BuildReport:
    entities: int = 0
    sections: int = 0
    relations: int = 0
    versions: int = 0
    dropped: list[str] = field(default_factory=list)
    per_entity: dict[str, list[str]] = field(default_factory=dict)


class BuildError(ValueError):
    """Um invariante do build falhou (por exemplo, tipo de documento sem papel); a mensagem nomeia o caso."""


def places_of(chunks: Sequence[Mapping[str, Any]]) -> list[ChunkPlace]:
    return [
        ChunkPlace(
            chunk_id=raw["chunk_id"],
            doc_key=raw["doc_key"],
            document_type=raw["document_type"],
            document_title=raw.get("document_title") or "",
            scope=raw.get("scope"),
            vigente=raw.get("status") == "vigente",
            path=tuple((p["unit_id"], p["kind"], p.get("title") or "") for p in raw.get("path") or []),
        )
        for raw in chunks
    ]


def build_records(
    *,
    organization: Mapping[str, Any],
    chunks: Sequence[Mapping[str, Any]],
    documents: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    organization_sha256: str,
) -> tuple[list[dict[str, Any]], BuildReport]:
    """Puro: recebe o que a porta `GraphSources` leu (o `scripts/grafo.py` compõe)."""
    raw = organization
    report = BuildReport()

    types = [str(doc["fields"]["document_type"]) for doc in documents]
    missing = missing_roles(types)
    if missing:
        raise BuildError(f"tipos de documento sem papel em ROLES (106 FR-005): {', '.join(missing)}")

    organs = organs_from_mapping(raw)
    places = places_of(chunks)
    found = sections(places, organs)
    relations = relations_from_mapping(raw, {place.chunk_id for place in places})
    cargos = cargo_terms(raw)

    prefix = {organ.id: ("tipo_inst" if organ.institution else "tipo_orgao") for organ in organs}
    section_of = {organ.id: ("tipos_de_instituicao" if organ.institution else "tipos_de_orgao") for organ in organs}
    entities: list[dict[str, Any]] = []
    for organ in organs:
        entities.append(
            {
                "kind": "entity",
                "id": f"{prefix[organ.id]}:{organ.id}",
                "type": prefix[organ.id],
                "name": organ.name,
                "terms": list(dict.fromkeys((*organ.terms, *cargos.get(organ.id, ())))),
                "aliases": list(organ.terms),
                "source": f"{ORGANIZATION}:{section_of[organ.id]}/{organ.id}",
                "origin": "curado",
                "level": organ.level or None,
            }
        )
    for entry in raw.get("cargos") or []:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("id"), str):
            continue
        fonte = entry.get("fonte")
        entities.append(
            {
                "kind": "entity",
                "id": f"cargo:{entry['id']}",
                "type": "cargo",
                "name": str(entry.get("nome") or entry["id"]),
                "terms": [],
                "aliases": [],
                "source": fonte
                if isinstance(fonte, str) and fonte != VERIFY
                else f"{ORGANIZATION}:cargos/{entry['id']}",
                "origin": "curado",
                "level": None,
            }
        )
        prefix.setdefault(str(entry["id"]), "cargo")
    report.entities = len(entities)

    versions: list[dict[str, Any]] = []
    for doc in documents:
        fields = doc["fields"]
        kind = str(fields["document_type"])
        versions.append(
            {
                "kind": "version",
                "doc_key": doc["doc_key"],
                "document_id": fields["document_id"],
                "type": kind,
                "role": ROLES[kind],
                "status": fields["status"],
                "scope": fields.get("scope"),
                "date": fields.get("effective_date"),
                "bodies": list(fields.get("bodies") or ()),
            }
        )
    report.versions = len(versions)
    role_of = {v["doc_key"]: v["role"] for v in versions}
    kind_of = {unit: kind for place in places for unit, kind, _ in place.path}

    units: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    for section in found:
        if section.unit_id not in units:
            units[section.unit_id] = {
                "kind": "unit",
                "unit_id": section.unit_id,
                "doc_key": section.doc_key,
                "unit_kind": kind_of.get(section.unit_id, "documento"),
                "title": section.title,
                "chunk_ids": list(section.chunk_ids),
            }
            edges.append(
                {
                    "kind": "edge",
                    "type": "CONTEM",
                    "from": section.doc_key,
                    "to": section.unit_id,
                    "source": section.unit_id,
                    "origin": "regra",
                }
            )
        edges.append(
            {
                "kind": "edge",
                "type": "SOBRE",
                "mode": MODE_OF[role_of[section.doc_key]],
                "from": section.unit_id,
                "to": f"{prefix[section.organ]}:{section.organ}",
                "source": section.unit_id,
                "origin": "regra",
            }
        )
        report.per_entity.setdefault(section.organ, []).append(f"{section.title} ({section.doc_key})")
    report.sections = len(found)

    for relation in relations:
        ends = (relation.source, relation.target)
        if any(end not in prefix for end in ends):
            report.dropped.append(f"{relation.kind} {relation.source} → {relation.target}: ponta fora do catálogo")
            continue
        edges.append(
            {
                "kind": "edge",
                "type": relation.kind,
                "from": f"{prefix[relation.source]}:{relation.source}",
                "to": f"{prefix[relation.target]}:{relation.target}",
                "source": relation.chunk_id,
                "origin": "curado",
            }
        )
        report.relations += 1

    meta = {
        "kind": "meta",
        "version": GRAPH_VERSION,
        "chunking": manifest.get("chunking"),
        "built_at": manifest.get("built_at"),
        "sources": {ORGANIZATION: organization_sha256},
    }
    return [meta, *entities, *versions, *units.values(), *edges], report
