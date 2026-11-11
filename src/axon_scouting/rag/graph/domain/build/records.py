"""Os registros do `graph.jsonl` v2 a partir das fontes (spec 106 T016, research R2, R10; contracts/graph-jsonl-v2.md).

No G1, o grafo é o do ADR-063 no formato novo:
- `entity`: os tipos de órgão (`tipo_orgao:`) e de instituição (`tipo_inst:`) com os mesmos termos do `organ_terms`, e
  os cargos (`cargo:`) sem termos — só para as pontas das relações de governança;
- `version`: todo documento do catálogo, com o papel (`ROLES`);
- `unit` e `edge` `CONTEM`/`SOBRE` (modo pelo papel): as seções que disciplinam cada órgão;
- `edge` de governança: as relações do `organizacao.yaml` com fonte conferida e presente no índice.
O G2 acrescenta as entidades de programa e adultos (`dominio.yaml`), o `SOBRE` delas, `E_UM`, `DE_RAMO`, `REGE` e
`DE_AREA`; o G3, as instituições e os órgãos reais (`inst:`, `orgao:`) com `INSTANCIA_DE` e `PARTE_DE` (também entre
os tipos), `APLICA_SE_A` e `EMITIDO_POR` pela taxonomia, e o `SOBRE` do documento inteiro cujo título nomeia a
instituição; o G4, os eventos das atas; o G5, as resoluções (`doc:` pelo número, o `SOBRE` pela ementa, `ALTERA`,
`REVOGA`, `CITA` e `SUBSTITUI`) e os trechos de cada versão.
A fonte de uma entidade curada é a entrada do catálogo (`organizacao.yaml:<seção>/<id>`), como o grafo de hoje: o órgão
cujo `instituido_em` está "a verificar" continua reconhecível (research R18). Puro.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from axon_scouting.rag.graph.api import MODE_OF, ROLES, reconhecer
from axon_scouting.rag.graph.domain.build.domain import DOMAIN, alias_map, domain_entities, rege
from axon_scouting.rag.graph.domain.build.instances import Instance, instances_from_mapping, type_parts
from axon_scouting.rag.graph.domain.build.minutes import events_graph
from axon_scouting.rag.graph.domain.build.organs import (
    VERIFY,
    ChunkPlace,
    cargo_terms,
    normalize,
    organs_from_mapping,
    relations_from_mapping,
    sections,
)
from axon_scouting.rag.graph.domain.build.resolutions import (
    checked_subjects,
    lineage_edges,
    norm_units,
    number_aliases,
    resolution_links,
    resolutions_of,
)
from axon_scouting.rag.graph.domain.build.sobre import ROLES_WITH_SOBRE, sobre_graph
from axon_scouting.rag.graph.domain.build.taxonomy import SOURCE, applies_to, folder_of, issued_by
from axon_scouting.rag.graph.domain.model import GRAPH_VERSION
from axon_scouting.rag.graph.domain.roles import missing_roles

ORGANIZATION: Final = "organizacao.yaml"


@dataclass(slots=True)
class BuildReport:
    entities: int = 0
    sections: int = 0
    relations: int = 0
    versions: int = 0
    applies: int = 0
    issued: int = 0
    events: int = 0
    resolutions: int = 0
    with_ementa: int = 0
    with_subject: int = 0
    model_subject: int = 0
    without_subject: list[str] = field(default_factory=list)
    links: dict[str, int] = field(default_factory=dict)
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
    domain: Mapping[str, Any] | None = None,
    domain_sha256: str | None = None,
    minutes: Sequence[Mapping[str, Any]] = (),
    subjects: Mapping[str, Any] | None = None,
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
    known_chunks = {place.chunk_id for place in places}
    instances = instances_from_mapping(raw, known_chunks)
    report.dropped.extend(instances.dropped)

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
    curated = domain_entities(domain or {}, raw)
    curated_ids = {entity.id for entity in curated}
    for entry in raw.get("cargos") or []:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("id"), str):
            continue
        if f"cargo:{entry['id']}" in curated_ids:
            prefix.setdefault(str(entry["id"]), "cargo")
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
    for entity in curated:
        entities.append(
            {
                "kind": "entity",
                "id": entity.id,
                "type": entity.type,
                "name": entity.name,
                "terms": list(entity.aliases) if entity.recognizable else [],
                "aliases": list(entity.aliases),
                "source": entity.source,
                "origin": "curado",
                "level": None,
            }
        )
    for instance in instances.found:
        entities.append(
            {
                "kind": "entity",
                "id": instance.id,
                "type": instance.type,
                "name": instance.name,
                "terms": list(instance.terms),
                "aliases": list(instance.terms),
                "source": instance.source,
                "origin": "curado",
                "level": instance.level,
            }
        )
    # G5: cada resolução é um `doc:` reconhecível pelo número ("resolucao can 15 2025"), com a fonte no 1º trecho.
    resolutions = resolutions_of(documents)
    taken = {str(e["id"]) for e in entities}
    for resolution in resolutions:
        ident = f"doc:{resolution.document_id}"
        if resolution.number is None or resolution.first_chunk is None or ident in taken:
            continue
        taken.add(ident)
        entities.append(
            {
                "kind": "entity",
                "id": ident,
                "type": "doc",
                "name": resolution.title,
                "terms": list(number_aliases(resolution.number)),
                "aliases": [],
                "source": resolution.first_chunk,
                "origin": "regra",
                "level": None,
            }
        )
    report.entities = len(entities)
    ementa_of = {r.doc_key: r.ementa for r in resolutions if r.ementa}

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
                "chunk_ids": list(doc.get("chunk_ids") or ()),
                **({"ementa": ementa_of[doc["doc_key"]]} if doc["doc_key"] in ementa_of else {}),
            }
        )
    report.versions = len(versions)
    role_of = {v["doc_key"]: v["role"] for v in versions}
    status_of = {v["doc_key"]: v["status"] for v in versions}
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

    # G2: SOBRE das entidades de programa e adultos, por título e pelo ramo do documento (research R6).
    branch_of = {doc["doc_key"]: doc["fields"]["branch"] for doc in documents if doc["fields"].get("branch")}
    generic = sobre_graph(
        chunks,
        {key: {"role": role_of[key], "status": status_of[key]} for key in role_of},
        alias_map([entity for entity in curated if entity.type != "doc"]),  # o documento é alvo de eventos (G4)
        branch_of,
    )
    for unit_id, unit in generic.units.items():
        if unit_id in units:
            units[unit_id]["ancestors"] = unit["ancestors"]
            continue
        units[unit_id] = {"kind": "unit", "unit_id": unit_id, **unit}
        edges.append(
            {
                "kind": "edge",
                "type": "CONTEM",
                "from": unit["doc_key"],
                "to": unit_id,
                "source": unit_id,
                "origin": "regra",
            }
        )
    for unit_id, target in generic.edges:
        doc_key = units[unit_id]["doc_key"]
        edges.append(
            {
                "kind": "edge",
                "type": "SOBRE",
                "mode": MODE_OF[role_of[doc_key]],
                "from": unit_id,
                "to": target,
                "source": unit_id,
                "origin": "regra",
            }
        )
        report.per_entity.setdefault(target, []).append(units[unit_id]["title"] or unit_id)
    for entity in curated:
        for specific in entity.covers:
            if specific in curated_ids:
                edges.append(
                    {
                        "kind": "edge",
                        "type": "E_UM",
                        "from": specific,
                        "to": entity.id,
                        "source": entity.source,
                        "origin": "curado",
                    }
                )
            else:
                report.dropped.append(f"E_UM {specific} → {entity.id}: entidade fora do catálogo")
        if entity.ramo:
            edges.append(
                {
                    "kind": "edge",
                    "type": "DE_RAMO",
                    "from": entity.id,
                    "to": entity.ramo,
                    "source": entity.source,
                    "origin": "curado",
                }
            )
    for governing, target, source in rege(domain or {}):
        if governing in curated_ids and target in curated_ids:
            edges.append(
                {"kind": "edge", "type": "REGE", "from": governing, "to": target, "source": source, "origin": "curado"}
            )
        else:
            report.dropped.append(f"REGE {governing} → {target}: entidade fora do catálogo")
    area_of = {category: entity.id for entity in curated for category in entity.categories}
    # O documento inteiro de uma área, para o salto REGE + DE_AREA (106 R19): os trechos vigentes de norma e orientação.
    doc_chunks: dict[str, list[str]] = {}
    for chunk in chunks:
        key = chunk["doc_key"]
        if role_of.get(key) in ("norma", "orientacao") and status_of.get(key) == "vigente":
            doc_chunks.setdefault(key, []).append(chunk["chunk_id"])
    for doc in documents:
        area = area_of.get(str(doc["fields"].get("category")))
        whole = f"{doc['doc_key']}#documento"
        if area and doc["doc_key"] in doc_chunks and whole not in units:
            units[whole] = {
                "kind": "unit",
                "unit_id": whole,
                "doc_key": doc["doc_key"],
                "unit_kind": "documento",
                "title": str(doc["fields"].get("document_title") or ""),
                "chunk_ids": doc_chunks[doc["doc_key"]],
            }
            edges.append(
                {
                    "kind": "edge",
                    "type": "CONTEM",
                    "from": doc["doc_key"],
                    "to": whole,
                    "source": whole,
                    "origin": "regra",
                }
            )
        if area:
            edges.append(
                {
                    "kind": "edge",
                    "type": "DE_AREA",
                    "from": doc["doc_key"],
                    "to": area,
                    "source": f"taxonomia:{doc['fields'].get('category')}",
                    "origin": "regra",
                }
            )

    # G3: as instâncias, as partes e a aplicabilidade (research R20).
    edges.extend(_instance_edges(instances.found, type_parts(raw, known_chunks)))
    institutions, organs_found = instances.institutions(), instances.organs()
    titled = {i.id: i.terms for i in institutions if i.terms}
    issuers: dict[str, list[str]] = {}
    for doc in documents:
        fields, source_file = doc["fields"], str(doc.get("source_file") or "")
        applied = applies_to(fields, source_file, institutions)
        source = f"{SOURCE}{folder_of(source_file)}"
        if applied is not None:
            edges.append(_edge("APLICA_SE_A", doc["doc_key"], applied, source, "regra"))
            report.applies += 1
        for issuer in issued_by(fields, applied, organs_found):
            edges.append(_edge("EMITIDO_POR", doc["doc_key"], issuer, source, "regra"))
            issuers.setdefault(doc["doc_key"], []).append(issuer)
            report.issued += 1
        # O documento cujo título nomeia a instituição trata dela por inteiro (o estatuto do GE, o regulamento de SP).
        key = doc["doc_key"]
        if role_of.get(key) not in ROLES_WITH_SOBRE or status_of.get(key) != "vigente" or key not in doc_chunks:
            continue
        for named in reconhecer(normalize(str(fields.get("document_title") or "")), titled):
            whole = f"{key}#documento"
            if whole not in units:
                units[whole] = {
                    "kind": "unit",
                    "unit_id": whole,
                    "doc_key": key,
                    "unit_kind": "documento",
                    "title": str(fields.get("document_title") or ""),
                    "chunk_ids": doc_chunks[key],
                }
                edges.append(_edge("CONTEM", key, whole, whole, "regra"))
            edges.append({**_edge("SOBRE", whole, named, whole, "regra"), "mode": MODE_OF[role_of[key]]})
            report.per_entity.setdefault(named, []).append(units[whole]["title"])

    # G5: a ementa de cada resolução vigente dá o assunto dela; as ligações escritas no texto (research R22).
    names = alias_map([entity for entity in curated if entity.type != "doc"])
    proposed = checked_subjects(subjects or {}, chunks, {entity.id for entity in curated if entity.type != "doc"})
    report.dropped.extend(proposed.dropped)
    report.resolutions = len(resolutions)
    for resolution in resolutions:
        key = resolution.doc_key
        if resolution.ementa:
            report.with_ementa += 1
        about = reconhecer(normalize(resolution.ementa), names) if resolution.ementa else ()
        origin, source = "regra", resolution.first_chunk or f"{key}#documento"
        if about:
            report.with_subject += 1
        elif key in proposed.found:  # o assunto proposto pelo modelo, com o trecho conferido
            about, source = proposed.found[key]
            origin = "modelo_conferido"
            report.model_subject += 1
        else:
            report.without_subject.append(key)
            continue
        if status_of.get(key) != "vigente" or key not in doc_chunks:
            continue
        whole = f"{key}#documento"
        if whole not in units:
            units[whole] = {
                "kind": "unit",
                "unit_id": whole,
                "doc_key": key,
                "unit_kind": "documento",
                "title": resolution.title,
                "chunk_ids": doc_chunks[key],
            }
            edges.append(_edge("CONTEM", key, whole, whole, "regra"))
        for target in about:
            edge = {**_edge("SOBRE", whole, target, source, origin), "mode": MODE_OF[role_of[key]]}
            edges.append({**edge, "checked": True} if origin == "modelo_conferido" else edge)
            report.per_entity.setdefault(target, []).append(resolution.title)
    norms = {
        str(v["type"]): str(v["doc_key"]) for v in versions if v["status"] == "vigente" and v["scope"] == "nacional"
    }
    links = resolution_links(
        resolutions,
        chunks,
        norm_units(chunks, norms.get("por"), "regra"),
        norm_units(chunks, norms.get("estatuto"), "artigo"),
    )
    placed = _unit_places(chunks)
    for unit_id in sorted(links.units):
        if unit_id not in units and unit_id in placed:
            units[unit_id] = {"kind": "unit", "unit_id": unit_id, **placed[unit_id]}
            edges.append(_edge("CONTEM", placed[unit_id]["doc_key"], unit_id, unit_id, "regra"))
    lineage = lineage_edges(documents)
    for edge in [*links.edges, *lineage]:
        edges.append(edge)
        report.links[edge["type"]] = report.links.get(edge["type"], 0) + 1
    report.dropped.extend(dict.fromkeys(links.dropped))

    # G4: os eventos das fichas das atas, com GERA e DE (research R21).
    recognizable = {str(e["id"]): tuple(e["terms"]) for e in entities if e["terms"]}
    dated = {str(v["doc_key"]): {"date": v["date"]} for v in versions}
    minutes_graph = events_graph(minutes, chunks, dated, recognizable, issuers)
    edges.extend(minutes_graph.edges)
    report.dropped.extend(minutes_graph.dropped)
    report.events = len(minutes_graph.events)

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
        "sources": {ORGANIZATION: organization_sha256, **({DOMAIN: domain_sha256} if domain_sha256 else {})},
    }
    return [meta, *entities, *versions, *units.values(), *edges, *minutes_graph.events], report


def _unit_places(chunks: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Cada unidade do caminho dos trechos: o documento, o tipo, o título e os trechos dela e dos de baixo."""
    found: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        for step in chunk.get("path") or []:
            unit = found.setdefault(
                str(step["unit_id"]),
                {
                    "doc_key": chunk["doc_key"],
                    "unit_kind": step.get("kind") or "documento",
                    "title": step.get("title") or "",
                    "chunk_ids": [],
                },
            )
            unit["chunk_ids"].append(chunk["chunk_id"])
    return found


def _edge(kind: str, start: str, end: str, source: str, origin: str) -> dict[str, Any]:
    return {"kind": "edge", "type": kind, "from": start, "to": end, "source": source, "origin": origin}


def _instance_edges(instances: Sequence[Instance], parts: Sequence[tuple[str, str, str]]) -> list[dict[str, Any]]:
    """`INSTANCIA_DE` (a instância → o tipo), `PARTE_DE` entre as instâncias e entre os tipos (o tipo de órgão → o tipo
    de instituição dele), todas curadas."""
    known = {i.id for i in instances}
    found = [_edge("INSTANCIA_DE", i.id, i.kind_of, i.source, "curado") for i in instances]
    for instance in instances:
        for parent in instance.parents:
            if parent in known:
                found.append(
                    _edge("PARTE_DE", instance.id, parent, instance.parent_source or instance.source, "curado")
                )
    found.extend(_edge("PARTE_DE", part, whole, source, "curado") for part, whole, source in parts)
    return found
