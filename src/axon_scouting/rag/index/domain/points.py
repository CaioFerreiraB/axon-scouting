"""Ids de ponto e conversão do modelo para payload JSON (spec 102 research R9, R12).

Puro: os adapters usam estas funções para gravar e ler; o schema que elas produzem é detalhe dos adapters, não seam.
"""

from __future__ import annotations

import dataclasses
import uuid
from collections.abc import Mapping
from datetime import date, datetime
from typing import Any, Final

from axon_scouting.rag.index.api import (
    CatalogEntry,
    EmbeddingModel,
    IndexedChunk,
    IndexVersion,
    UnitKind,
    UnitNode,
    UnitRef,
    ValidityStatus,
)
from axon_scouting.rag.index.ports import DocumentFields, DocumentRecord, Original, StoredDocument

NAMESPACE_CHUNK: Final = uuid.UUID("5d0f3c1e-7f4b-5a43-9a11-2c6e8b0f1a01")
NAMESPACE_DOCUMENT: Final = uuid.UUID("5d0f3c1e-7f4b-5a43-9a11-2c6e8b0f1a02")

FILTER_BRANCHES: Final = "filter_branches"
"""Os ramos efetivos do trecho (os dele, senão o do documento), gravados à parte para o filtro do armazenamento."""
_COMPUTED: Final = ("score", "similarity")
"""Calculados na busca, nunca gravados."""
_DOCUMENT_DATES: Final = ("effective_date", "publication_date", "expiry_date", "successor_expiry_date")


def chunk_point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(NAMESPACE_CHUNK, chunk_id))


def document_point_id(doc_key: str) -> str:
    return str(uuid.uuid5(NAMESPACE_DOCUMENT, doc_key))


def build_name(index: str, built_at: datetime, existing: set[str]) -> str:
    """`<índice>__<AAAAMMDDTHHMMSSZ>`, com `-2`, `-3`… se o nome já existe."""
    base = f"{index}__{built_at.strftime('%Y%m%dT%H%M%SZ')}"
    name, n = base, 1
    while name in existing:
        n += 1
        name = f"{base}-{n}"
    return name


def with_document(chunk: IndexedChunk, fields: DocumentFields) -> IndexedChunk:
    """O chunk com os campos de documento (metadados e vigência) reescritos a partir do registro."""
    return dataclasses.replace(chunk, doc_key=fields.doc_key, **_fields_dict(fields))


def _fields_dict(fields: DocumentFields) -> dict[str, Any]:
    return {f.name: getattr(fields, f.name) for f in dataclasses.fields(fields)}


def original_of(chunk: IndexedChunk, record: DocumentRecord) -> Original:
    """As páginas brutas da faixa do chunk (FR-009)."""
    return Original(
        chunk_id=chunk.chunk_id,
        doc_key=record.doc_key,
        document_title=record.fields.document_title,
        version=record.fields.version,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        pages=record.pages[chunk.page_start - 1 : chunk.page_end],
        source_file=record.source_file,
        source_sha256=record.source_sha256,
        source_url=record.source_url,
    )


def chunk_to_payload(chunk: IndexedChunk) -> dict[str, Any]:
    payload = {f.name: _json(getattr(chunk, f.name)) for f in dataclasses.fields(chunk) if f.name not in _COMPUTED}
    payload["path"] = [{"unit_id": u.unit_id, "kind": u.kind, "title": u.title} for u in chunk.path]
    payload["branches"] = list(chunk.branches)
    payload[FILTER_BRANCHES] = list(chunk.effective_branches)  # o filtro do Qdrant lê só este campo
    payload["kind"] = "chunk"
    return payload


def chunk_from_payload(
    payload: Mapping[str, Any], score: float | None = None, similarity: float | None = None
) -> IndexedChunk:
    values: dict[str, Any] = {
        f.name: payload.get(f.name) for f in dataclasses.fields(IndexedChunk) if f.name not in _COMPUTED
    }
    for name in _DOCUMENT_DATES:
        values[name] = _date(values[name])
    values["status"] = ValidityStatus(values["status"])
    values["bodies"] = tuple(values["bodies"] or ())
    values["path"] = tuple(UnitRef(**ref) for ref in values["path"] or ())  # artefato anterior ao ADR-048: sem árvore
    values["branches"] = tuple(values["branches"] or ())
    values["shape"] = values["shape"] or "prose"
    return IndexedChunk(**values, score=score, similarity=similarity)


def document_to_payload(record: DocumentRecord) -> dict[str, Any]:
    return {
        "kind": "document",
        "doc_key": record.doc_key,
        "fields": {name: _json(value) for name, value in _fields_dict(record.fields).items()},
        "content_key": record.content_key,
        "source_file": record.source_file,
        "source_sha256": record.source_sha256,
        "source_url": record.source_url,
        "chunk_ids": list(record.chunk_ids),
        "units": [_unit_json(unit) for unit in record.units],
        "pages": list(record.pages),
    }


def _unit_json(unit: UnitNode) -> dict[str, Any]:
    return {
        "unit_id": unit.unit_id,
        "kind": unit.kind,
        "title": unit.title,
        "page_start": unit.page_start,
        "page_end": unit.page_end,
        "parent": unit.parent,
        "branches": list(unit.branches),
    }


def stored_from_payload(payload: Mapping[str, Any]) -> StoredDocument:
    return StoredDocument(
        fields=_fields_from(payload["fields"]),
        content_key=payload["content_key"],
        source_file=payload["source_file"],
        source_sha256=payload["source_sha256"],
        source_url=payload.get("source_url"),
        chunk_ids=tuple(payload["chunk_ids"]),
        units=tuple(
            UnitNode(**{**raw, "branches": tuple(raw.get("branches") or ())}) for raw in payload.get("units") or ()
        ),
    )


def record_from_payload(payload: Mapping[str, Any]) -> DocumentRecord:
    stored = stored_from_payload(payload)
    fields = {f.name: getattr(stored, f.name) for f in dataclasses.fields(stored)}
    return DocumentRecord(**fields, pages=tuple(payload["pages"]))


def stored(record: DocumentRecord) -> StoredDocument:
    """O registro sem as páginas (listagem)."""
    return StoredDocument(**{f.name: getattr(record, f.name) for f in dataclasses.fields(StoredDocument)})


def catalog_entry(document: StoredDocument) -> CatalogEntry:
    """A entrada do catálogo (ADR-047): os campos do documento, a chave e o link do original."""
    return CatalogEntry(doc_key=document.doc_key, source_url=document.source_url, **_fields_dict(document.fields))


def _fields_from(raw: Mapping[str, Any]) -> DocumentFields:
    values: dict[str, Any] = dict(raw)
    for name in _DOCUMENT_DATES:
        values[name] = _date(values.get(name))
    values["status"] = ValidityStatus(values["status"])
    values["bodies"] = tuple(values.get("bodies") or ())
    return DocumentFields(**values)


def version_to_json(version: IndexVersion) -> dict[str, Any]:
    return {
        "chunking": version.chunking,
        "embedding": {
            "name": version.embedding.name,
            "fingerprint": version.embedding.fingerprint,
            "dimensions": version.embedding.dimensions,
        },
        "sparse": version.sparse,
        "built_at": version.built_at.isoformat(),
        **({"unit_kinds": unit_kinds_to_json(version.unit_kinds)} if version.unit_kinds else {}),
    }


def unit_kinds_to_json(kinds: tuple[UnitKind, ...]) -> list[dict[str, Any]]:
    return [{"name": k.name, "label": k.label, "plurals": list(k.plurals)} for k in kinds]


def unit_kinds_from_json(raw: object) -> tuple[UnitKind, ...]:
    if not isinstance(raw, list):
        return ()
    return tuple(UnitKind(name=k["name"], label=k["label"], plurals=tuple(k.get("plurals") or ())) for k in raw)


def version_from_json(raw: Mapping[str, Any]) -> IndexVersion:
    embedding = raw["embedding"]
    return IndexVersion(
        chunking=raw["chunking"],
        embedding=EmbeddingModel(
            name=embedding["name"], fingerprint=embedding["fingerprint"], dimensions=int(embedding["dimensions"])
        ),
        sparse=raw["sparse"],
        built_at=datetime.fromisoformat(raw["built_at"]),
        unit_kinds=unit_kinds_from_json(raw.get("unit_kinds")),
    )


def _json(value: object) -> object:
    if isinstance(value, ValidityStatus):
        return value.value
    if isinstance(value, date):
        return value.isoformat()
    return value


def _date(value: object) -> date | None:
    if value is None or isinstance(value, date):
        return value
    return date.fromisoformat(str(value))
