"""Ids de ponto e conversão do modelo para payload JSON (spec 102 research R9, R12).

Puro: os adapters usam estas funções para gravar e ler; o schema que elas produzem é detalhe dos adapters, não seam.
"""

from __future__ import annotations

import dataclasses
import uuid
from collections.abc import Mapping
from datetime import date, datetime
from typing import Any, Final

from ai_scouting.rag.index.api import DocumentType, EmbeddingModel, IndexedChunk, IndexVersion, ValidityStatus
from ai_scouting.rag.index.ports import DocumentFields, DocumentRecord, StoredDocument

NAMESPACE_CHUNK: Final = uuid.UUID("5d0f3c1e-7f4b-5a43-9a11-2c6e8b0f1a01")
NAMESPACE_DOCUMENT: Final = uuid.UUID("5d0f3c1e-7f4b-5a43-9a11-2c6e8b0f1a02")

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


def chunk_to_payload(chunk: IndexedChunk) -> dict[str, Any]:
    payload = {f.name: _json(getattr(chunk, f.name)) for f in dataclasses.fields(chunk) if f.name != "score"}
    payload["kind"] = "chunk"
    return payload


def chunk_from_payload(payload: Mapping[str, Any], score: float | None = None) -> IndexedChunk:
    values: dict[str, Any] = {
        f.name: payload.get(f.name) for f in dataclasses.fields(IndexedChunk) if f.name != "score"
    }
    for name in _DOCUMENT_DATES:
        values[name] = _date(values[name])
    values["document_type"] = DocumentType(values["document_type"])
    values["status"] = ValidityStatus(values["status"])
    return IndexedChunk(**values, score=score)


def document_to_payload(record: DocumentRecord) -> dict[str, Any]:
    return {
        "kind": "document",
        "doc_key": record.doc_key,
        "fields": {name: _json(value) for name, value in _fields_dict(record.fields).items()},
        "content_key": record.content_key,
        "source_file": record.source_file,
        "source_sha256": record.source_sha256,
        "chunk_ids": list(record.chunk_ids),
        "pages": list(record.pages),
    }


def stored_from_payload(payload: Mapping[str, Any]) -> StoredDocument:
    return StoredDocument(
        fields=_fields_from(payload["fields"]),
        content_key=payload["content_key"],
        source_file=payload["source_file"],
        source_sha256=payload["source_sha256"],
        chunk_ids=tuple(payload["chunk_ids"]),
    )


def record_from_payload(payload: Mapping[str, Any]) -> DocumentRecord:
    stored = stored_from_payload(payload)
    fields = {f.name: getattr(stored, f.name) for f in dataclasses.fields(stored)}
    return DocumentRecord(**fields, pages=tuple(payload["pages"]))


def stored(record: DocumentRecord) -> StoredDocument:
    """O registro sem as páginas (listagem)."""
    return StoredDocument(**{f.name: getattr(record, f.name) for f in dataclasses.fields(StoredDocument)})


def _fields_from(raw: Mapping[str, Any]) -> DocumentFields:
    values: dict[str, Any] = dict(raw)
    for name in _DOCUMENT_DATES:
        values[name] = _date(values.get(name))
    values["document_type"] = DocumentType(values["document_type"])
    values["status"] = ValidityStatus(values["status"])
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
    }


def version_from_json(raw: Mapping[str, Any]) -> IndexVersion:
    embedding = raw["embedding"]
    return IndexVersion(
        chunking=raw["chunking"],
        embedding=EmbeddingModel(
            name=embedding["name"], fingerprint=embedding["fingerprint"], dimensions=int(embedding["dimensions"])
        ),
        sparse=raw["sparse"],
        built_at=datetime.fromisoformat(raw["built_at"]),
    )


def _json(value: object) -> object:
    if isinstance(value, ValidityStatus | DocumentType):
        return value.value
    if isinstance(value, date):
        return value.isoformat()
    return value


def _date(value: object) -> date | None:
    if value is None or isinstance(value, date):
        return value
    return date.fromisoformat(str(value))
