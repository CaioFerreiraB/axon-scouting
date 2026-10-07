"""Kit dos testes do índice: documentos e chunks sintéticos e os alvos da suíte de conformidade (spec 102 R16)."""

from __future__ import annotations

import dataclasses
import os
import uuid
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, date, datetime
from typing import Any, Protocol

from ai_scouting.rag.index.adapters.bm25 import Bm25Encoder
from ai_scouting.rag.index.adapters.memory import HashingEmbedder, MemoryIndex, MemoryStorage
from ai_scouting.rag.index.api import ChunkIndex, DocumentType, IndexedChunk, IndexVersion, ValidityStatus
from ai_scouting.rag.index.domain.points import with_document
from ai_scouting.rag.index.ports import (
    DocumentFields,
    DocumentRecord,
    DocumentStore,
    Embedder,
    EncodedChunk,
    IndexWriter,
    SparseEncoder,
)

INSTANTE = datetime(2026, 10, 7, 15, 30, tzinfo=UTC)


class Index(ChunkIndex, IndexWriter, DocumentStore, Protocol):
    """O que os adapters do índice implementam juntos."""


Make = Callable[[Embedder, SparseEncoder], Index]
Scenario = Callable[[Make], Awaitable[None]]


def versao(embedder: Embedder | None = None, sparse: SparseEncoder | None = None, **changes: Any) -> IndexVersion:
    embedder = embedder or HashingEmbedder()
    sparse = sparse or Bm25Encoder()
    base = IndexVersion(
        chunking="estrutural-v1;janela=300-600",
        embedding=embedder.model(),
        sparse=sparse.identity(),
        built_at=INSTANTE,
    )
    return dataclasses.replace(base, **changes)


def campos(document_id: str = "manual-x", version: str = "2026", **changes: Any) -> DocumentFields:
    base = DocumentFields(
        document_id=document_id,
        version=version,
        document_type=DocumentType.MANUAL,
        document_title="Manual X",
        jurisdiction="nacional",
        effective_date=date(2026, 3, 1),
        status=ValidityStatus.VIGENTE,
    )
    return dataclasses.replace(base, **changes)


def trecho(fields: DocumentFields, section: str, page: int, text: str, **changes: Any) -> IndexedChunk:
    base = IndexedChunk(
        chunk_id=f"{fields.doc_key}#{section}@p{page}",
        text=text,
        doc_key=fields.doc_key,
        document_id=fields.document_id,
        version=fields.version,
        document_type=fields.document_type,
        document_title=fields.document_title,
        jurisdiction=fields.jurisdiction,
        page_start=page,
        page_end=page,
        effective_date=fields.effective_date,
        status=fields.status,
    )
    return dataclasses.replace(with_document(base, fields), **changes)


def registro(fields: DocumentFields, chunks: Sequence[IndexedChunk], pages: Sequence[str]) -> DocumentRecord:
    return DocumentRecord(
        fields=fields,
        content_key=f"ck-{fields.doc_key}",
        source_file=f"{fields.doc_key}.pdf",
        source_sha256="0" * 64,
        chunk_ids=tuple(c.chunk_id for c in chunks),
        pages=tuple(pages),
    )


async def codificar(chunks: Sequence[IndexedChunk], embedder: Embedder, sparse: SparseEncoder) -> list[EncodedChunk]:
    dense = await embedder.embed_passages([c.text for c in chunks])
    return [
        EncodedChunk(chunk=c, dense=d, sparse=sparse.encode_passage(c.text)) for c, d in zip(chunks, dense, strict=True)
    ]


# --- alvos da suíte de conformidade -----------------------------------------------------------------------------


class Target(Protocol):
    name: str

    async def run(self, scenario: Scenario) -> None: ...


class MemoryTarget:
    name = "memoria"

    async def run(self, scenario: Scenario) -> None:
        storage = MemoryStorage()
        await scenario(lambda e, s: MemoryIndex("scouting", e, s, storage=storage))


def qdrant_targets() -> list[Any]:
    """Os alvos Qdrant (modo local sempre; servidor com `AI_SCOUTING_TEST_QDRANT_URL`), quando o adapter existe."""
    try:
        from qdrant_client import AsyncQdrantClient

        from ai_scouting.rag.index.adapters.qdrant import QdrantIndex
    except ImportError:  # pragma: no cover - o adapter chega na T031
        return []

    class LocalTarget:
        name = "qdrant-local"

        async def run(self, scenario: Scenario) -> None:
            client = AsyncQdrantClient(location=":memory:")
            try:
                await scenario(lambda e, s: QdrantIndex(client, "scouting", e, s))
            finally:
                await client.close()

    class ServerTarget:
        name = "qdrant-servidor"

        def __init__(self, url: str) -> None:
            self.url = url

        async def run(self, scenario: Scenario) -> None:
            client = AsyncQdrantClient(url=self.url)
            index = f"t{uuid.uuid4().hex[:10]}"
            try:
                await scenario(lambda e, s: QdrantIndex(client, index, e, s))
            finally:
                for collection in (await client.get_collections()).collections:
                    if collection.name.startswith(f"{index}__"):
                        await client.delete_collection(collection.name)
                await client.close()

    targets: list[Any] = [LocalTarget()]
    url = os.environ.get("AI_SCOUTING_TEST_QDRANT_URL")
    if url:
        targets.append(ServerTarget(url))
    return targets


def outro_embedder(fingerprint: str = "outro", dimensions: int = 64) -> HashingEmbedder:
    return HashingEmbedder(dimensions=dimensions, fingerprint=fingerprint)
