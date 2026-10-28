"""`QdrantIndex`: o índice do scouting no Qdrant (spec 102 research R12–R15).

Um build é uma coleção; o nome do índice é um alias para o build publicado, trocado numa operação só. Na coleção: os
vetores nomeados `dense` (cosseno) e `sparse` (IDF no servidor), a versão do índice na `metadata`, pontos de chunk e
pontos de documento sem vetor (o registro com o texto original). A busca usa dois `prefetch` com o mesmo filtro e
fusão RRF nativa; o resultado é reordenado por (`-score`, `chunk_id`) para empate determinístico. Implementa
`ChunkIndex`, `IndexWriter` e `DocumentStore`.
"""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Any, Final

from qdrant_client import AsyncQdrantClient
from qdrant_client import models as m

from axon_scouting.rag.index.api import (
    CatalogEntry,
    IndexedChunk,
    IndexFilters,
    IndexUnavailable,
    IndexVersion,
    UnitEntry,
    UnitKind,
    ValidityStatus,
)
from axon_scouting.rag.index.domain.compatibility import check_encoders
from axon_scouting.rag.index.domain.filters import matches
from axon_scouting.rag.index.domain.outline import outline_of, units_of
from axon_scouting.rag.index.domain.points import (
    FILTER_BRANCHES,
    build_name,
    catalog_entry,
    chunk_from_payload,
    chunk_point_id,
    chunk_to_payload,
    document_point_id,
    document_to_payload,
    original_of,
    record_from_payload,
    stored_from_payload,
    version_from_json,
    version_to_json,
    with_document,
)
from axon_scouting.rag.index.domain.scoring import cosine, ordered
from axon_scouting.rag.index.ports import (
    Build,
    DocumentRecord,
    Embedder,
    EncodedChunk,
    Original,
    SparseEncoder,
    StoredDocument,
)

DENSE: Final = "dense"
SPARSE: Final = "sparse"
BATCH: Final = 64
KEYWORD_FIELDS: Final = (
    "kind",
    "doc_key",
    "document_id",
    "status",
    "branch",
    "document_type",
    "bodies",
    "scope",
    FILTER_BRANCHES,
    "shape",
)
DATE_FIELDS: Final = ("effective_date", "expiry_date", "successor_expiry_date")


class QdrantIndex:
    def __init__(self, client: AsyncQdrantClient, index_name: str, embedder: Embedder, sparse: SparseEncoder) -> None:
        self._client = client
        self._name = index_name
        self._embedder = embedder
        self._sparse = sparse

    # --- ChunkIndex -------------------------------------------------------------------------------------------

    async def search(self, query: str, *, filters: IndexFilters, limit: int) -> list[IndexedChunk]:
        published = await self._require_published()
        check_encoders(published.version, self._embedder.model(), self._sparse.identity())
        sparse_query = self._sparse.encode_query(query)
        if not sparse_query.indices:
            return []
        dense_query = await self._embedder.embed_query(query)
        flt = _filter(filters)
        async with self._errors():
            response = await self._client.query_points(
                collection_name=self._name,
                prefetch=[
                    m.Prefetch(query=list(dense_query), using=DENSE, filter=flt, limit=limit),
                    m.Prefetch(
                        query=m.SparseVector(indices=list(sparse_query.indices), values=list(sparse_query.values)),
                        using=SPARSE,
                        filter=flt,
                        limit=limit,
                    ),
                ],
                query=m.FusionQuery(fusion=m.Fusion.RRF),
                limit=limit,
                with_payload=True,
                with_vectors=[DENSE],
            )
        hits = (
            chunk_from_payload(_payload(p), score=p.score, similarity=cosine(dense_query, _dense(p)))
            for p in response.points
        )
        return ordered(hits)[:limit]

    async def get(self, chunk_ids: Sequence[str]) -> list[IndexedChunk]:
        await self._require_published()
        async with self._errors():
            points = await self._client.retrieve(
                collection_name=self._name, ids=[chunk_point_id(c) for c in chunk_ids], with_payload=True
            )
        found = {chunk.chunk_id: chunk for chunk in (chunk_from_payload(_payload(p)) for p in points)}
        return [found[c] for c in chunk_ids if c in found]

    async def version(self) -> IndexVersion:
        return (await self._require_published()).version

    # --- DocumentStore ----------------------------------------------------------------------------------------

    async def original(self, chunk_id: str) -> Original | None:
        published = await self._require_published()
        async with self._errors():
            points = await self._client.retrieve(
                collection_name=published.name, ids=[chunk_point_id(chunk_id)], with_payload=True
            )
            if not points:
                return None
            chunk = chunk_from_payload(_payload(points[0]))
            documents = await self._client.retrieve(
                collection_name=published.name, ids=[document_point_id(chunk.doc_key)], with_payload=True
            )
        if not documents:
            raise IndexUnavailable(f"build {published.name} sem o registro do documento {chunk.doc_key}")
        return original_of(chunk, record_from_payload(_payload(documents[0])))

    # --- DocumentCatalog --------------------------------------------------------------------------------------

    async def catalog(self, *, filters: IndexFilters) -> list[CatalogEntry]:
        """Os pontos de documento do build publicado, filtrados aqui: são centenas, e o predicado é o da memória."""
        build = await self._require_published()
        documents = await self.documents(build.name)
        return [entry for entry in map(catalog_entry, documents) if matches(filters, entry)]

    # --- DocumentOutline --------------------------------------------------------------------------------------

    async def unit_kinds(self) -> tuple[UnitKind, ...]:
        return (await self._require_published()).version.unit_kinds

    async def units(self, kind: str, *, filters: IndexFilters) -> list[UnitEntry]:
        """Dos pontos de documento, como o catálogo: são centenas, e o predicado é o da memória."""
        build = await self._require_published()
        return units_of(await self.documents(build.name), kind, filters)

    async def outline(self, doc_key: str) -> list[UnitEntry]:
        build = await self._require_published()
        document = next((d for d in await self.documents(build.name) if d.doc_key == doc_key), None)
        return [] if document is None else outline_of(document)

    # --- IndexWriter ------------------------------------------------------------------------------------------

    async def published(self) -> Build | None:
        async with self._errors():
            aliases = (await self._client.get_aliases()).aliases
            collection = next((a.collection_name for a in aliases if a.alias_name == self._name), None)
            if collection is None:
                return None
            metadata = (await self._client.get_collection(collection)).config.metadata or {}
        if "index_version" not in metadata:
            raise IndexUnavailable(f"o build {collection} do índice {self._name!r} não tem versão gravada")
        return Build(name=collection, version=version_from_json(metadata["index_version"]))

    async def documents(self, build: str) -> list[StoredDocument]:
        await self._require_build(build)
        points = await self._scroll(build, _kind("document"), with_payload=m.PayloadSelectorExclude(exclude=["pages"]))
        return sorted((stored_from_payload(_payload(p)) for p in points), key=lambda d: d.doc_key)

    async def create(self, version: IndexVersion) -> Build:
        async with self._errors():
            existing = {c.name for c in (await self._client.get_collections()).collections}
            name = build_name(self._name, version.built_at, existing)
            await self._client.create_collection(
                collection_name=name,
                vectors_config={DENSE: m.VectorParams(size=version.embedding.dimensions, distance=m.Distance.COSINE)},
                sparse_vectors_config={SPARSE: m.SparseVectorParams(modifier=m.Modifier.IDF)},
                metadata={"index_version": version_to_json(version)},
            )
            with warnings.catch_warnings():  # o modo local avisa que índices de payload não têm efeito nele
                warnings.simplefilter("ignore")
                for field in KEYWORD_FIELDS:
                    await self._client.create_payload_index(name, field, field_schema=m.PayloadSchemaType.KEYWORD)
                for field in DATE_FIELDS:
                    await self._client.create_payload_index(name, field, field_schema=m.PayloadSchemaType.DATETIME)
        return Build(name=name, version=version)

    async def write(self, build: str, document: DocumentRecord, chunks: Sequence[EncodedChunk]) -> None:
        points = []
        for encoded in chunks:
            if encoded.chunk.doc_key != document.doc_key:
                raise ValueError(f"chunk {encoded.chunk.chunk_id} não é de {document.doc_key}")
            points.append(
                m.PointStruct(
                    id=chunk_point_id(encoded.chunk.chunk_id),
                    vector={
                        DENSE: list(encoded.dense),
                        SPARSE: m.SparseVector(
                            indices=list(encoded.sparse.indices), values=list(encoded.sparse.values)
                        ),
                    },
                    payload=chunk_to_payload(dataclasses.replace(encoded.chunk, score=None)),
                )
            )
        await self._upsert(build, [*points, _document_point(document)])

    async def copy(self, source: str, target: str, document: DocumentRecord) -> None:
        flt = m.Filter(must=[_match("kind", "chunk"), _match("doc_key", document.doc_key)])
        points = await self._scroll(source, flt, with_payload=True, with_vectors=True)
        copied = [
            m.PointStruct(
                id=p.id,
                vector=p.vector,  # type: ignore[arg-type]
                payload=chunk_to_payload(
                    with_document(chunk_from_payload(_payload(p)), document.fields, source_url=document.source_url)
                ),
            )
            for p in points
        ]
        await self._upsert(target, [*copied, _document_point(document)])

    async def publish(self, build: str) -> None:
        await self._require_build(build)
        async with self._errors():
            aliases = (await self._client.get_aliases()).aliases
            operations: list[Any] = []
            if any(a.alias_name == self._name for a in aliases):
                operations.append(m.DeleteAliasOperation(delete_alias=m.DeleteAlias(alias_name=self._name)))
            operations.append(
                m.CreateAliasOperation(create_alias=m.CreateAlias(collection_name=build, alias_name=self._name))
            )
            await self._client.update_collection_aliases(change_aliases_operations=operations)
            for collection in (await self._client.get_collections()).collections:
                if collection.name.startswith(f"{self._name}__") and collection.name != build:
                    await self._client.delete_collection(collection.name)

    async def discard(self, build: str) -> None:
        published = await self.published()
        if published is not None and published.name == build:
            raise ValueError(f"o build {build} está publicado; não se descarta o publicado")
        async with self._errors():
            if await self._client.collection_exists(build):
                await self._client.delete_collection(build)

    # --- internos -----------------------------------------------------------------------------------------------

    async def _require_published(self) -> Build:
        published = await self.published()
        if published is None:
            raise IndexUnavailable(f"índice {self._name!r} não existe (nenhum build publicado)")
        return published

    async def _require_build(self, build: str) -> None:
        async with self._errors():
            exists = await self._client.collection_exists(build)
        if not exists:
            raise KeyError(f"build inexistente: {build}")

    async def _scroll(self, build: str, flt: m.Filter, **options: Any) -> list[m.Record]:
        records: list[m.Record] = []
        offset: Any = None
        async with self._errors():
            while True:
                page, offset = await self._client.scroll(
                    collection_name=build, scroll_filter=flt, limit=256, offset=offset, **options
                )
                records.extend(page)
                if offset is None:
                    return records

    async def _upsert(self, build: str, points: list[m.PointStruct]) -> None:
        async with self._errors():
            for start in range(0, len(points), BATCH):
                await self._client.upsert(collection_name=build, points=points[start : start + BATCH], wait=True)

    @asynccontextmanager
    async def _errors(self) -> AsyncIterator[None]:
        """Falha de conexão ou do servidor vira `IndexUnavailable` (103 FR-007), nomeando o índice."""
        try:
            yield
        except (IndexUnavailable, KeyError, ValueError):
            raise
        except Exception as exc:
            raise IndexUnavailable(f"índice {self._name!r} indisponível: {type(exc).__name__}: {exc}") from exc


def _payload(point: Any) -> dict[str, Any]:
    payload = point.payload
    assert payload is not None
    return dict(payload)


def _dense(point: Any) -> list[float]:
    """O vetor denso devolvido pela busca (`with_vectors=["dense"]`); o Qdrant o guarda normalizado."""
    vectors = point.vector
    assert isinstance(vectors, dict) and DENSE in vectors
    return list(vectors[DENSE])


def _document_point(document: DocumentRecord) -> m.PointStruct:
    return m.PointStruct(id=document_point_id(document.doc_key), vector={}, payload=document_to_payload(document))


def _match(key: str, value: str) -> m.FieldCondition:
    return m.FieldCondition(key=key, match=m.MatchValue(value=value))


def _kind(kind: str) -> m.Filter:
    return m.Filter(must=[_match("kind", kind)])


def _empty(key: str) -> m.IsEmptyCondition:
    return m.IsEmptyCondition(is_empty=m.PayloadField(key=key))


def _filter(filters: IndexFilters) -> m.Filter:
    """A semântica de `domain/filters.py` (research R5) em condições do Qdrant."""
    must: list[Any] = [_match("kind", "chunk")]
    must_not: list[Any] = [_match("status", ValidityStatus.RASCUNHO.value)]
    if filters.branch is not None:  # os ramos efetivos do trecho; sem nenhum, geral (domain/filters.py)
        must.append(m.Filter(should=[_match(FILTER_BRANCHES, filters.branch), _empty(FILTER_BRANCHES)]))
    if filters.shape is not None:
        must.append(_match("shape", filters.shape))
    else:  # o sumário só entra quando pedido (domain/filters.py)
        must_not.append(_match("shape", "outline"))
    if filters.document_types:
        must.append(m.FieldCondition(key="document_type", match=m.MatchAny(any=list(filters.document_types))))
    if filters.bodies:
        must.append(m.FieldCondition(key="bodies", match=m.MatchAny(any=list(filters.bodies))))
    if filters.scopes:
        must.append(m.FieldCondition(key="scope", match=m.MatchAny(any=list(filters.scopes))))
    until, start = filters.effective_until, filters.effective_from
    if filters.previous_version:
        must += [
            m.Filter(must_not=[_empty("superseded_by")]),
            m.FieldCondition(key="expiry_date", range=m.DatetimeRange(lt=until)),
            m.Filter(
                should=[
                    _empty("successor_expiry_date"),
                    m.FieldCondition(key="successor_expiry_date", range=m.DatetimeRange(gte=until)),
                ]
            ),
        ]
    else:
        must.append(m.FieldCondition(key="effective_date", range=m.DatetimeRange(lte=until)))
        must_not.append(m.FieldCondition(key="expiry_date", range=m.DatetimeRange(lt=start)))
        if not filters.include_superseded:
            must_not.append(m.FieldCondition(key="expiry_date", range=m.DatetimeRange(lt=until)))
    return m.Filter(must=must, must_not=must_not)
