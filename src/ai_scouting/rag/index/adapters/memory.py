"""Adapters em memória do índice, para testes (spec 102 research R16).

`MemoryIndex` implementa `ChunkIndex`, `IndexWriter` e `DocumentStore` como o adapter Qdrant: filtros pela função pura
do domínio, cosseno, IDF e RRF com as fórmulas do Qdrant. `HashingEmbedder` é um embedder determinístico — saco de
palavras com hash —, sem semântica: serve para filtros, ids e plumbing, não para medir qualidade.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, field

from ai_scouting.rag.index.api import (
    EmbeddingModel,
    IndexedChunk,
    IndexFilters,
    IndexUnavailable,
    IndexVersion,
)
from ai_scouting.rag.index.domain.filters import matches
from ai_scouting.rag.index.domain.points import build_name, stored, with_document
from ai_scouting.rag.index.domain.scoring import cosine, idf, ordered, rrf, sparse_score
from ai_scouting.rag.index.ports import (
    Build,
    DocumentRecord,
    Embedder,
    EncodedChunk,
    SparseEncoder,
    StoredDocument,
)

_WORD = re.compile(r"\w+")


class HashingEmbedder:
    """Embedder de teste: cada palavra dobrada soma 1 numa dimensão escolhida por hash; vetor normalizado."""

    def __init__(self, dimensions: int = 64, fingerprint: str = "hashing-v1") -> None:
        self._model = EmbeddingModel(name="hashing-test", fingerprint=fingerprint, dimensions=dimensions)

    def model(self) -> EmbeddingModel:
        return self._model

    async def embed_passages(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        return [self._embed("passage", text) for text in texts]

    async def embed_query(self, text: str) -> tuple[float, ...]:
        return self._embed("query", text)

    def _embed(self, role: str, text: str) -> tuple[float, ...]:
        vector = [0.0] * self._model.dimensions
        vector[_slot(f"__{role}__", self._model.dimensions)] = 0.5  # pergunta ≠ passagem, como no e5
        for word in _WORD.findall(_fold(text)):
            vector[_slot(word, self._model.dimensions)] += 1.0
        norm = math.sqrt(sum(x * x for x in vector))
        return tuple(x / norm for x in vector)


def _slot(word: str, dimensions: int) -> int:
    return int.from_bytes(hashlib.blake2b(word.encode(), digest_size=4).digest(), "big") % dimensions


def _fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


@dataclass
class _Build:
    version: IndexVersion
    chunks: dict[str, EncodedChunk] = field(default_factory=dict)
    documents: dict[str, DocumentRecord] = field(default_factory=dict)


@dataclass
class MemoryStorage:
    """O "servidor" em memória: builds e aliases, compartilháveis entre instâncias do índice."""

    builds: dict[str, _Build] = field(default_factory=dict)
    aliases: dict[str, str] = field(default_factory=dict)


class MemoryIndex:
    def __init__(
        self,
        index_name: str,
        embedder: Embedder,
        sparse: SparseEncoder,
        *,
        storage: MemoryStorage | None = None,
    ) -> None:
        self._name = index_name
        self._embedder = embedder
        self._sparse = sparse
        self.storage = storage if storage is not None else MemoryStorage()

    # --- ChunkIndex -------------------------------------------------------------------------------------------

    async def search(self, query: str, *, filters: IndexFilters, limit: int) -> list[IndexedChunk]:
        build = self._published_build()
        sparse_query = self._sparse.encode_query(query)
        if not sparse_query.indices:
            return []
        dense_query = await self._embedder.embed_query(query)
        candidates = [e for e in build.chunks.values() if matches(filters, e.chunk)]

        dense = sorted(candidates, key=lambda e: (-cosine(dense_query, e.dense), e.chunk.chunk_id))[:limit]

        total = len(build.chunks)
        containing: dict[int, int] = {}
        for encoded in build.chunks.values():
            for term in encoded.sparse.indices:
                containing[term] = containing.get(term, 0) + 1
        weights = {term: idf(total, containing.get(term, 0)) for term in sparse_query.indices}
        scored = [(sparse_score(sparse_query, e.sparse, weights), e) for e in candidates]
        sparse = [e for s, e in sorted(scored, key=lambda p: (-p[0], p[1].chunk.chunk_id)) if s > 0][:limit]

        fused = rrf([[e.chunk.chunk_id for e in dense], [e.chunk.chunk_id for e in sparse]])
        hits = [dataclasses.replace(build.chunks[chunk_id].chunk, score=score) for chunk_id, score in fused.items()]
        return ordered(hits)[:limit]

    async def get(self, chunk_ids: Sequence[str]) -> list[IndexedChunk]:
        build = self._published_build()
        return [build.chunks[c].chunk for c in chunk_ids if c in build.chunks]

    async def version(self) -> IndexVersion:
        return self._published_build().version

    # --- IndexWriter ------------------------------------------------------------------------------------------

    async def published(self) -> Build | None:
        name = self.storage.aliases.get(self._name)
        return None if name is None else Build(name=name, version=self.storage.builds[name].version)

    async def documents(self, build: str) -> list[StoredDocument]:
        return [stored(r) for _, r in sorted(self._build(build).documents.items())]

    async def create(self, version: IndexVersion) -> Build:
        name = build_name(self._name, version.built_at, set(self.storage.builds))
        self.storage.builds[name] = _Build(version=version)
        return Build(name=name, version=version)

    async def write(self, build: str, document: DocumentRecord, chunks: Sequence[EncodedChunk]) -> None:
        target = self._build(build)
        for encoded in chunks:
            if encoded.chunk.doc_key != document.doc_key:
                raise ValueError(f"chunk {encoded.chunk.chunk_id} não é de {document.doc_key}")
            target.chunks[encoded.chunk.chunk_id] = EncodedChunk(
                chunk=dataclasses.replace(encoded.chunk, score=None), dense=encoded.dense, sparse=encoded.sparse
            )
        target.documents[document.doc_key] = document

    async def copy(self, source: str, target: str, document: DocumentRecord) -> None:
        origin, destination = self._build(source), self._build(target)
        for encoded in origin.chunks.values():
            if encoded.chunk.doc_key == document.doc_key:
                destination.chunks[encoded.chunk.chunk_id] = EncodedChunk(
                    chunk=with_document(encoded.chunk, document.fields), dense=encoded.dense, sparse=encoded.sparse
                )
        destination.documents[document.doc_key] = document

    async def publish(self, build: str) -> None:
        self._build(build)
        self.storage.aliases[self._name] = build
        prefix = f"{self._name}__"
        for name in [n for n in self.storage.builds if n.startswith(prefix) and n != build]:
            del self.storage.builds[name]

    async def discard(self, build: str) -> None:
        if self.storage.aliases.get(self._name) == build:
            raise ValueError(f"o build {build} está publicado; não se descarta o publicado")
        self.storage.builds.pop(build, None)

    # --- internos -----------------------------------------------------------------------------------------------

    def _published_build(self) -> _Build:
        name = self.storage.aliases.get(self._name)
        if name is None:
            raise IndexUnavailable(f"índice {self._name!r} não existe (nenhum build publicado)")
        return self.storage.builds[name]

    def _build(self, name: str) -> _Build:
        try:
            return self.storage.builds[name]
        except KeyError:
            raise KeyError(f"build inexistente: {name}") from None
