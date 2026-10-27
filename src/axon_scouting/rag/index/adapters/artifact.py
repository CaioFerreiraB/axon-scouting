"""`ArtifactIndex`: o índice num artefato em disco, carregado em memória (spec 102; ADR-043, itens 1, 2 e 4).

Implementa `ChunkIndex`, `IndexWriter` e `DocumentStore` sobre o `MemoryIndex` — as mesmas fórmulas de busca (cosseno,
IDF, RRF) — e guarda cada build publicado num diretório: `manifest.json`, `chunks.jsonl` (os trechos com o vetor
esparso), `documents.jsonl` (os registros com o texto original) e `vectors/<modelo>.f32`, um conjunto de vetores densos
por modelo (formato em `domain/artifact.py`).

Cada instância codifica com um modelo: carrega só os vetores dele (ou, sem eles, os do primeiro conjunto, e a busca
recusa a mistura) e, ao publicar, grava os dele e preserva os dos outros modelos para os trechos cujo texto não mudou.
Um conjunto que deixaria algum trecho sem vetor sai do artefato. Publicar troca o diretório inteiro de uma vez.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from axon_scouting.rag.index.adapters.memory import MemoryIndex, MemoryStorage
from axon_scouting.rag.index.api import IndexedChunk, IndexFilters, IndexVersion
from axon_scouting.rag.index.domain.artifact import (
    CHUNKS,
    DOCUMENTS,
    MANIFEST,
    Manifest,
    ModelVectors,
    manifest_from_json,
    manifest_to_json,
    pack_vectors,
    unpack_vectors,
    vectors_file,
)
from axon_scouting.rag.index.domain.points import (
    chunk_from_payload,
    chunk_to_payload,
    document_to_payload,
    record_from_payload,
)
from axon_scouting.rag.index.ports import (
    Build,
    DocumentRecord,
    Embedder,
    EncodedChunk,
    Original,
    SparseEncoder,
    SparseVector,
    StoredDocument,
)


class ArtifactIndex:
    def __init__(self, root: Path, embedder: Embedder, sparse: SparseEncoder, *, name: str = "scouting") -> None:
        self._root = root
        self._name = name
        self._embedder = embedder
        self._memory = MemoryIndex(name, embedder, sparse, storage=MemoryStorage())
        self._manifest: Manifest | None = None
        self._stamp: tuple[int, int] | None = None

    def manifest(self) -> Manifest | None:
        """O manifesto publicado (para `status`), ou `None` sem artefato."""
        self._sync()
        return self._manifest

    async def export(self) -> tuple[IndexVersion, list[tuple[DocumentRecord, list[EncodedChunk]]]]:
        """O build publicado, documento a documento, com os vetores do modelo desta instância (para carregar o Qdrant a
        partir do artefato, sem PDF nem modelo)."""
        version = await self.version()
        assert self._manifest is not None
        _, chunks, documents = self._memory.storage.export(self._manifest.build)
        by_document: dict[str, list[EncodedChunk]] = {}
        for encoded in chunks:
            by_document.setdefault(encoded.chunk.doc_key, []).append(encoded)
        return version, [(record, by_document.get(record.doc_key, [])) for record in documents]

    # --- ChunkIndex e DocumentStore: o build publicado, recarregado quando o artefato muda --------------------------

    async def search(self, query: str, *, filters: IndexFilters, limit: int) -> list[IndexedChunk]:
        self._sync()
        return await self._memory.search(query, filters=filters, limit=limit)

    async def get(self, chunk_ids: Sequence[str]) -> list[IndexedChunk]:
        self._sync()
        return await self._memory.get(chunk_ids)

    async def version(self) -> IndexVersion:
        self._sync()
        return await self._memory.version()

    async def original(self, chunk_id: str) -> Original | None:
        self._sync()
        return await self._memory.original(chunk_id)

    # --- IndexWriter: builds em memória até publicar --------------------------------------------------------------

    async def published(self) -> Build | None:
        self._sync()
        return await self._memory.published()

    async def documents(self, build: str) -> list[StoredDocument]:
        self._sync()
        return await self._memory.documents(build)

    async def create(self, version: IndexVersion) -> Build:
        self._sync()
        return await self._memory.create(version)

    async def write(self, build: str, document: DocumentRecord, chunks: Sequence[EncodedChunk]) -> None:
        await self._memory.write(build, document, chunks)

    async def copy(self, source: str, target: str, document: DocumentRecord) -> None:
        await self._memory.copy(source, target, document)

    async def publish(self, build: str) -> None:
        self._sync()
        version, chunks, documents = self._memory.storage.export(build)
        self._write_artifact(build, version, chunks, documents)
        await self._memory.publish(build)
        self._sync()

    async def discard(self, build: str) -> None:
        await self._memory.discard(build)

    # --- disco ----------------------------------------------------------------------------------------------------

    def _sync(self) -> None:
        """Carrega o build publicado se o manifesto mudou desde a última leitura (outro processo pode ter publicado)."""
        path = self._root / MANIFEST
        try:
            status = path.stat()
        except FileNotFoundError:
            self._forget()
            return
        stamp = (status.st_mtime_ns, status.st_size)
        if stamp == self._stamp:
            return
        manifest = manifest_from_json(json.loads(path.read_text(encoding="utf-8")))
        if self._manifest is None or manifest.build != self._manifest.build:
            self._load(manifest)
        self._manifest, self._stamp = manifest, stamp

    def _forget(self) -> None:
        storage = self._memory.storage
        if self._manifest is not None and storage.aliases.get(self._name) == self._manifest.build:
            del storage.aliases[self._name]
            storage.builds.pop(self._manifest.build, None)
        self._manifest, self._stamp = None, None

    def _load(self, manifest: Manifest) -> None:
        payloads = _jsonl(self._root / CHUNKS, manifest.chunks_sha256)
        records = [record_from_payload(raw) for raw in _jsonl(self._root / DOCUMENTS, manifest.documents_sha256)]
        version = manifest.version_for(self._embedder.model())
        entry = manifest.vectors_of(version.embedding)
        assert entry is not None
        dense = unpack_vectors(_read(self._root / entry.file, entry.sha256), entry.model.dimensions, len(payloads))
        chunks = [
            EncodedChunk(chunk=chunk_from_payload(raw), dense=vector, sparse=_sparse(raw["sparse"]))
            for raw, vector in zip(payloads, dense, strict=True)
        ]
        storage = self._memory.storage
        if self._manifest is not None and storage.aliases.get(self._name) == self._manifest.build:
            storage.builds.pop(self._manifest.build, None)
        storage.put(manifest.build, version, chunks, records)
        storage.aliases[self._name] = manifest.build

    def _write_artifact(
        self, build: str, version: IndexVersion, chunks: Sequence[EncodedChunk], documents: Sequence[DocumentRecord]
    ) -> None:
        chunks_data = "".join(_line(_chunk_line(encoded)) for encoded in chunks).encode()
        documents_data = "".join(_line(document_to_payload(record)) for record in documents).encode()
        files: dict[str, bytes] = {CHUNKS: chunks_data, DOCUMENTS: documents_data}
        models: list[ModelVectors] = []
        own = pack_vectors([encoded.dense for encoded in chunks], version.embedding.dimensions)
        models.append(ModelVectors(model=version.embedding, file=vectors_file(version.embedding), sha256=_sha(own)))
        files[models[0].file] = own
        for entry, data in self._carried(version, chunks):
            models.append(entry)
            files[entry.file] = data
        manifest = Manifest(
            index=self._name,
            build=build,
            chunking=version.chunking,
            sparse=version.sparse,
            built_at=version.built_at,
            chunks_sha256=_sha(chunks_data),
            documents_sha256=_sha(documents_data),
            models=tuple(models),
        )
        files[MANIFEST] = (json.dumps(manifest_to_json(manifest), ensure_ascii=False, indent=2) + "\n").encode()

        staging = self._root.with_name(f".{self._root.name}.{build}.novo")
        shutil.rmtree(staging, ignore_errors=True)
        for relative, data in files.items():
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        previous = self._root.with_name(f".{self._root.name}.{build}.anterior")
        shutil.rmtree(previous, ignore_errors=True)
        if self._root.exists():
            self._root.rename(previous)
        staging.rename(self._root)
        shutil.rmtree(previous, ignore_errors=True)

    def _carried(self, version: IndexVersion, chunks: Sequence[EncodedChunk]) -> list[tuple[ModelVectors, bytes]]:
        """Os conjuntos de outros modelos do artefato publicado que ainda cobrem todo trecho do build novo — mesmo id e
        mesmo texto (o vetor denso depende só do texto)."""
        previous = self._manifest
        if previous is None:
            return []
        old = _jsonl(self._root / CHUNKS, previous.chunks_sha256)
        row_of = {raw["chunk_id"]: (row, raw["text"]) for row, raw in enumerate(old)}
        rows: list[int] = []
        for encoded in chunks:
            found = row_of.get(encoded.chunk.chunk_id)
            if found is None or found[1] != encoded.chunk.text:
                return []
            rows.append(found[0])
        carried: list[tuple[ModelVectors, bytes]] = []
        for entry in previous.models:
            if entry.model == version.embedding:
                continue
            vectors = unpack_vectors(_read(self._root / entry.file, entry.sha256), entry.model.dimensions, len(old))
            data = pack_vectors([vectors[row] for row in rows], entry.model.dimensions)
            carried.append((dataclasses.replace(entry, sha256=_sha(data)), data))
        return carried


def chunk_texts(root: Path) -> Callable[[str], str | None]:
    """O texto de cada trecho do artefato em `root`, por `chunk_id` — para a métrica de fatos da 105 (research R13).

    Lê `chunks.jsonl` só na primeira consulta, conferindo o sha256 do manifesto, e guarda só o texto: o pack declara a
    métrica em toda carga, mas só uma rodada de avaliação a consulta.
    """
    texts: dict[str, str] = {}
    lock = threading.Lock()
    loaded = False

    def text_of(chunk_id: str) -> str | None:
        nonlocal loaded
        with lock:
            if not loaded:
                manifest = manifest_from_json(json.loads((root / MANIFEST).read_text(encoding="utf-8")))
                texts.update((line["chunk_id"], line["text"]) for line in _jsonl(root / CHUNKS, manifest.chunks_sha256))
                loaded = True
        return texts.get(chunk_id)

    return text_of


def _chunk_line(encoded: EncodedChunk) -> dict[str, Any]:
    payload = chunk_to_payload(encoded.chunk)
    payload.pop("kind", None)
    payload["sparse"] = {"indices": list(encoded.sparse.indices), "values": list(encoded.sparse.values)}
    return payload


def _sparse(raw: dict[str, Any]) -> SparseVector:
    return SparseVector(tuple(int(i) for i in raw["indices"]), tuple(float(v) for v in raw["values"]))


def _line(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"


def _jsonl(path: Path, sha256: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in _read(path, sha256).decode().splitlines() if line]


def _read(path: Path, sha256: str) -> bytes:
    data = path.read_bytes()
    if _sha(data) != sha256:
        raise ValueError(f"{path.name}: sha256 diferente do registrado no manifesto (artefato corrompido)")
    return data


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
