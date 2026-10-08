"""Portas secundárias do índice (spec 102, research R16): codificação, escrita e originais.

Ficam aqui, e não na ingestão, porque o `ChunkIndex` codifica a pergunta com o mesmo embedder da ingestão, e escrita,
leitura e originais são o mesmo armazenamento — os mesmos adapters as implementam.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Final, Protocol, runtime_checkable

from ai_scouting.rag.index.api import (
    DocumentType,
    EmbeddingModel,
    IndexedChunk,
    IndexVersion,
    ValidityStatus,
)

U32: Final = 2**32


@dataclass(frozen=True, slots=True)
class SparseVector:
    """Vetor esparso: índices u32 ordenados e únicos, um valor por índice."""

    indices: tuple[int, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.indices) != len(self.values):
            raise ValueError("indices e values precisam ter o mesmo tamanho")
        if any(not 0 <= i < U32 for i in self.indices):
            raise ValueError("índice esparso fora de u32")
        if any(a >= b for a, b in zip(self.indices, self.indices[1:], strict=False)):
            raise ValueError("índices esparsos precisam estar ordenados e sem repetição")


@dataclass(frozen=True, slots=True, kw_only=True)
class EncodedChunk:
    """Um chunk pronto para gravar: o modelo e as duas representações (FR-001)."""

    chunk: IndexedChunk
    dense: tuple[float, ...]
    sparse: SparseVector


@dataclass(frozen=True, slots=True, kw_only=True)
class DocumentFields:
    """Campos de documento de uma versão — os que todo chunk dela repete (catálogo + vigência, FR-006, FR-007)."""

    document_id: str
    version: str
    document_type: DocumentType
    document_title: str
    jurisdiction: str
    effective_date: date
    status: ValidityStatus
    branch: str | None = None
    publication_date: date | None = None
    expiry_date: date | None = None
    superseded_by: str | None = None
    successor_expiry_date: date | None = None

    @property
    def doc_key(self) -> str:
        return f"{self.document_id}-{self.version}"


@dataclass(frozen=True, slots=True, kw_only=True)
class StoredDocument:
    """O registro de uma versão no build, sem as páginas (listagem)."""

    fields: DocumentFields
    content_key: str
    source_file: str
    source_sha256: str
    chunk_ids: tuple[str, ...]
    source_url: str | None = None

    @property
    def doc_key(self) -> str:
        return self.fields.doc_key


@dataclass(frozen=True, slots=True, kw_only=True)
class DocumentRecord(StoredDocument):
    """O registro completo: com o texto original bruto de cada página (FR-009)."""

    pages: tuple[str, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class Build:
    """Um build do índice: nome e versão."""

    name: str
    version: IndexVersion


@dataclass(frozen=True, slots=True, kw_only=True)
class Original:
    """O texto original das páginas de um chunk (FR-009)."""

    chunk_id: str
    doc_key: str
    document_title: str
    version: str
    page_start: int
    page_end: int
    pages: tuple[str, ...]
    source_file: str
    source_sha256: str
    source_url: str | None = None


@runtime_checkable
class Embedder(Protocol):
    def model(self) -> EmbeddingModel: ...

    async def embed_passages(self, texts: Sequence[str]) -> list[tuple[float, ...]]: ...

    async def embed_query(self, text: str) -> tuple[float, ...]: ...


@runtime_checkable
class SparseEncoder(Protocol):
    def identity(self) -> str: ...

    def encode_passage(self, text: str) -> SparseVector: ...

    def encode_query(self, text: str) -> SparseVector: ...


@runtime_checkable
class IndexWriter(Protocol):
    async def published(self) -> Build | None:
        """O build atrás do alias, com a versão; `None` se o índice não existe."""
        ...

    async def documents(self, build: str) -> list[StoredDocument]: ...

    async def create(self, version: IndexVersion) -> Build:
        """Build novo e vazio, com a versão, chamado `<índice>__<built_at AAAAMMDDTHHMMSSZ>`; se o nome já existe,
        acrescenta `-2`, `-3`…"""
        ...

    async def write(self, build: str, document: DocumentRecord, chunks: Sequence[EncodedChunk]) -> None: ...

    async def copy(self, source: str, target: str, document: DocumentRecord) -> None:
        """Copia os chunks de `document.doc_key` com os vetores, reescrevendo os campos de documento a partir de
        `document`; grava o registro do documento em `target`."""
        ...

    async def publish(self, build: str) -> None:
        """Alias → `build` numa operação; depois apaga o build anterior e os órfãos do mesmo índice."""
        ...

    async def discard(self, build: str) -> None: ...


@runtime_checkable
class DocumentStore(Protocol):
    async def original(self, chunk_id: str) -> Original | None: ...
