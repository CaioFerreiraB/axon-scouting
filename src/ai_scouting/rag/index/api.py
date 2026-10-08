"""Seam *Índice do scouting* (spec 102): modelo de chunk indexado, filtros, versão do índice e `ChunkIndex`.

Consumido por 103 e 106 (busca) e por 104 (valores de status). Importa só stdlib e os contratos — nunca o resto do
índice nem a ingestão (research R14, R18). O schema do Qdrant é detalhe do adapter.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Final, Protocol, runtime_checkable

from ai_core_contracts import Chunk, JSONValue


class ValidityStatus(StrEnum):
    """Posição da versão na linhagem (research R4). Se ela estava em vigor numa data decidem as datas, na busca."""

    VIGENTE = "vigente"
    REVOGADO = "revogado"
    SUBSTITUIDO = "substituido"
    RASCUNHO = "rascunho"


STATUS_LABELS: Final[Mapping[ValidityStatus, str]] = MappingProxyType(
    {
        ValidityStatus.VIGENTE: "vigente",
        ValidityStatus.REVOGADO: "revogado",
        ValidityStatus.SUBSTITUIDO: "substituído",
        ValidityStatus.RASCUNHO: "rascunho",
    }
)
"""Os status com acento, para quem os exibe (104)."""


class DocumentType(StrEnum):
    """Tabela fixa de tipos do pack (FR-002, clarificação 3); os ids são os do léxico da 101."""

    POR = "por"
    RESOLUCAO = "resolucao"
    ESTATUTO = "estatuto"
    MANUAL = "manual"
    PROGRAMA_EDUCATIVO = "programa_educativo"
    GUIA = "guia"


NORMATIVE_TYPES: Final = frozenset({DocumentType.POR, DocumentType.RESOLUCAO, DocumentType.ESTATUTO})
"""Divididos por artigo ou regra; os demais tipos, em janelas (FR-002)."""

BRANCHES: Final = ("lobinho", "escoteiro", "senior", "pioneiro")
"""Ids de ramo (os da 101). Documento sem ramo é geral."""

_ROUNDING: Final = 1e-6
_CHUNK_ID: Final = re.compile(r"^[a-z0-9][a-z0-9._~#@-]*$")


def _require_branch(branch: str | None) -> None:
    if branch is not None and branch not in BRANCHES:
        raise ValueError(f"branch desconhecido: {branch!r} (esperado um de {', '.join(BRANCHES)})")


@dataclass(frozen=True, slots=True, kw_only=True)
class IndexedChunk:
    """Um chunk do índice com os metadados do design (FR-006) e a vigência da versão (FR-007)."""

    chunk_id: str
    text: str
    doc_key: str
    document_id: str
    version: str
    document_type: DocumentType
    document_title: str
    jurisdiction: str
    page_start: int
    page_end: int
    effective_date: date
    status: ValidityStatus
    branch: str | None = None
    chapter: str | None = None
    section: str | None = None
    subsection: str | None = None
    article: str | None = None
    publication_date: date | None = None
    expiry_date: date | None = None
    superseded_by: str | None = None
    successor_expiry_date: date | None = None
    score: float | None = None
    similarity: float | None = None
    """Cosseno entre o vetor denso da pergunta e o do trecho; só em `search` (ADR-025)."""

    def __post_init__(self) -> None:
        if not _CHUNK_ID.match(self.chunk_id):
            raise ValueError(f"chunk_id inválido: {self.chunk_id!r} (só ASCII minúsculo, dígitos e -._~#@)")
        if not self.text.strip():
            raise ValueError(f"chunk {self.chunk_id} sem texto")
        if self.page_start < 1 or self.page_end < self.page_start:
            raise ValueError(f"faixa de páginas inválida em {self.chunk_id}: {self.page_start}–{self.page_end}")
        _require_branch(self.branch)
        if self.similarity is not None and not -1.0 - _ROUNDING <= self.similarity <= 1.0 + _ROUNDING:
            raise ValueError(f"similarity fora de [-1, 1] em {self.chunk_id}: {self.similarity}")

    @property
    def citation_section(self) -> str | None:
        """A etiqueta mais específica: artigo ou regra, senão subseção, seção, capítulo."""
        return self.article or self.subsection or self.section or self.chapter

    def to_chunk(self) -> Chunk:
        """O `Chunk` do contrato 001; `document` é o título do catálogo, que a citação mostra (research R14)."""
        metadata: dict[str, JSONValue] = {
            "document_id": self.document_id,
            "document_type": self.document_type.value,
            "branch": self.branch,
            "jurisdiction": self.jurisdiction,
            "chapter": self.chapter,
            "subsection": self.subsection,
            "page_end": self.page_end,
            "publication_date": _iso(self.publication_date),
            "effective_date": _iso(self.effective_date),
            "expiry_date": _iso(self.expiry_date),
            "superseded_by": self.superseded_by,
        }
        return Chunk(
            chunk_id=self.chunk_id,
            text=self.text,
            score=0.0 if self.score is None else self.score,
            document=self.document_title,
            version=self.version,
            status=self.status.value,
            section=self.citation_section,
            page=self.page_start,
            metadata=metadata,
        )


def _iso(value: date | None) -> str | None:
    return None if value is None else value.isoformat()


@dataclass(frozen=True, slots=True, kw_only=True)
class IndexFilters:
    """Filtros da busca (research R5). As datas são inclusivas; rascunho nunca entra.

    - em vigor em algum dia de [`effective_from`, `effective_until`];
    - `include_superseded=False`: além disso, ainda em vigor em `effective_until`;
    - `previous_version=True`: troca o critério de datas pela versão imediatamente substituída em relação a
      `effective_until`;
    - `branch`: o ramo ou documentos gerais; `document_types` vazio = todos.
    """

    effective_from: date
    effective_until: date
    include_superseded: bool = False
    previous_version: bool = False
    branch: str | None = None
    document_types: tuple[DocumentType, ...] = field(default=())

    def __post_init__(self) -> None:
        if self.effective_from > self.effective_until:
            raise ValueError(
                f"effective_from ({self.effective_from}) depois de effective_until ({self.effective_until})"
            )
        _require_branch(self.branch)
        object.__setattr__(self, "document_types", tuple(DocumentType(t) for t in self.document_types))


@dataclass(frozen=True, slots=True, kw_only=True)
class EmbeddingModel:
    """Identidade do modelo denso: nome, impressão digital dos pesos e dimensões (FR-010)."""

    name: str
    fingerprint: str
    dimensions: int

    def __post_init__(self) -> None:
        if not self.name or not self.fingerprint or self.dimensions < 1:
            raise ValueError("modelo de embedding precisa de nome, impressão digital e dimensões")

    def describe(self) -> str:
        return f"{self.name} (sha256 {self.fingerprint[:12]}, {self.dimensions} dimensões)"


@dataclass(frozen=True, slots=True, kw_only=True)
class IndexVersion:
    """Estratégia de chunking + modelo de embedding + codificador esparso + data do build (FR-010)."""

    chunking: str
    embedding: EmbeddingModel
    sparse: str
    built_at: datetime

    def __post_init__(self) -> None:
        if self.built_at.tzinfo is None:
            raise ValueError("built_at precisa de fuso (UTC)")

    def compatible_with(self, other: IndexVersion) -> bool:
        """Mesma codificação e mesmos chunks; a data do build não conta."""
        return (self.chunking, self.embedding, self.sparse) == (other.chunking, other.embedding, other.sparse)

    def describe(self) -> str:
        return f"{self.chunking} · {self.embedding.describe()} · {self.sparse}"


class IndexUnavailable(Exception):
    """Índice inexistente, servidor fora ou build sem versão. A mensagem nomeia o índice, nunca a credencial."""


class IndexMismatch(Exception):
    """A codificação desta instância difere da do build publicado (FR-012)."""

    def __init__(self, *, index: str, encoder: str) -> None:
        self.index = index
        self.encoder = encoder
        super().__init__(
            f"o índice publicado foi construído com {index}, mas esta instância codifica com {encoder}; "
            "rode `ai-scouting-corpus reindex` ou monte o pack com a codificação do índice"
        )


@runtime_checkable
class ChunkIndex(Protocol):
    """Porta de consulta do índice (seam da 102), consumida por 103 e 106."""

    async def search(self, query: str, *, filters: IndexFilters, limit: int) -> list[IndexedChunk]:
        """Busca densa + esparsa com fusão no armazenamento, já filtrada; no máximo `limit`, por score decrescente e,
        no empate, por `chunk_id`. Pergunta vazia ou só com termos de parada → []."""
        ...

    async def get(self, chunk_ids: Sequence[str]) -> list[IndexedChunk]:
        """Os chunks existentes, na ordem pedida, com `score=None`; ids ausentes são omitidos."""
        ...

    async def version(self) -> IndexVersion:
        """A versão do build publicado; `IndexUnavailable` se não há."""
        ...
