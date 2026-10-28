"""Seam *Índice do scouting* (spec 102): modelo de chunk indexado, filtros, versão do índice, `ChunkIndex` e a árvore de
unidades de cada documento (`DocumentOutline`, ADR-048).

Consumido por 103 e 106 (busca) e por 104 (valores de status). Importa só stdlib e os contratos — nunca o resto do
índice nem a ingestão (research R14, R18). O schema do Qdrant é detalhe do adapter.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Final, Literal, Protocol, runtime_checkable

from axon_ai_contracts import Chunk, JSONValue


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


_ID_SEPARATORS: Final = re.compile(r"[^a-z0-9]+")
SCOPES: Final = ("mundial", "interamericano", "nacional", "regional", "local")
"""A abrangência de um documento (o `nivel` da taxonomia; sem ele, a do catálogo; ADR-049)."""


def taxonomy_id(label: str) -> str:
    """O id de um rótulo da taxonomia (tipo, categoria, subcategoria, nível; ADR-049): sem acento nem caixa, os pontos
    fora e o resto separado por `_` — "P.O.R." → `por`, "Planejamento Estratégico" → `planejamento_estrategico`,
    "Conselho de Administração Nacional" → `conselho_de_administracao_nacional`."""
    decomposed = unicodedata.normalize("NFKD", label.casefold().replace(".", ""))
    plain = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return _ID_SEPARATORS.sub("_", plain).strip("_")


def _require_id(value: str, what: str) -> None:
    if not value or taxonomy_id(value) != value:
        raise ValueError(f"{what} precisa ser um id da taxonomia (minúsculas, dígitos e _): {value!r}")


BRANCHES: Final = ("filhotes", "lobinho", "escoteiro", "senior", "pioneiro")
"""Ids de ramo (os da 101), na ordem das idades; `filhotes` desde o ADR-045. Documento sem ramo é geral."""

UNIT_GROUP: Final = "unit_group"
"""Chave, nos metadados do `Chunk`, da unidade que reúne o trecho e os irmãos dele (ADR-048)."""
SHAPES: Final = ("prose", "list", "outline")
"""A forma do trecho (FR-027): itens marcados e curtos são `list`; o resto, `prose`; o sumário que a ingestão gera da
árvore do documento é `outline` — fora da busca comum, só quando pedido (ADR-048)."""
Shape = Literal["prose", "list", "outline"]

_ROUNDING: Final = 1e-6
_CHUNK_ID: Final = re.compile(r"^[a-z0-9][a-z0-9._~#@-]*$")


def _require_scope(scope: str | None) -> None:
    if scope is not None and scope not in SCOPES:
        raise ValueError(f"abrangência desconhecida: {scope!r} (esperado um de {', '.join(SCOPES)})")


def _require_branch(branch: str | None) -> None:
    if branch is not None and branch not in BRANCHES:
        raise ValueError(f"branch desconhecido: {branch!r} (esperado um de {', '.join(BRANCHES)})")


@dataclass(frozen=True, slots=True, kw_only=True)
class UnitRef:
    """Uma unidade no caminho de um trecho: id estável, tipo ("especialidade", "artigo"…) e título (ADR-048)."""

    unit_id: str
    kind: str
    title: str


@dataclass(frozen=True, slots=True, kw_only=True)
class IndexedChunk:
    """Um chunk do índice com os metadados do design (FR-006) e a vigência da versão (FR-007)."""

    chunk_id: str
    text: str
    doc_key: str
    document_id: str
    version: str
    document_type: str
    """O tipo do documento na taxonomia, como id (`ata`, `resolucao`, `por`…; ADR-049)."""
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
    category: str | None = None
    subcategory: str | None = None
    level: str | None = None
    semantic_type: str | None = None
    """Da entrada da taxonomia do acervo, pela pasta do arquivo (ADR-048): os rótulos, para exibir."""
    bodies: tuple[str, ...] = ()
    """Os órgãos do documento, como ids: a categoria e a subcategoria da taxonomia (ADR-049)."""
    scope: str | None = None
    """A abrangência (um de `SCOPES`; ADR-049)."""
    path: tuple[UnitRef, ...] = ()
    """As unidades acima do trecho, da raiz para baixo (ADR-048)."""
    unit_kind: str | None = None
    shape: Shape = "prose"
    branches: tuple[str, ...] = ()
    """Os ramos a que o trecho vale; vazio = os do documento (`branch`), e sem eles, geral."""
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
        for branch in self.branches:
            _require_branch(branch)
        if self.shape not in SHAPES:
            raise ValueError(f"shape inválido em {self.chunk_id}: {self.shape!r}")
        _require_id(self.document_type, f"document_type de {self.chunk_id}")
        _require_scope(self.scope)
        if self.similarity is not None and not -1.0 - _ROUNDING <= self.similarity <= 1.0 + _ROUNDING:
            raise ValueError(f"similarity fora de [-1, 1] em {self.chunk_id}: {self.similarity}")

    @property
    def unit_group(self) -> str | None:
        """A unidade que reúne o trecho e os irmãos dele: a mais funda do caminho que não é seção (ADR-048) — a
        especialidade dos trechos "Sugestão de temas" e "Fazer" dela; o próprio artigo, numa norma."""
        return next((ref.unit_id for ref in reversed(self.path) if ref.kind != "secao"), None)

    @property
    def effective_branches(self) -> tuple[str, ...]:
        """Os ramos do trecho, ou o do documento; vazio = geral."""
        return self.branches or ((self.branch,) if self.branch else ())

    @property
    def citation_section(self) -> str | None:
        """A etiqueta mais específica: artigo ou regra, senão subseção, seção, capítulo."""
        return self.article or self.subsection or self.section or self.chapter

    def to_chunk(self) -> Chunk:
        """O `Chunk` do contrato 001; `document` é o título do catálogo, que a citação mostra (research R14)."""
        metadata: dict[str, JSONValue] = {
            "document_id": self.document_id,
            "document_type": self.document_type,
            "semantic_type": self.semantic_type,
            "category": self.category,
            "branch": self.branch,
            "jurisdiction": self.jurisdiction,
            "chapter": self.chapter,
            "subsection": self.subsection,
            "page_end": self.page_end,
            "publication_date": _iso(self.publication_date),
            "effective_date": _iso(self.effective_date),
            "expiry_date": _iso(self.expiry_date),
            "superseded_by": self.superseded_by,
            "unit_kind": self.unit_kind,
            UNIT_GROUP: self.unit_group,
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
    - `branch`: o ramo ou documentos gerais (os ramos do trecho, senão o do documento);
    - `document_types`, `bodies`, `scopes`: ids da taxonomia (ADR-049) — tipo, órgão (categoria ou subcategoria) e
      abrangência; vazio = todos;
    - `shape`: só trechos `prose` ou `list`.
    """

    effective_from: date
    effective_until: date
    include_superseded: bool = False
    previous_version: bool = False
    branch: str | None = None
    document_types: tuple[str, ...] = field(default=())
    bodies: tuple[str, ...] = field(default=())
    scopes: tuple[str, ...] = field(default=())
    shape: Shape | None = None
    """Só trechos dessa forma (ADR-048); `None` = `prose` ou `list` (o sumário só entra quando pedido)."""

    def __post_init__(self) -> None:
        if self.shape is not None and self.shape not in SHAPES:
            raise ValueError(f"shape inválido: {self.shape!r}")
        if self.effective_from > self.effective_until:
            raise ValueError(
                f"effective_from ({self.effective_from}) depois de effective_until ({self.effective_until})"
            )
        _require_branch(self.branch)
        for value in (*self.document_types, *self.bodies):
            _require_id(value, "filtro da taxonomia")
        for scope in self.scopes:
            _require_scope(scope)


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
class UnitKind:
    """Um tipo de unidade: o nome, o rótulo do trecho e os plurais que uma pergunta usa (ADR-048)."""

    name: str
    label: str
    plurals: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class IndexVersion:
    """Estratégia de chunking + modelo de embedding + codificador esparso + data do build (FR-010)."""

    chunking: str
    embedding: EmbeddingModel
    sparse: str
    built_at: datetime
    unit_kinds: tuple[UnitKind, ...] = ()
    """Os tipos de unidade do build (os do `estrutura.yaml` na ingestão); não decidem a compatibilidade."""

    def __post_init__(self) -> None:
        if self.built_at.tzinfo is None:
            raise ValueError("built_at precisa de fuso (UTC)")

    def compatible_with(self, other: IndexVersion) -> bool:
        """Mesma codificação e mesmos chunks; a data do build não conta."""
        return (self.chunking, self.embedding, self.sparse) == (other.chunking, other.embedding, other.sparse)

    def describe(self) -> str:
        return f"{self.chunking} · {self.embedding.describe()} · {self.sparse}"


@dataclass(frozen=True, slots=True, kw_only=True)
class CatalogEntry:
    """Um documento do acervo, como o catálogo da ingestão o registrou: identidade, vigência e o link do original
    (ADR-047). Os campos de vigência têm os nomes dos de `IndexedChunk`, e os mesmos `IndexFilters` decidem."""

    doc_key: str
    document_id: str
    version: str
    document_type: str
    document_title: str
    jurisdiction: str
    effective_date: date
    status: ValidityStatus
    branch: str | None = None
    publication_date: date | None = None
    expiry_date: date | None = None
    superseded_by: str | None = None
    successor_expiry_date: date | None = None
    category: str | None = None
    subcategory: str | None = None
    level: str | None = None
    semantic_type: str | None = None
    bodies: tuple[str, ...] = ()
    scope: str | None = None
    source_url: str | None = None

    def __post_init__(self) -> None:
        if not _CHUNK_ID.match(self.doc_key):
            raise ValueError(f"doc_key inválido: {self.doc_key!r}")
        if not self.document_title.strip():
            raise ValueError(f"documento {self.doc_key} sem título")
        _require_branch(self.branch)


@dataclass(frozen=True, slots=True, kw_only=True)
class UnitNode:
    """Um nó da árvore de um documento, como a ingestão o gravou: `parent` é o id do nó de cima (`None` na raiz) e
    `branches`, os ramos a que a unidade vale (vazio = os do documento)."""

    unit_id: str
    kind: str
    title: str
    page_start: int
    page_end: int
    parent: str | None = None
    branches: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.page_start < 1 or self.page_end < self.page_start:
            raise ValueError(f"faixa de páginas inválida na unidade {self.unit_id}")
        for branch in self.branches:
            _require_branch(branch)


@dataclass(frozen=True, slots=True, kw_only=True)
class UnitEntry:
    """Uma unidade com o caminho até a raiz (sem ela) e o documento em que está."""

    unit: UnitNode
    path: tuple[UnitRef, ...]
    document: CatalogEntry


class IndexUnavailable(Exception):
    """Índice inexistente, servidor fora ou build sem versão. A mensagem nomeia o índice, nunca a credencial."""


class IndexMismatch(Exception):
    """A codificação desta instância difere da do build publicado (FR-012)."""

    def __init__(self, *, index: str, encoder: str) -> None:
        self.index = index
        self.encoder = encoder
        super().__init__(
            f"o índice publicado foi construído com {index}, mas esta instância codifica com {encoder}; "
            "rode `axon-scouting-corpus reindex` ou monte o pack com a codificação do índice"
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


@runtime_checkable
class DocumentCatalog(Protocol):
    """O acervo documento a documento (seam da 102, ADR-047), consumido pela lista de documentos da 103."""

    async def catalog(self, *, filters: IndexFilters) -> list[CatalogEntry]:
        """Os documentos que `filters` admite — a mesma semântica da busca —, em ordem de `doc_key`; `IndexUnavailable`
        sem build publicado."""
        ...


@runtime_checkable
class DocumentOutline(Protocol):
    """A árvore de unidades dos documentos (seam da 102, ADR-048), consumida pela enumeração da 103 e pela 101."""

    async def unit_kinds(self) -> tuple[UnitKind, ...]:
        """Os tipos de unidade do build publicado; `IndexUnavailable` sem build."""
        ...

    async def units(self, kind: str, *, filters: IndexFilters) -> list[UnitEntry]:
        """As unidades de `kind` em documentos que `filters` admite — com os ramos da unidade —, na ordem do documento
        (por `doc_key`, depois a da árvore). Tipo desconhecido → []."""
        ...

    async def outline(self, doc_key: str) -> list[UnitEntry]:
        """A árvore do documento em pré-ordem, sem texto; documento inexistente → []."""
        ...
