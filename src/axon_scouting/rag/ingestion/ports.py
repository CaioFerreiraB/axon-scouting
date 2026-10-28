"""Portas secundárias da ingestão (spec 102, research R16): o parser de documentos e o relógio.

As portas de codificação e de escrita são do índice (`rag/index/ports.py`). O catálogo não é porta: a CLI o lê e
entrega o objeto ao serviço.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True, kw_only=True)
class ParsedDocument:
    """O texto bruto de cada página (a primeira é a 1) e a identidade da fonte e do parser.

    `reading` é o texto de cada página na ordem de leitura (`domain/layout.py`), quando o parser conhece a posição das
    linhas; vazio, vale `pages`. O original preservado (FR-009) é sempre `pages`.
    """

    source_sha256: str
    pages: tuple[str, ...]
    parser: str
    reading: tuple[str, ...] = ()
    bookmarks: tuple[tuple[int, str, int], ...] = ()
    """Os marcadores de navegação do PDF: (nível, título, página), o nível 0 no topo (ADR-048)."""
    headings: tuple[tuple[int, str, int], ...] = ()
    """Os títulos pela tipografia (ADR-050): (página, texto com os espaços colapsados, nível); vazio quando o documento
    não tem estilo de título ou o parser não lê fonte (a transcrição de OCR)."""

    @property
    def for_chunking(self) -> tuple[str, ...]:
        return self.reading or self.pages


@runtime_checkable
class DocumentParser(Protocol):
    def parse(self, path: Path) -> ParsedDocument:
        """Levanta `FileNotFoundError` se o arquivo não existe. Documento sem texto volta com páginas vazias."""
        ...


@runtime_checkable
class Clock(Protocol):
    def now(self) -> datetime:
        """Instante atual em UTC (com fuso)."""
        ...


@runtime_checkable
class TokenCounter(Protocol):
    """Os tokens que o embedder lê de um trecho (ADR-050): o teto da divisão. A identidade entra na versão do índice."""

    def identity(self) -> str: ...

    def count(self, text: str) -> int: ...


class EstimatedTokens:
    """`TokenCounter` sem tokenizador: a estimativa do core (`⌈caracteres/4⌉`), para as suítes."""

    def identity(self) -> str:
        return "estimativa"

    def count(self, text: str) -> int:
        return -(-len(text) // 4)
