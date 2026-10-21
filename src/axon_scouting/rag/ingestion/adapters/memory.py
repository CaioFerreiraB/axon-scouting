"""Adapters em memória da ingestão, para testes: `MemoryParser` e `FixedClock`."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path

from axon_scouting.rag.ingestion.ports import ParsedDocument


class MemoryParser:
    """Páginas dadas por caminho; o `sha256` é o do texto das páginas, concatenado."""

    def __init__(self, documents: Mapping[Path | str, Sequence[str]] | None = None) -> None:
        self._documents = {Path(path): tuple(pages) for path, pages in (documents or {}).items()}

    def put(self, path: Path | str, pages: Sequence[str]) -> None:
        self._documents[Path(path)] = tuple(pages)

    def parse(self, path: Path) -> ParsedDocument:
        try:
            pages = self._documents[Path(path)]
        except KeyError:
            raise FileNotFoundError(path) from None
        digest = hashlib.sha256("\f".join(pages).encode()).hexdigest()
        return ParsedDocument(source_sha256=digest, pages=pages, parser="memory")


class FixedClock:
    def __init__(self, instant: datetime) -> None:
        if instant.tzinfo is None:
            raise ValueError("FixedClock precisa de um instante com fuso")
        self._instant = instant

    def now(self) -> datetime:
        return self._instant
