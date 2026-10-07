"""Janelas narrativas: cabeçalhos abrem seção, parágrafos são agrupados até o máximo (spec 102 FR-004; research R8).

Tamanho em tokens estimados (`⌈caracteres/4⌉`, a estimativa do core), para não acoplar o chunking ao tokenizador do
embedder. Os números são provisórios e entram na identidade da estratégia — e, portanto, na versão do índice
(princípio X).
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from ai_scouting.rag.ingestion.domain.ids import slug
from ai_scouting.rag.ingestion.domain.text import Line

_CHAPTER: Final = re.compile(r"^CAP[IÍ]TULO\s+(\d+|[IVXLCDM]+)\b", re.IGNORECASE)
_NUMBERED: Final = re.compile(r"^(\d+(?:\.\d+){0,3})\.?\s+(\S.*)$")
_TERMINAL: Final = (".", ";", ":", ",", "!", "?")
_PARAGRAPH_END: Final = (".", "!", "?", ":", ";")
MAX_NUMBERED_HEADING: Final = 80
MAX_CAPS_HEADING: Final = 60


@dataclass(frozen=True, slots=True, kw_only=True)
class ChunkingPolicy:
    window_min: int = 300
    window_max: int = 600
    strategy: str = "estrutural-v1"

    def identity(self) -> str:
        return f"{self.strategy};janela={self.window_min}-{self.window_max}"


@dataclass(frozen=True, slots=True, kw_only=True)
class ChunkDraft:
    """Um chunk antes do id e dos metadados de documento."""

    section_key: str
    prefix: str
    body: str
    page_start: int
    page_end: int
    chapter: str | None = None
    section: str | None = None
    subsection: str | None = None
    article: str | None = None
    normative: bool = False

    @property
    def text(self) -> str:
        return f"{self.prefix}\n{self.body}"


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / 4)


@dataclass(frozen=True, slots=True)
class _Heading:
    level: int  # 1 capítulo, 2 seção, 3 subseção
    key: str | None  # None: seção sem número (recebe `s<n>`)
    text: str


@dataclass(frozen=True, slots=True)
class _Paragraph:
    text: str
    page_start: int
    page_end: int


def narrative_chunks(
    lines: Sequence[Line], title: str, policy: ChunkingPolicy, *, section_key: str | None = None
) -> list[ChunkDraft]:
    """Janelas de um texto narrativo. `section_key` fixa a chave da seção inicial (ex.: `preambulo`)."""
    chunks: list[ChunkDraft] = []
    path: dict[int, str] = {}
    unnamed = 0
    key = section_key
    body: list[Line] = []

    def flush() -> None:
        nonlocal unnamed, key
        if not any(line.text for line in body):
            return
        if key is None:
            unnamed += 1
            key = f"s{unnamed}"
        prefix = "[" + " · ".join([title, *(path[level] for level in sorted(path))]) + "]"
        chunks.extend(
            ChunkDraft(
                section_key=key,
                prefix=prefix,
                body=window_body,
                page_start=start,
                page_end=end,
                chapter=path.get(1),
                section=path.get(2),
                subsection=path.get(3),
            )
            for window_body, start, end in _windows(_paragraphs(body), policy)
        )

    for line in lines:
        heading = _heading(line.text)
        if heading is None:
            body.append(line)
            continue
        flush()
        body = []
        for level in [lv for lv in path if lv >= heading.level]:
            del path[level]
        path[heading.level] = heading.text
        if heading.key is None:
            unnamed += 1
            key = f"s{unnamed}"
        else:
            key = heading.key
    flush()
    return chunks


def _heading(text: str) -> _Heading | None:
    if not text or text.endswith(_TERMINAL):
        return None
    if match := _CHAPTER.match(text):
        return _Heading(level=1, key=f"cap{slug(match.group(1))}", text=text)
    match = _NUMBERED.match(text)
    if match and len(text) <= MAX_NUMBERED_HEADING and match.group(2)[0].isupper():
        number = match.group(1)
        return _Heading(level=min(number.count(".") + 1, 3), key=number, text=text)
    letters = [ch for ch in text if ch.isalpha()]
    if len(text) <= MAX_CAPS_HEADING and len(letters) >= 3 and all(ch.isupper() for ch in letters):
        return _Heading(level=1, key=None, text=text)
    return None


def _starts_paragraph(text: str) -> bool:
    first = text[:1]
    return first.isupper() or first.isdigit() or first in '•-–—*"“(«'


def _paragraphs(lines: Sequence[Line]) -> list[_Paragraph]:
    paragraphs: list[_Paragraph] = []
    current: list[Line] = []
    for line in lines:
        if not line.text:
            if current:
                paragraphs.append(_join(current))
                current = []
            continue
        if current and current[-1].text.endswith(_PARAGRAPH_END) and _starts_paragraph(line.text):
            paragraphs.append(_join(current))
            current = []
        current.append(line)
    if current:
        paragraphs.append(_join(current))
    return paragraphs


def _join(lines: list[Line]) -> _Paragraph:
    return _Paragraph(text=" ".join(line.text for line in lines), page_start=lines[0].page, page_end=lines[-1].page)


def _windows(paragraphs: list[_Paragraph], policy: ChunkingPolicy) -> list[tuple[str, int, int]]:
    """Empacotamento guloso de parágrafos até `window_max` tokens estimados."""
    windows: list[tuple[str, int, int]] = []
    current: list[_Paragraph] = []
    for paragraph in paragraphs:
        candidate = "\n".join(p.text for p in [*current, paragraph])
        if current and estimate_tokens(candidate) > policy.window_max:
            windows.append(_window(current))
            current = []
        current.append(paragraph)
    if current:
        windows.append(_window(current))
    return windows


def _window(paragraphs: list[_Paragraph]) -> tuple[str, int, int]:
    return "\n".join(p.text for p in paragraphs), paragraphs[0].page_start, paragraphs[-1].page_end
