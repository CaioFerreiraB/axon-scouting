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
_ITEM: Final = re.compile(r"^(?:[a-z]\)|§|[IVXLC]+\s*[–—-])")
"""Alínea (`a)`), parágrafo de norma (`§`) e inciso (`I –`) começam parágrafo mesmo sem maiúscula."""
MAX_NUMBERED_HEADING: Final = 80
MAX_CAPS_HEADING: Final = 60


_BOUNDARY: Final = re.compile(r"[.!?…]+[\"”»)']*\s+(?=[A-ZÀ-ÖØ-Ý0-9\"“«(])")
ABBREVIATIONS: Final = frozenset(
    {
        "art",
        "arts",
        "nº",
        "n",
        "no",
        "sr",
        "sra",
        "srs",
        "dr",
        "dra",
        "p",
        "pp",
        "pág",
        "pag",
        "ex",
        "etc",
        "inc",
        "cap",
    }
)
"""Abreviações que terminam em ponto sem terminar a frase ("Art. 5", "nº. 3", "p. 12")."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ChunkingPolicy:
    """Faixa das janelas em tokens estimados. Configurável pela composição (`pack.py`, harness da 105), não pelo
    curador: mudar é outra versão do índice (research R8)."""

    window_min: int = 300
    window_max: int = 600
    strategy: str = "estrutural-v1"

    def __post_init__(self) -> None:
        if not 0 < self.window_min < self.window_max:
            raise ValueError(f"janela inválida: {self.window_min}-{self.window_max} (0 < mínimo < máximo)")

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
    starts: tuple[tuple[int, int], ...] = ()
    """(posição no texto, página) do começo de cada linha — dá a página de cada frase."""

    def page_at(self, offset: int) -> int:
        page = self.page_start
        for start, line_page in self.starts:
            if start > offset:
                break
            page = line_page
        return page


@dataclass(frozen=True, slots=True)
class _Piece:
    text: str
    page_start: int
    page_end: int
    opens_paragraph: bool


def narrative_chunks(
    lines: Sequence[Line],
    title: str,
    policy: ChunkingPolicy,
    *,
    section_key: str | None = None,
    context: Sequence[str] = (),
) -> list[ChunkDraft]:
    """Janelas de um texto narrativo. `section_key` fixa a chave da seção inicial (ex.: `preambulo`); `context` entra no
    prefixo depois do título (ex.: o anexo de uma norma)."""
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
        prefix = "[" + " · ".join([title, *context, *(path[level] for level in sorted(path))]) + "]"
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
        if heading is not None and heading.key is None and section_key is not None:
            heading = None  # numa seção de chave fixa (preâmbulo, anexo), linha em maiúsculas é texto
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


def paragraph_text(lines: Sequence[Line]) -> str:
    """As linhas unidas em parágrafos, um por linha do resultado."""
    return "\n".join(p.text for p in _paragraphs(lines))


def _join(lines: list[Line]) -> _Paragraph:
    starts, offset = [], 0
    for line in lines:
        starts.append((offset, line.page))
        offset += len(line.text) + 1
    return _Paragraph(
        text=" ".join(line.text for line in lines),
        page_start=lines[0].page,
        page_end=lines[-1].page,
        starts=tuple(starts),
    )


def split_sentences(text: str) -> list[str]:
    """Frases: pontuação final seguida de espaço e maiúscula, número ou aspas — salvo depois de abreviação."""
    sentences: list[str] = []
    start = 0
    for match in _BOUNDARY.finditer(text):
        before = text[start : match.start()].split()
        word = before[-1].casefold() if before else ""
        if text[match.start()] == "." and (word in ABBREVIATIONS or (len(word) == 1 and word.isalpha())):
            continue
        sentences.append(text[start : match.end()].strip())
        start = match.end()
    rest = text[start:].strip()
    if rest:
        sentences.append(rest)
    return sentences


def _sentence_pieces(paragraph: _Paragraph) -> list[_Piece]:
    pieces: list[_Piece] = []
    offset = 0
    for index, sentence in enumerate(split_sentences(paragraph.text)):
        begin = paragraph.text.find(sentence, offset)
        offset = begin + len(sentence)
        pieces.append(
            _Piece(
                text=sentence,
                page_start=paragraph.page_at(begin),
                page_end=paragraph.page_at(offset - 1),
                opens_paragraph=index == 0,
            )
        )
    return pieces


def _render(pieces: Sequence[_Piece]) -> str:
    parts: list[str] = []
    for index, piece in enumerate(pieces):
        if index:
            parts.append("\n" if piece.opens_paragraph else " ")
        parts.append(piece.text)
    return "".join(parts)


def _windows(paragraphs: list[_Paragraph], policy: ChunkingPolicy) -> list[tuple[str, int, int]]:
    """Empacotamento guloso até `window_max` tokens estimados, preferindo fronteira de parágrafo.

    Parágrafo que não cabe: se a janela já passou do mínimo, ela fecha ali; senão (ou se ele sozinho passa do máximo),
    o parágrafo entra frase a frase. Frase nunca é cortada; a frase única maior que o máximo fica inteira.
    """
    windows: list[tuple[str, int, int]] = []
    current: list[_Piece] = []

    def fits(pieces: Sequence[_Piece]) -> bool:
        return estimate_tokens(_render(pieces)) <= policy.window_max

    def flush() -> None:
        nonlocal current
        if current:
            windows.append((_render(current), current[0].page_start, current[-1].page_end))
            current = []

    for paragraph in paragraphs:
        whole = _Piece(paragraph.text, paragraph.page_start, paragraph.page_end, opens_paragraph=True)
        if fits([*current, whole]):
            current.append(whole)
            continue
        if current and estimate_tokens(_render(current)) >= policy.window_min:
            flush()
            if fits([whole]):
                current.append(whole)
                continue
        for piece in _sentence_pieces(paragraph):
            if current and not fits([*current, piece]):
                flush()
            current.append(piece)
    flush()
    return windows
