"""Janelas narrativas: cabeçalhos abrem seção, parágrafos são agrupados até o máximo (spec 102 FR-004; research R8).

Tamanho em tokens estimados (`⌈caracteres/4⌉`, a estimativa do core), para não acoplar o chunking ao tokenizador do
embedder. Os números são provisórios e entram na identidade da estratégia — e, portanto, na versão do índice
(princípio X).
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Final

from axon_scouting.rag.ingestion.domain.ids import slug
from axon_scouting.rag.ingestion.domain.text import Line

_CHAPTER: Final = re.compile(r"^CAP[IÍ]TULO\s+(\d+|[IVXLCDM]+)\b", re.IGNORECASE)
_NUMBERED: Final = re.compile(r"^(\d+(?:\.\d+){0,3})\.?\s+(\S.*)$")
_TERMINAL: Final = (".", ";", ":", ",", "!", "?")
_PARAGRAPH_END: Final = (".", "!", "?", ":", ";")
_ITEM: Final = re.compile(r"^(?:[a-z]\)|§|[IVXLC]+\s*[–—-])")
"""Alínea (`a)`), parágrafo de norma (`§`) e inciso (`I –`) começam parágrafo mesmo sem maiúscula."""
MAX_NUMBERED_HEADING: Final = 80
MAX_CAPS_HEADING: Final = 60
MIN_BODY_LETTERS: Final = 40
"""Corpo com menos letras que isso e sem nenhuma frase terminada ("O Marco Simbólico" entre dois cabeçalhos) não vira
trecho: segue para o corpo da seção seguinte (research R28). Frase curta de verdade ("Use o lenço.") fica."""


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

    window_min: int = 250
    window_max: int = 450
    strategy: str = "estrutural-v2"

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
    carried: list[Line] = []

    def flush(*, last: bool = False) -> None:
        nonlocal unnamed, key, carried
        if not any(line.text for line in body):
            return
        if is_stub(body) and not last:
            carried = [*carried, *body]  # sem corpo que valha um trecho: vai com a seção seguinte
            return
        body[:0], carried = carried, []
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
        if path and not any(item.text for item in body) and _continues_heading(path[max(path)], line.text, heading):
            current = path[max(path)]
            path[max(path)] = f"{current} – {line.text}" if _CHAPTER.fullmatch(current) else f"{current} {line.text}"
            continue
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
    if carried:  # sobra curta no fim do texto: entra no último trecho, ou vira o único
        if chunks:
            last = chunks[-1]
            chunks[-1] = replace(last, body=f"{last.body}\n{paragraph_text(carried)}", page_end=carried[-1].page)
        else:
            body, carried = carried, []
            flush(last=True)
    return chunks


def is_stub(lines: Sequence[Line]) -> bool:
    """Poucas letras e nenhuma frase terminada: um título solto, não um trecho (research R28)."""
    letters = sum(ch.isalpha() for line in lines for ch in line.text)
    return letters < MIN_BODY_LETTERS and not _has_sentence("\n".join(line.text for line in lines))


def _continues_heading(current: str, text: str, heading: _Heading | None) -> bool:
    """Linha logo depois de um título, antes de qualquer corpo, que é parte dele: o nome do capítulo depois de
    "CAPÍTULO 3", ou a segunda linha de um título em maiúsculas (que pode terminar em vírgula)."""
    if not text or len(text) > MAX_NUMBERED_HEADING or (heading is not None and heading.key is not None):
        return False
    if _CHAPTER.fullmatch(current):
        return text[:1].isupper() and not text.endswith(_PARAGRAPH_END)
    letters = [ch for ch in current if ch.isalpha()]
    caps = bool(letters) and all(ch.isupper() for ch in letters)
    line_letters = [ch for ch in text if ch.isalpha()]
    return caps and len(line_letters) >= 3 and all(ch.isupper() for ch in line_letters) and not text.endswith(".")


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


def _word_slices(piece: _Piece, window_max: int) -> list[_Piece]:
    """A frase como está, ou, maior que a janela, em fatias de palavras inteiras que cabem nela."""
    if estimate_tokens(piece.text) <= window_max:
        return [piece]
    slices: list[_Piece] = []
    words: list[str] = []
    for word in piece.text.split():
        if words and estimate_tokens(" ".join([*words, word])) > window_max:
            slices.append(replace(piece, text=" ".join(words), opens_paragraph=piece.opens_paragraph and not slices))
            words = []
        words.append(word)
    slices.append(replace(piece, text=" ".join(words), opens_paragraph=piece.opens_paragraph and not slices))
    return slices


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
    o parágrafo entra frase a frase. Frase não é cortada, salvo a maior que o máximo — uma lista de nomes sem
    pontuação, num relatório —, que se divide entre palavras (research R28).
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
        for sentence in _sentence_pieces(paragraph):
            for piece in _word_slices(sentence, policy.window_max):
                if current and not fits([*current, piece]):
                    flush()
                current.append(piece)
    flush()
    return _without_stub_windows(windows)


def _without_stub_windows(windows: list[tuple[str, int, int]]) -> list[tuple[str, int, int]]:
    """Janela sem frase e com poucas letras (a sobra do fim de uma seção longa) junta-se à anterior (research R28)."""
    merged: list[tuple[str, int, int]] = []
    for text, start, end in windows:
        if merged and sum(ch.isalpha() for ch in text) < MIN_BODY_LETTERS and not _has_sentence(text):
            previous, first, _ = merged[-1]
            merged[-1] = (f"{previous}\n{text}", first, end)
        else:
            merged.append((text, start, end))
    return merged


def _has_sentence(text: str) -> bool:
    return any(line.rstrip().endswith(_PARAGRAPH_END) for line in text.split("\n"))
