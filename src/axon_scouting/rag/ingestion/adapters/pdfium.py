"""`PdfiumParser`: texto de cada página de um PDF com camada de texto (spec 102 research R6, R26).

Devolve o texto bruto — o original preservado (FR-009) —, o `sha256` dos bytes e o texto na ordem de leitura: as linhas
do PDFium com a caixa de cada uma, ordenadas pelo domínio (`domain/layout.py`). PDF sem camada de texto volta com
páginas vazias; quem rejeita é o serviço (OCR fora do escopo).
"""

from __future__ import annotations

import ctypes
import hashlib
import math
import re
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from typing import Final

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c

from axon_scouting.rag.ingestion.domain.layout import COLUMN_BREAK, TextLine, ordered_lines
from axon_scouting.rag.ingestion.domain.typography import headings
from axon_scouting.rag.ingestion.ports import ParsedDocument

READING: Final = "leitura-v2"
"""Identidade da ordem de leitura: entra na do parser e, com ela, na chave de conteúdo de cada documento. v2: colunas
de texto lidas uma a uma (ADR-050)."""
TYPOGRAPHY: Final = "tipografia-v1"
"""Identidade da leitura de fonte e tamanho por linha (ADR-050)."""
_SAMPLES: Final = 3
"""Caracteres por linha lidos para o estilo: o primeiro, o do meio e o último com letra."""
_SUBSET: Final = re.compile(r"^[A-Z]{6}\+")
_BOLD: Final = re.compile(r"bold|black|heavy|semibold|demi", re.IGNORECASE)
BOOKMARKS: Final = "marcadores-v1"
"""Identidade da leitura dos bookmarks (ADR-048)."""
MAX_BOOKMARK_DEPTH: Final = 3
_BREAKS: Final = frozenset("\r\n")


class PdfiumParser:
    def __init__(self) -> None:
        self._identity = f"pdfium-{version('pypdfium2')};{READING};{BOOKMARKS};{TYPOGRAPHY}"

    def parse(self, path: Path) -> ParsedDocument:
        data = path.read_bytes()
        document = pdfium.PdfDocument(data)
        pages: list[str] = []
        reading: list[str] = []
        styled: list[list[TextLine]] = []
        try:
            bookmarks = _bookmarks(document)
            for index in range(len(document)):
                page = document[index]
                textpage = page.get_textpage()
                try:
                    raw = textpage.get_text_range()
                    pages.append(raw)
                    ordered = ordered_lines(_lines(textpage, raw), page.get_width())
                    reading.append("\n".join(line.text if isinstance(line, TextLine) else line for line in ordered))
                    styled.append([line for line in ordered if isinstance(line, TextLine)])
                finally:
                    textpage.close()
                    page.close()
        finally:
            document.close()
        return ParsedDocument(
            source_sha256=hashlib.sha256(data).hexdigest(),
            pages=tuple(pages),
            parser=self._identity,
            reading=tuple(reading),
            bookmarks=bookmarks,
            headings=tuple((h.page, h.text, h.level) for h in headings(styled)),
        )


def _bookmarks(document: pdfium.PdfDocument) -> tuple[tuple[int, str, int], ...]:
    """(nível, título, página) de cada marcador com destino; PDF com marcadores ilegíveis fica sem nenhum."""
    found: list[tuple[int, str, int]] = []
    try:
        for bookmark in document.get_toc(max_depth=MAX_BOOKMARK_DEPTH):
            destination = bookmark.get_dest()
            index = destination.get_index() if destination is not None else None
            if index is not None:
                found.append((bookmark.level, bookmark.get_title(), index + 1))
    except Exception:  # marcador é ajuda, nunca motivo de rejeição
        return ()
    return tuple(found)


def _lines(textpage: pdfium.PdfTextPage, raw: str) -> list[TextLine | str]:
    """As linhas do texto bruto (separadas pelas quebras que o PDFium gera), cada uma com a caixa dos seus caracteres.

    O índice de cada caractere do texto é o do PDFium (`count_chars`); caractere sem caixa (gerado) não conta, e linha
    sem nenhuma caixa volta como texto puro.
    """
    lines: list[TextLine | str] = []
    current: list[int] = []
    for index in [*range(min(len(raw), textpage.count_chars())), -1]:
        if index >= 0 and raw[index] not in _BREAKS:
            current.append(index)
            continue
        if current:
            lines.append(_line(textpage, raw, current))
            current = []
    return lines


def _line(textpage: pdfium.PdfTextPage, raw: str, indices: list[int]) -> TextLine | str:
    text = "".join(raw[index] for index in indices)
    boxes = [box for box in (textpage.get_charbox(index) for index in indices) if box[2] > box[0] and box[3] > box[1]]
    if not boxes or text == COLUMN_BREAK:
        return text
    size, font, bold = _style(textpage, raw, indices)
    return TextLine(
        text=text,
        left=min(box[0] for box in boxes),
        bottom=min(box[1] for box in boxes),
        right=max(box[2] for box in boxes),
        top=max(box[3] for box in boxes),
        size=size,
        font=font,
        bold=bold,
    )


def _style(textpage: pdfium.PdfTextPage, raw: str, indices: list[int]) -> tuple[float, str, bool]:
    """Tamanho efetivo (o da fonte vezes a escala da matriz do texto), família e negrito da maioria dos caracteres
    amostrados; linha ilegível volta com tamanho 0 (estilo desconhecido)."""
    letters = [index for index in indices if raw[index].isalpha()]
    if not letters:
        return 0.0, "", False
    picks = sorted({letters[0], letters[len(letters) // 2], letters[-1]})[:_SAMPLES]
    found: Counter[tuple[float, str, bool]] = Counter()
    matrix = pdfium_c.FS_MATRIX()
    name = ctypes.create_string_buffer(128)
    flags = ctypes.c_int()
    for index in picks:
        try:
            size = pdfium_c.FPDFText_GetFontSize(textpage.raw, index)
            if pdfium_c.FPDFText_GetMatrix(textpage.raw, index, ctypes.byref(matrix)):
                size *= math.sqrt(abs(matrix.a * matrix.d - matrix.b * matrix.c))
            pdfium_c.FPDFText_GetFontInfo(textpage.raw, index, name, len(name), ctypes.byref(flags))
            weight = pdfium_c.FPDFText_GetFontWeight(textpage.raw, index)
        except Exception:  # estilo é ajuda, nunca motivo de rejeição
            return 0.0, "", False
        family = _SUBSET.sub("", name.value.decode("utf-8", errors="ignore"))
        bold = weight >= 600 or bool(_BOLD.search(family))
        found[(round(size, 1), family, bold)] += 1
    return found.most_common(1)[0][0]
