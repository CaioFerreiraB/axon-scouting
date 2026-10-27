"""`PdfiumParser`: texto de cada página de um PDF com camada de texto (spec 102 research R6, R26).

Devolve o texto bruto — o original preservado (FR-009) —, o `sha256` dos bytes e o texto na ordem de leitura: as linhas
do PDFium com a caixa de cada uma, ordenadas pelo domínio (`domain/layout.py`). PDF sem camada de texto volta com
páginas vazias; quem rejeita é o serviço (OCR fora do escopo).
"""

from __future__ import annotations

import hashlib
from importlib.metadata import version
from pathlib import Path
from typing import Final

import pypdfium2 as pdfium

from axon_scouting.rag.ingestion.domain.layout import TextLine, reading_order
from axon_scouting.rag.ingestion.ports import ParsedDocument

READING: Final = "leitura-v1"
"""Identidade da ordem de leitura: entra na do parser e, com ela, na chave de conteúdo de cada documento."""
_BREAKS: Final = frozenset("\r\n")


class PdfiumParser:
    def __init__(self) -> None:
        self._identity = f"pdfium-{version('pypdfium2')};{READING}"

    def parse(self, path: Path) -> ParsedDocument:
        data = path.read_bytes()
        document = pdfium.PdfDocument(data)
        pages: list[str] = []
        reading: list[str] = []
        try:
            for index in range(len(document)):
                page = document[index]
                textpage = page.get_textpage()
                try:
                    raw = textpage.get_text_range()
                    pages.append(raw)
                    reading.append("\n".join(reading_order(_lines(textpage, raw), page.get_width())))
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
        )


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
    if not boxes:
        return text
    return TextLine(
        text=text,
        left=min(box[0] for box in boxes),
        bottom=min(box[1] for box in boxes),
        right=max(box[2] for box in boxes),
        top=max(box[3] for box in boxes),
    )
