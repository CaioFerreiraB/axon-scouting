"""`PdfiumParser`: texto de cada página de um PDF com camada de texto (spec 102 research R6).

Devolve o texto bruto — o original preservado (FR-009) — e o `sha256` dos bytes. PDF sem camada de texto volta com
páginas vazias; quem rejeita é o serviço (OCR fora do escopo).
"""

from __future__ import annotations

import hashlib
from importlib.metadata import version
from pathlib import Path

import pypdfium2 as pdfium

from axon_scouting.rag.ingestion.ports import ParsedDocument


class PdfiumParser:
    def __init__(self) -> None:
        self._identity = f"pdfium-{version('pypdfium2')}"

    def parse(self, path: Path) -> ParsedDocument:
        data = path.read_bytes()
        document = pdfium.PdfDocument(data)
        pages: list[str] = []
        try:
            for index in range(len(document)):
                page = document[index]
                textpage = page.get_textpage()
                try:
                    pages.append(textpage.get_text_range())
                finally:
                    textpage.close()
                    page.close()
        finally:
            document.close()
        return ParsedDocument(source_sha256=hashlib.sha256(data).hexdigest(), pages=tuple(pages), parser=self._identity)
