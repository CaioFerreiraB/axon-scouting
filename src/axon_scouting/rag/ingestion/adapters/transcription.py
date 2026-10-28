"""Transcrição de documento digitalizado e o parser que escolhe pela extensão (spec 102 FR-033; ADR-050).

A transcrição de um PDF sem camada de texto mora em `corpus/ocr/<caminho do original>.txt`, uma página por bloco,
separadas por quebra de página (U+000C, como o `pdftotext`). É texto no git: dá para revisar e corrigir linha a linha,
e o que ela diz é o que a ingestão lê — o original preservado (FR-009) passa a ser a transcrição, e o PDF digitalizado
continua no `fontes.yaml`. Sem posição nem fonte, a ordem de leitura é a do arquivo e não há títulos tipográficos.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Final

from axon_scouting.rag.ingestion.ports import DocumentParser, ParsedDocument

PAGE_BREAK: Final = "\f"
TRANSCRIPTION: Final = "transcricao-v1"
SUFFIX: Final = ".txt"


class TranscriptionParser:
    def parse(self, path: Path) -> ParsedDocument:
        data = path.read_bytes()
        text = data.decode("utf-8").replace("\r\n", "\n")
        pages = text.split(PAGE_BREAK)
        if pages and not pages[-1].strip():
            pages = pages[:-1]  # a quebra depois da última página
        return ParsedDocument(
            source_sha256=hashlib.sha256(data).hexdigest(),
            pages=tuple(pages),
            parser=TRANSCRIPTION,
        )


class ParserBySuffix:
    """Implementa `DocumentParser`: a transcrição pelo `.txt`, o resto pelo parser de PDF."""

    def __init__(self, pdf: DocumentParser, transcription: DocumentParser | None = None) -> None:
        self._pdf = pdf
        self._transcription = transcription or TranscriptionParser()

    def parse(self, path: Path) -> ParsedDocument:
        if path.suffix.lower() == SUFFIX:
            return self._transcription.parse(path)
        return self._pdf.parse(path)
