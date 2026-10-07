"""Suíte da porta `DocumentParser` contra `PdfiumParser` e `MemoryParser` (spec 102 research R6, R16; T024)."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

import pytest
from fixtures import pdfs

from ai_scouting.rag.ingestion.adapters.memory import MemoryParser
from ai_scouting.rag.ingestion.adapters.pdfium import PdfiumParser
from ai_scouting.rag.ingestion.ports import DocumentParser

PAGINAS = (("Primeira página com acentuação: ação, órgão.",), ("Segunda página.", "Outra linha."))

Factory = Callable[[Path, tuple[tuple[str, ...], ...]], tuple[DocumentParser, Path]]


def _pdfium(tmp: Path, pages: tuple[tuple[str, ...], ...]) -> tuple[DocumentParser, Path]:
    return PdfiumParser(), pdfs.write_pdf(tmp / "doc.pdf", pages)


def _memory(tmp: Path, pages: tuple[tuple[str, ...], ...]) -> tuple[DocumentParser, Path]:
    path = tmp / "doc.pdf"
    return MemoryParser({path: ["\n".join(lines) for lines in pages]}), path


@pytest.fixture(params=[_pdfium, _memory], ids=["pdfium", "memoria"])
def factory(request: pytest.FixtureRequest) -> Factory:
    return request.param  # type: ignore[no-any-return]


def test_uma_entrada_por_pagina_com_acentos(factory: Factory, tmp_path: Path) -> None:
    parser, path = factory(tmp_path, PAGINAS)
    documento = parser.parse(path)
    assert len(documento.pages) == 2
    assert "ação, órgão" in documento.pages[0]
    assert "Segunda página." in documento.pages[1]
    assert "Outra linha." in documento.pages[1]
    assert len(documento.source_sha256) == 64
    assert documento.parser


def test_arquivo_inexistente(factory: Factory, tmp_path: Path) -> None:
    parser, _ = factory(tmp_path, PAGINAS)
    with pytest.raises(FileNotFoundError):
        parser.parse(tmp_path / "nao-existe.pdf")


def test_pdfium_sha256_dos_bytes_e_identidade(tmp_path: Path) -> None:
    path = pdfs.write_pdf(tmp_path / "doc.pdf", PAGINAS)
    documento = PdfiumParser().parse(path)
    assert documento.source_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert documento.parser.startswith("pdfium-")


def test_pdfium_sem_camada_de_texto(tmp_path: Path) -> None:
    documento = PdfiumParser().parse(pdfs.empty_pdf(tmp_path / "scan.pdf", pages=3))
    assert len(documento.pages) == 3
    assert all(not page.strip() for page in documento.pages)
