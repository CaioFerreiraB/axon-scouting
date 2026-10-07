"""PDFs sintéticos para os testes da ingestão (spec 102 research R16), gerados com `fpdf2` e fonte core.

A fonte core cobre latin-1 (acentos do português); travessão vira hífen. Cada página é uma lista de linhas; linhas
longas quebram sozinhas, como num PDF de verdade.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from fpdf import FPDF


def _pdf() -> FPDF:
    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=False)
    pdf.set_font("Helvetica", size=11)
    return pdf


def _latin1(text: str) -> str:
    return text.replace("–", "-").replace("—", "-").replace("·", "-")


def write_pdf(path: Path, pages: Sequence[Sequence[str]]) -> Path:
    """Um PDF com texto: uma página por item, uma linha (ou parágrafo) por string."""
    pdf = _pdf()
    for lines in pages:
        pdf.add_page()
        for line in lines:
            if line:
                pdf.multi_cell(0, 6, _latin1(line), new_x="LMARGIN", new_y="NEXT")
            else:
                pdf.ln(6)
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(path))
    return path


def empty_pdf(path: Path, pages: int = 2) -> Path:
    """Páginas sem camada de texto (como um PDF escaneado)."""
    pdf = _pdf()
    for _ in range(pages):
        pdf.add_page()
        pdf.rect(20, 20, 100, 100)
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(path))
    return path


NARRATIVE_PAGES: Sequence[Sequence[str]] = (
    (
        "APRESENTAÇÃO",
        "Este manual orienta a vida em patrulha e o planejamento das atividades do ramo.",
        "4 Programa de Atividades",
        "O programa é dividido em ciclos de planejamento, execução e avaliação, sempre com a patrulha.",
    ),
    (
        "4.2 Acampamento",
        "O acampamento de patrulha exige cardápio, lista de material e autorização dos responsáveis.",
        "A organização do campo é dividida entre os monitores e os demais membros da patrulha.",
    ),
)


def narrative_pdf(path: Path, pages: Sequence[Sequence[str]] = NARRATIVE_PAGES) -> Path:
    return write_pdf(path, pages)
