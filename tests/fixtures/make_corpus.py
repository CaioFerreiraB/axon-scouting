"""Corpus de demonstração do quickstart §4 (spec 102; T050): PDFs sintéticos e o `corpus.yaml`.

Uso: `uv run python tests/fixtures/make_corpus.py <dir>` — um manual em duas versões (2023 substituída pela 2026),
uma norma com o `Art. 42` na página 3 e um rascunho.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # roda como script: `tests/` no caminho de import

from fixtures import pdfs

CATALOGO = """\
documents:
  - file: manuais/manual-x-2023.pdf
    document_id: manual-x
    version: "2023"
    type: manual
    title: Manual X
    branch: pioneiro
    jurisdiction: nacional
    effective_date: 2023-01-01

  - file: manuais/manual-x-2026.pdf
    document_id: manual-x
    version: "2026"
    type: manual
    title: Manual X
    branch: pioneiro
    jurisdiction: nacional
    effective_date: 2026-03-01
    supersedes: "2023"

  - file: normas/norma-teste.pdf
    document_id: norma-teste
    version: "1"
    type: resolucao
    title: Norma de Teste
    jurisdiction: nacional
    effective_date: 2024-01-01

  - file: manuais/guia-rascunho.pdf
    document_id: guia-y
    version: "2027"
    type: guia
    title: Guia Y
    jurisdiction: nacional
    effective_date: 2027-01-01
    status: rascunho
"""

MANUAL_2023 = (
    ("APRESENTAÇÃO", "Edição de 2023 do manual do ramo, com o programa anterior."),
    ("4.2 Acampamento", "O acampamento exige lista de material e cardápio aprovado pela chefia."),
)
NORMA = (
    ("O Conselho resolve:", "CAPÍTULO I - Das Disposições Gerais", "Art. 1º Esta norma regula as atividades."),
    ("Art. 2º As atividades dependem de autorização.", "Art. 3º O grupo mantém registros."),
    (
        "CAPÍTULO II - Da Segurança",
        "Art. 42. O responsável pela atividade deverá comunicar os pais e registrar o plano de segurança.",
        "Art. 43. A autorização dos pais é obrigatória.",
    ),
)
RASCUNHO = (("Guia em elaboração.", "Este texto ainda não vale."),)


def main(directory: Path) -> None:
    pdfs.write_pdf(directory / "manuais" / "manual-x-2023.pdf", MANUAL_2023)
    pdfs.narrative_pdf(directory / "manuais" / "manual-x-2026.pdf")
    pdfs.write_pdf(directory / "normas" / "norma-teste.pdf", NORMA)
    pdfs.write_pdf(directory / "manuais" / "guia-rascunho.pdf", RASCUNHO)
    (directory / "corpus.yaml").write_text(CATALOGO, encoding="utf-8")
    print(directory / "corpus.yaml")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
