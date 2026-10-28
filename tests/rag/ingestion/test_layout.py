"""Ordem de leitura pela geometria das linhas (spec 102 US6, FR-017; research R26)."""

from __future__ import annotations

from pathlib import Path

from fixtures import pdfs

from axon_scouting.rag.ingestion.adapters.pdfium import READING, PdfiumParser
from axon_scouting.rag.ingestion.domain.layout import COLUMN_BREAK, TextLine, reading_order

LARGURA = 567.0


def linha(texto: str, topo: float, esquerda: float = 43.0, direita: float = 526.0) -> TextLine:
    return TextLine(text=texto, left=esquerda, bottom=topo - 10, right=direita, top=topo)


def corpo(*linhas_: tuple[str, float]) -> list[TextLine]:
    return [linha(texto, topo) for texto, topo in linhas_]


def test_titulo_desenhado_depois_do_corpo_volta_para_cima() -> None:
    """O POR: o rodapé, o corpo, e só no fim o título de cada regra — que fica acima do corpo dela."""
    linhas_ = [
        linha("58 CAPÍTULO 8 • Ramo Filhotes", 31, direita=181),
        *corpo(("I - Especialmente concebido…", 689), ("O lema do Ramo Filhotes é “Viver Juntos”.", 437)),
        linha("REGRA", 739, esquerda=52, direita=82),
        linha("053 ÊNFASE EDUCATIVA DO RAMO FILHOTES", 734, esquerda=52, direita=291),
        linha("REGRA", 486, esquerda=52, direita=82),
        linha("054 LEMA DOS FILHOTES", 479, esquerda=52, direita=201),
    ]
    assert reading_order(linhas_, LARGURA) == [
        "REGRA",
        "053 ÊNFASE EDUCATIVA DO RAMO FILHOTES",
        "I - Especialmente concebido…",
        "REGRA",
        "054 LEMA DOS FILHOTES",
        "O lema do Ramo Filhotes é “Viver Juntos”.",
        "58 CAPÍTULO 8 • Ramo Filhotes",
    ]


def test_mesma_faixa_fica_na_ordem_do_parser() -> None:
    marcador = TextLine(text="I -", left=43, bottom=682, right=55, top=689.5)
    texto = TextLine(text="Especialmente concebido…", left=57, bottom=679.7, right=526, top=689.7)
    assert reading_order([marcador, texto], LARGURA) == ["I -", "Especialmente concebido…"]


def test_linha_impressa_duas_vezes_no_mesmo_lugar_sai() -> None:
    cabecalho = linha("CAPÍTULO 3", 672, esquerda=14, direita=59)
    assert reading_order([cabecalho, linha("Texto.", 500), cabecalho], LARGURA) == ["CAPÍTULO 3", "Texto."]


def test_colunas_alinhadas_sao_tabela_e_ficam_na_ordem_do_parser() -> None:
    esquerda = [linha(f"esquerda {n}", 700 - 20 * n, esquerda=40, direita=270) for n in range(6)]
    direita = [linha(f"direita {n}", 700 - 20 * n, esquerda=300, direita=530) for n in range(6)]
    assert reading_order([*esquerda, *direita], LARGURA) == [line.text for line in [*esquerda, *direita]]


def test_linha_sem_posicao_mantem_a_ordem_do_parser() -> None:
    assert reading_order([linha("embaixo", 100), "sem caixa", linha("em cima", 700)], LARGURA) == [
        "embaixo",
        "sem caixa",
        "em cima",
    ]


def test_pdfium_entrega_a_ordem_de_leitura_e_preserva_o_bruto(tmp_path: Path) -> None:
    path = pdfs.positioned_pdf(
        tmp_path / "regras.pdf", [(60, "O corpo da regra vem primeiro no arquivo."), (40, "REGRA 001 TITULO")]
    )
    parsed = PdfiumParser().parse(path)
    assert [line for line in parsed.reading[0].splitlines() if line.strip()] == [
        "REGRA 001 TITULO",
        "O corpo da regra vem primeiro no arquivo.",
    ]
    assert parsed.pages[0].index("O corpo") < parsed.pages[0].index("REGRA")  # o original é o bruto (FR-009)
    assert parsed.for_chunking == parsed.reading and READING in parsed.parser


def test_colunas_de_texto_sao_lidas_uma_a_uma() -> None:
    """ADR-050: o PDF desenha a coluna da direita antes; a leitura é a da esquerda inteira, uma fronteira, a da direita.
    As linhas das duas colunas não compartilham o topo (texto corrido, não tabela)."""
    direita = [linha(f"direita {n}", 690 - 20 * n, esquerda=300, direita=530) for n in range(6)]
    esquerda = [linha(f"esquerda {n}", 700 - 20 * n, esquerda=40, direita=270) for n in range(6)]
    rodape = linha("Escoteiros do Brasil - Região de São Paulo 33", 30, esquerda=40, direita=540)
    lido = reading_order([*direita, *esquerda, rodape], LARGURA)
    assert lido == [
        *(f"esquerda {n}" for n in range(6)),
        COLUMN_BREAK,
        *(f"direita {n}" for n in range(6)),
        COLUMN_BREAK,
        "Escoteiros do Brasil - Região de São Paulo 33",
    ]
