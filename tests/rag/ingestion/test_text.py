"""Limpeza do texto extraído (spec 102 research R7; T021)."""

from __future__ import annotations

from ai_scouting.rag.ingestion.domain.text import Line, clean_pages


def textos(linhas: list[Line]) -> list[str]:
    return [linha.text for linha in linhas if linha.text]


def test_numero_de_pagina_sai() -> None:
    paginas = ["Primeira linha.\n12", "- 13 -\nSegunda linha.", "Página 14\nTerceira.", "Terceira e meia.\n15 / 40"]
    assert textos(clean_pages(paginas)) == ["Primeira linha.", "Segunda linha.", "Terceira.", "Terceira e meia."]


def test_cabecalho_e_rodape_corridos_saem() -> None:
    paginas = [f"Manual do Ramo Pioneiro\nConteúdo {n}.\nEdição 2026" for n in range(1, 5)]
    assert textos(clean_pages(paginas)) == [f"Conteúdo {n}." for n in range(1, 5)]


def test_repeticao_em_minoria_ou_em_poucas_paginas_fica() -> None:
    poucas = ["Título\nA.", "Título\nB."]  # menos de 3 páginas
    assert textos(clean_pages(poucas)) == ["Título", "A.", "Título", "B."]
    minoria = ["Aviso\nA.", "Aviso\nB.", "C.", "D.", "E."]  # 2 de 5 páginas
    assert "Aviso" in textos(clean_pages(minoria))


def test_hifenizacao_na_quebra_de_linha() -> None:
    assert textos(clean_pages(["A organiza-\nção do grupo."])) == ["A organização do grupo."]
    assert textos(clean_pages(["O guarda-\nChuva fica."])) == ["O guarda-", "Chuva fica."]


def test_espacos_colapsados_e_pagina_de_origem() -> None:
    linhas = [linha for linha in clean_pages(["  Um   texto\t com  espaços  ", "", "Outra página."]) if linha.text]
    assert linhas == [Line(text="Um texto com espaços", page=1), Line(text="Outra página.", page=3)]


def test_linhas_vazias_marcam_paragrafo() -> None:
    linhas = clean_pages(["Primeiro parágrafo.\n\nSegundo parágrafo."])
    assert [linha.text for linha in linhas] == ["Primeiro parágrafo.", "", "Segundo parágrafo."]


# --- formatos do corpus real (SC-005, T053) ------------------------------------------------------------------------


def test_linha_de_indice_com_pontilhado_sai() -> None:
    paginas = [
        "ÍNDICE\nRegra 001 - Definição do Escotismo .................... 14\nRegra 002 - Propósito … … … 15\nTexto."
    ]
    assert textos(clean_pages(paginas)) == ["ÍNDICE", "Texto."]


def test_cabecalho_corrido_com_o_numero_da_pagina_sai() -> None:
    """O POR repete "<página> CAPÍTULO n • <nome>" no topo: muda por capítulo, mas traz o número da própria página."""
    conteudo = [f"Conteúdo {letra}." for letra in "abcdefghi"]
    paginas = [f"{n} CAPÍTULO 1 • Fundamentos do Escotismo\n{conteudo[n - 1]}" for n in range(1, 4)]
    paginas += [f"{n} CAPÍTULO 2 • Orientação Geral\n{conteudo[n - 1]}" for n in range(4, 10)]
    assert textos(clean_pages(paginas)) == conteudo


def test_titulo_numerado_no_topo_da_pagina_fica() -> None:
    """Começar com o número da página não basta: precisa se repetir em 3 páginas."""
    paginas = ["Texto.", "2 Programa de Atividades\nMais texto.", "Fim."]
    assert "2 Programa de Atividades" in textos(clean_pages(paginas))


def test_cabecalho_corrido_alterna_o_lado_do_numero() -> None:
    """Páginas pares "14 CAPÍTULO 1 • …", ímpares "CAPÍTULO 1 • … 15"."""
    conteudo = [f"Conteúdo {letra}." for letra in "abcd"]
    paginas = [
        (f"{n} CAPÍTULO 1 • Fundamentos" if n % 2 == 0 else f"CAPÍTULO 1 • Fundamentos {n}") + f"\n{conteudo[n - 1]}"
        for n in range(1, 5)
    ]
    assert textos(clean_pages(paginas)) == conteudo


def test_pagina_de_indice_sai_inteira() -> None:
    """Entrada longa do índice quebra a linha, e o pontilhado fica só na segunda: a página de índice sai inteira."""
    indice = "\n".join(
        [
            "CAPÍTULO 6 - VESTUÁRIO",
            "Regra 049 - Uniforme ................................ 40",
            "Regra 050 - Proíbe o uso de peças ou equipamentos de uso privado das forças",
            "armadas ............................................ 41",
            "Regra 051 - Lenço ................................... 42",
        ]
    )
    assert textos(clean_pages([indice, "Texto da página."])) == ["Texto da página."]


def test_credito_de_foto_colado_sai() -> None:
    """O POR cola o crédito da foto no título: "Fundamentos do Escotismo@ Guillerme Ferreira Heck"."""
    paginas = ["CAPÍTULO 1\nFundamentos do Escotismo@ Guillerme Ferreira Heck\nContato: grupo@escoteiros.org.br."]
    assert textos(clean_pages(paginas)) == [
        "CAPÍTULO 1",
        "Fundamentos do Escotismo",
        "Contato: grupo@escoteiros.org.br.",
    ]
