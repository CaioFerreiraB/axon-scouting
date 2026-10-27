"""Limpeza do texto extraído (spec 102 research R7; T021)."""

from __future__ import annotations

from axon_scouting.rag.ingestion.domain.text import Line, clean_pages


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


# --- passada de 2026-10-27 (US6, research R27) ------------------------------------------------------------------


def test_cabecalho_corrido_em_bloco_de_varias_linhas_sai() -> None:
    """O Manual de Filhotes: "CAPÍTULO 3" e o nome do capítulo no topo de cada página, e o número embaixo."""
    nomes = ("um", "dois", "três", "quatro", "cinco")
    paginas = [
        "CAPÍTULO 3\nO Marco Simbólico\n" + "\n".join(f"Linha {c} da página {nome}." for c in "abcdefg") + f"\n{n}"
        for n, nome in enumerate(nomes, start=1)
    ]
    assert textos(clean_pages(paginas))[:2] == ["Linha a da página um.", "Linha b da página um."]


def test_titulo_do_capitulo_nas_paginas_dele_sai() -> None:
    """Corrido só no trecho do capítulo: 80% das páginas entre a primeira e a última ocorrência."""
    outras = [f"Abertura {n}.\nCorpo a{n}.\nCorpo b{n}." for n in range(1, 7)]
    capitulo = [f"As Atividades\nCorpo c{n}.\nCorpo d{n}." for n in range(1, 5)]
    assert "As Atividades" not in textos(clean_pages([*outras, *capitulo]))


def test_marcador_de_regra_no_topo_de_muitas_paginas_fica() -> None:
    paginas = [f"REGRA\n{n:03d} TITULO {n}\nCorpo da regra {n}.\nMais corpo {n}." for n in range(1, 7)]
    assert textos(clean_pages(paginas)).count("REGRA") == 6


def test_hifen_de_quebra_do_pdfium() -> None:
    assert textos(clean_pages(["Elementos que com\ufffepõem o marco."])) == ["Elementos que compõem o marco."]
    assert textos(clean_pages(["ATA DA UEB\ufffeRealizada no dia 3."])) == ["ATA DA UEB", "Realizada no dia 3."]
    assert textos(clean_pages(["Hífen sua\u00ad\nve no fim."])) == ["Hífen suave no fim."]


def test_numero_de_pagina_colado_ao_titulo_sai() -> None:
    assert textos(clean_pages(["30IDENTIDADE E SÍMBOLOS\nTexto."])) == ["IDENTIDADE E SÍMBOLOS", "Texto."]
    assert textos(clean_pages(["2026 foi um ano.\n3D Escotismo."])) == ["2026 foi um ano.", "3D Escotismo."]
