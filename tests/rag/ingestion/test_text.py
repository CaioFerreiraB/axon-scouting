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
