"""Sumário, rodapé corrido e bookmarks como fontes da árvore (spec 102 FR-026; ADR-048)."""

from __future__ import annotations

from axon_scouting.rag.ingestion.domain.sources import Bookmark, bookmarks_from, toc_entries
from axon_scouting.rag.ingestion.domain.text import clean_pages, running_labels

CAPA = "Guia X\nEdição 2026"


def paginas_com_sumario(deslocamento: int) -> list[str]:
    """Capa, sumário e as páginas apontadas; o PDF tem `deslocamento` páginas antes da primeira numerada."""
    sumario = "SUMÁRIO\nComunicações . . . . . . . . 3\nGastronomia . . . . . . . . . 4\nViagens ---------------- 5"
    corpo = ["Corpo de comunicações.\nCOMUNICAÇÕES", "Corpo de gastronomia.\nGASTRONOMIA", "VIAGENS\nCorpo."]
    return [CAPA, sumario, *([""] * deslocamento), *corpo]


def test_sumario_com_pontilhado_e_tracejado_na_pagina_do_pdf() -> None:
    entradas = toc_entries(paginas_com_sumario(deslocamento=0))
    assert [(e.title, e.page) for e in entradas] == [("Comunicações", 3), ("Gastronomia", 4), ("Viagens", 5)]


def test_deslocamento_entre_pagina_impressa_e_do_pdf() -> None:
    entradas = toc_entries(paginas_com_sumario(deslocamento=2))
    assert [(e.title, e.page) for e in entradas] == [("Comunicações", 5), ("Gastronomia", 6), ("Viagens", 7)]


def test_sumario_que_nao_alinha_e_ignorado() -> None:
    paginas = paginas_com_sumario(deslocamento=0)
    paginas[2:] = ["Outro texto.", "Mais texto.", "Fim."]
    assert toc_entries(paginas) == ()


def test_a_pagina_de_sumario_sai_do_texto() -> None:
    linhas = clean_pages(paginas_com_sumario(deslocamento=0))
    assert not any("Gastronomia . ." in line.text or "----" in line.text for line in linhas)


def test_rodape_corrido_vira_rotulo_sem_o_numero() -> None:
    eixo = ["Habilidades para a Vida"] * 3 + ["Meio Ambiente"] * 3
    paginas = [
        f"{n} {rotulo}\nTexto da página {n}, com uma frase inteira.\nOutra frase da página."
        if n % 2
        else f"Texto da página {n}, com uma frase inteira.\nOutra frase da página.\n{rotulo} {n}"
        for n, rotulo in enumerate(eixo, start=1)
    ]
    rotulos = running_labels(paginas)
    assert rotulos[0] == ("Habilidades para a Vida",) and rotulos[1] == ("Habilidades para a Vida",)
    assert rotulos[3] == ("Meio Ambiente",)
    assert not any("Meio Ambiente" in line.text for line in clean_pages(paginas))  # o texto continua sem ele


def test_bookmarks_sem_titulo_ou_fora_do_documento_saem() -> None:
    marcadores = bookmarks_from([(0, " Juventudes ", 11), (1, "", 12), (0, "Fim", 99)], page_count=50)
    assert marcadores == (Bookmark(level=0, title="Juventudes", page=11),)


def test_ordinal_na_linha_de_cima_e_entrada_repetida() -> None:
    sumario = (
        "ÍNDICE\n1a\nReunião de Alcateia . . . . . . . 3\n2a\nReunião de Alcateia . . . . . . . 4\n"
        "Lusofonia . . . . . . . . 5\nLusofonia . . . . . . . . 6"
    )
    paginas = [CAPA, sumario, "1ª REUNIÃO\nReunião de Alcateia", "Reunião de Alcateia\nMais.", "Lusofonia\nTexto.", "x"]
    assert [(e.title, e.page) for e in toc_entries(paginas)] == [
        ("1a Reunião de Alcateia", 3),
        ("2a Reunião de Alcateia", 4),
        ("Lusofonia", 5),
    ]
