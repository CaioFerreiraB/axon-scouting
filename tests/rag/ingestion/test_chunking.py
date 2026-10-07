"""Janelas narrativas (spec 102 FR-004; research R8; T023 casos base, T043 propriedades)."""

from __future__ import annotations

from ai_scouting.rag.ingestion.domain.chunking import ChunkingPolicy, estimate_tokens, narrative_chunks
from ai_scouting.rag.ingestion.domain.text import Line


def linhas(*textos: str, pagina: int = 1) -> list[Line]:
    return [Line(text=t, page=pagina) for t in textos]


def paragrafo(n_palavras: int, palavra: str = "atividade") -> str:
    return " ".join(["Começo", *[palavra] * (n_palavras - 2)]) + " fim."


# --- casos base (US1, T023) -----------------------------------------------------------------------------------


def test_identidade_da_politica() -> None:
    assert ChunkingPolicy().identity() == "estrutural-v1;janela=300-600"
    assert estimate_tokens("a" * 9) == 3


def test_cabecalhos_numerados_abrem_secao_e_quebram_janela() -> None:
    texto = linhas(
        "4 Programa de Atividades",
        "O programa é dividido em ciclos.",
        "4.2 Atividades",
        "As atividades são planejadas pela patrulha.",
        "4.2.1 Acampamento",
        "O acampamento exige cardápio.",
    )
    chunks = narrative_chunks(texto, "Manual X", ChunkingPolicy())
    assert [c.section_key for c in chunks] == ["4", "4.2", "4.2.1"]
    assert chunks[0].prefix == "[Manual X · 4 Programa de Atividades]"
    assert chunks[1].prefix == "[Manual X · 4 Programa de Atividades · 4.2 Atividades]"
    assert chunks[2].prefix == "[Manual X · 4 Programa de Atividades · 4.2 Atividades · 4.2.1 Acampamento]"
    assert (chunks[2].chapter, chunks[2].section, chunks[2].subsection) == (
        "4 Programa de Atividades",
        "4.2 Atividades",
        "4.2.1 Acampamento",
    )
    assert chunks[1].body == "As atividades são planejadas pela patrulha."
    assert (
        chunks[1].text
        == "[Manual X · 4 Programa de Atividades · 4.2 Atividades]\nAs atividades são planejadas pela patrulha."
    )


def test_capitulo_e_titulo_em_maiusculas() -> None:
    texto = linhas("APRESENTAÇÃO", "Este manual orienta.", "CAPÍTULO 3 Vida em grupo", "A patrulha decide junto.")
    chunks = narrative_chunks(texto, "Manual X", ChunkingPolicy())
    assert [c.section_key for c in chunks] == ["s1", "cap3"]
    assert chunks[0].prefix == "[Manual X · APRESENTAÇÃO]"
    assert chunks[1].chapter == "CAPÍTULO 3 Vida em grupo"


def test_texto_antes_do_primeiro_cabecalho() -> None:
    chunks = narrative_chunks(
        linhas("Introdução sem título.", "4.2 Atividades", "Texto."), "Manual X", ChunkingPolicy()
    )
    assert [c.section_key for c in chunks] == ["s1", "4.2"]
    assert chunks[0].prefix == "[Manual X]"


def test_linha_com_ponto_final_nao_e_cabecalho() -> None:
    chunks = narrative_chunks(linhas("1 Escoteiro é leal.", "Mais texto."), "Manual X", ChunkingPolicy())
    assert [c.section_key for c in chunks] == ["s1"]


def test_paragrafos_agrupados_ate_o_maximo() -> None:
    politica = ChunkingPolicy()
    paragrafos = [paragrafo(150) for _ in range(6)]  # ~150 palavras ≈ 340 tokens estimados cada
    texto = [Line(text=p, page=1 + i // 2) for i, p in enumerate(paragrafos)]
    chunks = narrative_chunks(texto, "Manual X", politica)
    assert len(chunks) > 1
    assert all(estimate_tokens(c.body) <= politica.window_max for c in chunks)
    assert "\n".join(c.body for c in chunks) == "\n".join(paragrafos)


def test_paginas_da_janela() -> None:
    texto = [Line(text="Começa numa página e", page=2), Line(text="termina na outra.", page=3)]
    (chunk,) = narrative_chunks(texto, "Manual X", ChunkingPolicy())
    assert (chunk.page_start, chunk.page_end) == (2, 3)
    assert chunk.body == "Começa numa página e termina na outra."
