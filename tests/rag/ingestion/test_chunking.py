"""Janelas narrativas (spec 102 FR-004; research R8; T023 casos base, T043 propriedades)."""

from __future__ import annotations

import random

import pytest

from axon_scouting.rag.ingestion.domain.chunking import (
    ChunkingPolicy,
    estimate_tokens,
    narrative_chunks,
    split_sentences,
)
from axon_scouting.rag.ingestion.domain.text import Line


def linhas(*textos: str, pagina: int = 1) -> list[Line]:
    return [Line(text=t, page=pagina) for t in textos]


def paragrafo(n_palavras: int, palavra: str = "atividade") -> str:
    return " ".join(["Começo", *[palavra] * (n_palavras - 2)]) + " fim."


# --- casos base (US1, T023) -----------------------------------------------------------------------------------


def test_identidade_da_politica() -> None:
    assert ChunkingPolicy().identity() == "estrutural-v5;janela=250-450"
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
    sem_juncao = ChunkingPolicy(window_min=1)  # nenhuma seção é pequena: cada título abre o seu trecho
    chunks = narrative_chunks(texto, "Manual X", sem_juncao)
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


# --- propriedades (US4, T043) ------------------------------------------------------------------------------------

PALAVRAS = [
    "patrulha",
    "atividade",
    "escoteiro",
    "tropa",
    "monitor",
    "acampamento",
    "cardápio",
    "material",
    "lenço",
    "jogo",
    "trilha",
    "fogo",
]


def frase(rng: random.Random) -> str:
    palavras = [rng.choice(PALAVRAS) for _ in range(rng.randint(4, 40))]
    return palavras[0].capitalize() + " " + " ".join(palavras[1:]) + "."


def texto_gerado(semente: int) -> list[Line]:
    """Seções com cabeçalho numerado e parágrafos de 1 a 12 frases, numa página por seção."""
    rng = random.Random(semente)
    linhas_: list[Line] = []
    for secao in range(1, rng.randint(2, 5)):
        linhas_.append(Line(text=f"{secao} Seção gerada", page=secao))
        for _ in range(rng.randint(1, 8)):
            linhas_.append(Line(text=" ".join(frase(rng) for _ in range(rng.randint(1, 12))), page=secao))
            linhas_.append(Line(text="", page=secao))
    return linhas_


def _normalizado(texto: str) -> str:
    return " ".join(texto.split())


@pytest.mark.parametrize("semente", range(40))
def test_propriedades_das_janelas(semente: int) -> None:
    politica = ChunkingPolicy()
    texto = texto_gerado(semente)
    chunks = narrative_chunks(texto, "Manual", politica)
    por_secao: dict[str, list[object]] = {}
    for chunk in chunks:
        por_secao.setdefault(chunk.section_key, []).append(chunk)
    for secao, janelas in por_secao.items():
        for i, janela in enumerate(janelas):
            corpo = janela.body  # type: ignore[attr-defined]
            assert corpo.endswith("."), (secao, corpo[-40:])  # nenhuma frase cortada
            tokens = estimate_tokens(corpo)
            assert tokens <= politica.window_max
            if i < len(janelas) - 1:  # só a última janela da seção pode ficar abaixo do mínimo
                assert tokens >= politica.window_min, (secao, i, tokens)
    corpo_total = _normalizado(" ".join(c.body for c in chunks))
    original = _normalizado(" ".join(linha.text for linha in texto if linha.text and not linha.text[0].isdigit()))
    assert corpo_total == original  # nada perdido, nada repetido (sem sobreposição)


def test_paragrafo_grande_divide_em_frases() -> None:
    frases = [f"A patrulha número {n} prepara o acampamento com cuidado e atenção." for n in range(80)]
    chunks = narrative_chunks([Line(text=" ".join(frases), page=1)], "Manual", ChunkingPolicy())
    assert len(chunks) > 1
    assert all(c.body.endswith("atenção.") for c in chunks)


def test_frase_maior_que_o_maximo_se_divide_entre_palavras() -> None:
    """Uma lista sem pontuação (nomes num relatório) não vira um trecho que o embedder truncaria (research R28)."""
    gigante = "A " + " ".join(f"patrulha{n}" for n in range(400)) + "."
    policy = ChunkingPolicy()
    chunks = narrative_chunks([Line(text=gigante, page=1)], "Manual", policy)
    assert len(chunks) > 1 and all(estimate_tokens(chunk.body) <= policy.window_max for chunk in chunks)
    assert " ".join(chunk.body for chunk in chunks) == gigante


@pytest.mark.parametrize(
    "texto",
    [
        "Conforme o Art. 5 do estatuto, a patrulha decide.",
        "A resolução nº. 3 regula o tema.",
        "O Sr. Silva e a Sra. Souza coordenam.",
        "Veja a p. 12 do manual.",
        "Leve material, ex. lanterna, corda etc. para o campo.",
    ],
)
def test_abreviacoes_nao_terminam_frase(texto: str) -> None:
    assert split_sentences(texto) == [texto]


def test_separa_frases() -> None:
    assert split_sentences("Primeira frase. Segunda frase! Terceira? «Quarta» frase. 5 itens.") == [
        "Primeira frase.",
        "Segunda frase!",
        "Terceira?",
        "«Quarta» frase.",
        "5 itens.",
    ]


def test_cabecalho_sempre_quebra() -> None:
    texto = linhas("1 Primeira", "Texto curto.", "2 Segunda", "Outro texto curto.")
    assert [c.body for c in narrative_chunks(texto, "M", ChunkingPolicy())] == ["Texto curto.", "Outro texto curto."]


def test_politica_valida() -> None:
    assert ChunkingPolicy(window_min=200, window_max=400).identity() == "estrutural-v5;janela=200-400"
    for minimo, maximo in ((400, 400), (500, 400), (0, 400), (-1, 10)):
        with pytest.raises(ValueError):
            ChunkingPolicy(window_min=minimo, window_max=maximo)


# --- passada de 2026-10-27 (US6, research R28) ------------------------------------------------------------------


def test_corpo_sem_frase_vai_para_a_secao_seguinte() -> None:
    """Nenhum trecho só com "O Marco Simbólico": o resto sem frase segue com o corpo da seção seguinte."""
    texto = linhas("1 Uniforme", "O Marco Simbólico", "2 Promessa", "A promessa é feita na cerimônia.")
    (chunk,) = narrative_chunks(texto, "Manual", ChunkingPolicy())
    assert chunk.section_key == "2" and chunk.body.startswith("O Marco Simbólico")


def test_capitulo_sem_corpo_nao_vira_trecho() -> None:
    texto = linhas("CAPÍTULO 3", "O Marco Simbólico", "CAPÍTULO 4", "Corpo do capítulo quatro com uma frase.")
    assert [chunk.section_key for chunk in narrative_chunks(texto, "Manual", ChunkingPolicy())] == ["cap4"]


def test_nome_do_capitulo_entra_no_titulo() -> None:
    texto = linhas("CAPÍTULO 3", "O Marco Simbólico", "O marco simbólico do ramo é uma frase.")
    (chunk,) = narrative_chunks(texto, "Manual", ChunkingPolicy())
    assert (
        chunk.prefix == "[Manual · CAPÍTULO 3 – O Marco Simbólico]"
        and chunk.chapter == "CAPÍTULO 3 – O Marco Simbólico"
    )


def test_titulo_em_maiusculas_em_varias_linhas() -> None:
    texto = linhas(
        "UM MARCO SIMBÓLICO: “UM", "TERRITÓRIO DE BRINCAR,", "APRENDER E CRESCER JUNTOS”", "O marco é a frase."
    )
    (chunk,) = narrative_chunks(texto, "Manual", ChunkingPolicy())
    assert chunk.prefix == "[Manual · UM MARCO SIMBÓLICO: “UM TERRITÓRIO DE BRINCAR, APRENDER E CRESCER JUNTOS”]"


def test_secao_curta_com_frase_fica() -> None:
    texto = linhas("1 Uniforme", "Use o lenço.", "2 Promessa", "A promessa é feita na cerimônia.")
    assert [chunk.section_key for chunk in narrative_chunks(texto, "Manual", ChunkingPolicy())] == ["1", "2"]


def test_sobra_curta_no_fim_entra_no_ultimo_trecho() -> None:
    texto = linhas("1 Uniforme", "Use o lenço na atividade semanal.", "2 Fim", "Créditos")
    (chunk,) = narrative_chunks(texto, "Manual", ChunkingPolicy())
    assert chunk.body.endswith("Créditos")


# --- hierarquia dos títulos (research R32) -----------------------------------------------------------------------


def _especialidade(nome: str) -> list[str]:
    return [
        nome,
        f"Nesta especialidade, você vai conhecer {nome.lower()}.",
        "SUGESTÃO DE TEMAS",
        f"Temas de {nome.lower()}: história, técnicas e cultura.",
        "CONHECER",
        "Escolha uma temática central e aprofunde-se.",
    ]


def test_subtitulo_recorrente_fica_sob_o_titulo_anterior() -> None:
    texto = linhas(*_especialidade("GASTRONOMIA"), *_especialidade("IDIOMAS"), *_especialidade("VIAGENS"))
    chunks = narrative_chunks(texto, "Guia", ChunkingPolicy(window_min=1))
    prefixos = [c.prefix for c in chunks]
    assert prefixos[:3] == [
        "[Guia · GASTRONOMIA]",
        "[Guia · GASTRONOMIA · SUGESTÃO DE TEMAS]",
        "[Guia · GASTRONOMIA · CONHECER]",
    ]
    assert "[Guia · VIAGENS · SUGESTÃO DE TEMAS]" in prefixos
    temas = next(c for c in chunks if c.body.startswith("Temas de idiomas"))
    assert (temas.chapter, temas.section) == ("IDIOMAS", "SUGESTÃO DE TEMAS")


def test_titulo_que_aparece_poucas_vezes_continua_de_nivel_um() -> None:
    texto = linhas("APRESENTAÇÃO", "Este guia orienta.", "SUGESTÃO DE TEMAS", "Temas variados.")
    chunks = narrative_chunks(texto, "Guia", ChunkingPolicy())
    assert [c.prefix for c in chunks] == ["[Guia · APRESENTAÇÃO]", "[Guia · SUGESTÃO DE TEMAS]"]


def test_requisitos_numerados_que_recomecam_nao_sao_titulos() -> None:
    requisitos = [
        "1. Explicar como as cavernas se formam.",
        "2. Listar os equipamentos",
        "e explicar os riscos.",
        "3. Fazer uma apresentação",
    ]
    texto = linhas(
        "ESPELEOTURISMO",
        "Nesta especialidade, você vai conhecer cavernas.",
        *requisitos,
        "ESCALADA",
        "Você vai escalar.",
        *requisitos,
        "RAPEL",
        "Você vai descer.",
        *requisitos,
    )
    chunks = narrative_chunks(texto, "Guia", ChunkingPolicy())
    assert [c.chapter for c in chunks] == ["ESPELEOTURISMO", "ESCALADA", "RAPEL"]
    assert "2. Listar os equipamentos e explicar os riscos." in chunks[0].body


def test_item_seguido_da_continuacao_da_frase_nao_e_titulo() -> None:
    texto = linhas("4 Programa", "O programa tem ciclos.", "5. Listar os equipamentos", "e explicar os riscos.")
    chunks = narrative_chunks(texto, "Manual X", ChunkingPolicy())
    assert [c.section_key for c in chunks] == ["4"]


def test_ano_no_comeco_da_linha_nao_e_titulo() -> None:
    texto = linhas("ATA DA REUNIÃO", "Os conselheiros se reuniram.", "2019. CRIS salienta a proposta orçamentária")
    chunks = narrative_chunks(texto, "Ata", ChunkingPolicy())
    assert len(chunks) == 1 and chunks[0].chapter == "ATA DA REUNIÃO"


# --- linhas de unidade (ADR-048) ------------------------------------------------------------------------------------


def _linhas(*textos: str) -> list[Line]:
    return [Line(text=t, page=1) for t in textos]


def test_itens_de_pauta_com_numeracao_crescente() -> None:
    from axon_scouting.rag.ingestion.domain.chunking import AGENDA_ITEM

    linhas = _linhas(
        "01) DATA: 19 de fevereiro de 2000.",
        "02) LOCAL: Curitiba.",
        "03) PROJETOS OPAS/UEB: O Diretor apresenta o convênio firmado com a OPAS e os trabalhos previstos.",
        "1) Uma lista dentro do item: recomeça e não é item.",
    )
    drafts = narrative_chunks(linhas, "Ata", ChunkingPolicy(), units=AGENDA_ITEM)
    assert [d.chapter for d in drafts] == ["01) DATA", "02) LOCAL", "03) PROJETOS OPAS/UEB"]
    assert drafts[2].body.startswith("O Diretor apresenta") and "recomeça" in drafts[2].body


def test_linha_numerada_sozinha_nao_e_unidade() -> None:
    from axon_scouting.rag.ingestion.domain.chunking import AGENDA_ITEM

    drafts = narrative_chunks(
        _linhas("1. Considerando: o texto segue normal."), "Ata", ChunkingPolicy(), units=AGENDA_ITEM
    )
    assert all(d.chapter is None for d in drafts)


def test_pergunta_do_faq_em_duas_linhas() -> None:
    from axon_scouting.rag.ingestion.domain.chunking import QUESTION

    linhas = _linhas(
        "1. O que é a Rede de Jovens Líderes?",
        "É a rede dos jovens que participam da gestão.",
        "2. Quando me torno Jovem Líder, eu deixo de",
        "ser pioneiro?",
        "Não: as duas coisas convivem.",
    )
    drafts = narrative_chunks(linhas, "FAQ", ChunkingPolicy(), units=QUESTION)
    assert [d.chapter for d in drafts] == [
        "1. O que é a Rede de Jovens Líderes?",
        "2. Quando me torno Jovem Líder, eu deixo de ser pioneiro?",
    ]


def test_secoes_pequenas_juntam_se_a_mae_com_o_titulo_no_corpo() -> None:
    """ADR-050: a filha pequena entra na mãe, e o título dela vira parágrafo do trecho."""
    texto = linhas(
        "4 Programa de Atividades",
        "O programa é dividido em ciclos.",
        "4.2 Atividades",
        "As atividades são planejadas pela patrulha.",
        "4.2.1 Acampamento",
        "O acampamento exige cardápio.",
    )
    (chunk,) = narrative_chunks(texto, "Manual X", ChunkingPolicy())
    assert chunk.prefix == "[Manual X · 4 Programa de Atividades]"
    assert chunk.body.split("\n") == [
        "O programa é dividido em ciclos.",
        "4.2 Atividades",
        "As atividades são planejadas pela patrulha.",
        "4.2.1 Acampamento",
        "O acampamento exige cardápio.",
    ]


def test_juncao_nao_engole_unidade_com_nome() -> None:
    texto = linhas("CAPÍTULO 1", "Texto curto da abertura.", "GASTRONOMIA", "Requisitos curtos.", "IDIOMAS", "Outros.")
    unidade = {"GASTRONOMIA", "IDIOMAS"}
    chunks = narrative_chunks(texto, "Guia", ChunkingPolicy(), boundary=lambda caminho, i: caminho[i] in unidade)
    assert [c.prefix for c in chunks] == ["[Guia · CAPÍTULO 1]", "[Guia · GASTRONOMIA]", "[Guia · IDIOMAS]"]
