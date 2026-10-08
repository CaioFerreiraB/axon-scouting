"""Estrutura normativa: uma unidade (artigo ou regra) = um chunk (spec 102 FR-002, FR-003, SC-001; R7; T036)."""

from __future__ import annotations

import pytest

from ai_scouting.rag.ingestion.domain.chunking import ChunkingPolicy
from ai_scouting.rag.ingestion.domain.structure import normative_chunks
from ai_scouting.rag.ingestion.domain.text import Line

POLITICA = ChunkingPolicy()


def linhas(*paginas: list[str]) -> list[Line]:
    return [Line(text=t, page=n) for n, textos in enumerate(paginas, start=1) for t in textos]


LEI = linhas(
    [
        "O Conselho Nacional, no uso de suas atribuições, resolve:",
        "TÍTULO II",
        "DA ORGANIZAÇÃO",
        "CAPÍTULO III – Dos Associados",
        "SEÇÃO I",
        "Dos direitos",
        "Art. 41. O associado tem direito a voto.",
        "Art. 42. O responsável pela atividade deverá:",
        "I – comunicar os pais;",
        "II – registrar a atividade, observando:",
        "a) a lista de participantes;",
        "b) o plano de segurança.",
    ],
    [
        "§ 1º A comunicação é por escrito.",
        "Parágrafo único. O registro fica com o grupo.",
        "Art. 42-A. Aplica-se o artigo anterior aos acampamentos, nos termos do",
        "Art. 5º do Estatuto.",
        "SUBSEÇÃO I",
        "Dos deveres",
        "Art. 43. O associado cumpre a Lei Escoteira.",
    ],
)


def por_chave(drafts: list[object]) -> dict[str, object]:
    return {d.section_key: d for d in drafts}  # type: ignore[attr-defined]


def test_um_chunk_por_artigo_com_prefixo_e_metadados() -> None:
    resultado = normative_chunks(LEI, "Lei X", POLITICA, unit="article")
    assert resultado.units == 4
    chaves = [d.section_key for d in resultado.drafts]
    assert chaves == ["preambulo", "art41", "art42", "art42-a", "art43"]
    art42 = por_chave(resultado.drafts)["art42"]
    assert art42.prefix == "[Lei X · Título II · Capítulo III · Seção I]"  # type: ignore[attr-defined]
    assert art42.article == "Art. 42"  # type: ignore[attr-defined]
    assert art42.chapter == "Capítulo III – Dos Associados"  # type: ignore[attr-defined]
    assert art42.section == "Seção I – Dos direitos"  # type: ignore[attr-defined]
    assert art42.subsection is None  # type: ignore[attr-defined]
    assert art42.normative  # type: ignore[attr-defined]
    corpo = art42.body  # type: ignore[attr-defined]
    for trecho in (
        "Art. 42. O responsável",
        "I – comunicar os pais;",
        "a) a lista",
        "b) o plano",
        "§ 1º A comunicação",
        "Parágrafo único.",
    ):
        assert trecho in corpo
    assert (art42.page_start, art42.page_end) == (1, 2)  # type: ignore[attr-defined]


def test_artigo_citado_no_meio_da_frase_nao_e_unidade() -> None:
    art42a = por_chave(normative_chunks(LEI, "Lei X", POLITICA, unit="article").drafts)["art42-a"]
    assert art42a.article == "Art. 42-A"  # type: ignore[attr-defined]
    assert "nos termos do Art. 5º do Estatuto." in art42a.body  # type: ignore[attr-defined]


def test_subsecao_entra_no_caminho() -> None:
    art43 = por_chave(normative_chunks(LEI, "Lei X", POLITICA, unit="article").drafts)["art43"]
    assert art43.prefix == "[Lei X · Título II · Capítulo III · Seção I · Subseção I]"  # type: ignore[attr-defined]
    assert art43.subsection == "Subseção I – Dos deveres"  # type: ignore[attr-defined]


def test_preambulo_em_janelas() -> None:
    preambulo = normative_chunks(LEI, "Lei X", POLITICA, unit="article").drafts[0]
    assert preambulo.section_key == "preambulo"
    assert preambulo.prefix == "[Lei X]"
    assert not preambulo.normative
    assert preambulo.body.startswith("O Conselho Nacional")


@pytest.mark.parametrize(
    ("linha", "chave", "rotulo"),
    [
        ("Art. 1º O grupo é a unidade básica.", "art1", "Art. 1"),
        ("ART. 7 o texto", "art7", "Art. 7"),
        ("Artigo 12 - O texto.", "art12", "Art. 12"),
        ("art. 42-a. O texto.", "art42-a", "Art. 42-A"),
        ("Regra 033 – O uniforme é obrigatório.", "regra33", "Regra 033"),
        ("REGRA 7. O texto.", "regra7", "Regra 7"),
    ],
)
def test_marcadores_de_unidade(linha: str, chave: str, rotulo: str) -> None:
    unidade = "rule" if chave.startswith("regra") else "article"
    (draft,) = normative_chunks(linhas([linha]), "Norma", POLITICA, unit=unidade).drafts
    assert (draft.section_key, draft.article) == (chave, rotulo)


@pytest.mark.parametrize(
    ("cabecalho", "rotulo"),
    [
        ("TÍTULO II", "Título II"),
        ("titulo ii", "Título II"),
        ("CAPÍTULO III", "Capítulo III"),
        ("Capítulo 4", "Capítulo 4"),
        ("SEÇÃO I", "Seção I"),
        ("Secao IV", "Seção IV"),
        ("SUBSEÇÃO I", "Subseção I"),
    ],
)
def test_marcadores_de_agrupamento(cabecalho: str, rotulo: str) -> None:
    (draft,) = normative_chunks(linhas([cabecalho, "Art. 1. Texto."]), "Norma", POLITICA, unit="article").drafts
    assert draft.prefix == f"[Norma · {rotulo}]"


def test_anexo() -> None:
    norma = linhas(
        ["Art. 1º A norma vale para todos.", "ANEXO I", "Modelo de autorização dos pais.", "Art. 1º O modelo é anexo."],
    )
    resultado = normative_chunks(norma, "Norma", POLITICA, unit="article")
    assert [(d.section_key, d.page_start) for d in resultado.drafts] == [("art1", 1), ("anexo-i", 1), ("art1", 1)]
    anexo, artigo = resultado.drafts[1], resultado.drafts[2]
    assert anexo.prefix == "[Norma · Anexo I]" and not anexo.normative
    assert artigo.prefix == "[Norma · Anexo I]" and artigo.article == "Art. 1"


def test_sem_unidade_sinaliza_fallback() -> None:
    resultado = normative_chunks(
        linhas(["Texto corrido sem artigos.", "Mais texto."]), "Norma", POLITICA, unit="article"
    )
    assert resultado.units == 0
    assert resultado.drafts == []


# --- formatos do corpus real (SC-005, T053) ------------------------------------------------------------------------


def test_regra_do_por_em_duas_linhas() -> None:
    """No POR, "REGRA" fica numa linha e "001 DEFINIÇÃO DO ESCOTISMO" na seguinte."""
    por = linhas(
        [
            "CAPÍTULO 1",
            "Fundamentos do Escotismo",
            "REGRA",
            "001 DEFINIÇÃO DO ESCOTISMO",
            "O Escotismo é um movimento.",
        ],
        ["REGRA", "002 PROPÓSITO DO ESCOTISMO", "O propósito é contribuir."],
    )
    resultado = normative_chunks(por, "POR", POLITICA, unit="rule")
    assert [(d.section_key, d.article, d.page_start) for d in resultado.drafts] == [
        ("regra1", "Regra 001", 1),
        ("regra2", "Regra 002", 2),
    ]
    assert resultado.drafts[0].body == "REGRA 001 DEFINIÇÃO DO ESCOTISMO\nO Escotismo é um movimento."
    assert resultado.drafts[0].chapter == "Capítulo 1 – Fundamentos do Escotismo"


def test_resolucao_cita_regras_dentro_do_artigo() -> None:
    """Numa resolução, a unidade é o artigo: a regra citada com a nova redação fica dentro dele."""
    resolucao = linhas(
        [
            "RESOLUÇÃO CAN 08/2021",
            "Altera as Regras 001 e 002.",
            "Art. 1º Alterar a Regra 001 do P.O.R, que passa a vigorar com a seguinte redação:",
            "REGRA 001 – DEFINIÇÃO DO ESCOTISMO",
            "O Escotismo é um movimento educacional.",
            "Art. 2º Esta resolução entra em vigor na data de sua publicação.",
        ]
    )
    resultado = normative_chunks(resolucao, "Resolução CAN 08/2021", POLITICA, unit="article")
    assert [(d.section_key, d.article) for d in resultado.drafts] == [
        ("preambulo", None),
        ("art1", "Art. 1"),
        ("art2", "Art. 2"),
    ]
    assert "REGRA 001 – DEFINIÇÃO DO ESCOTISMO" in resultado.drafts[1].body
    assert resultado.drafts[0].prefix == "[Resolução CAN 08/2021]"  # linha em maiúsculas no preâmbulo não muda a chave


def test_regras_seguidas_sem_texto_entre_elas() -> None:
    """Títulos de regra em maiúsculas abrem unidade mesmo sem pontuação na linha anterior (POR, regras 199–201)."""
    por = linhas(
        [
            "REGRA",
            "199 LENÇO DA ORGANIZAÇÃO MUNDIAL",
            "REGRA",
            "200 DISTINTIVO DA ORGANIZAÇÃO MUNDIAL",
            "Usado por quem exerce função",
            "REGRA",
            "201 CONDECORAÇÕES",
        ]
    )
    chaves = [d.section_key for d in normative_chunks(por, "POR", POLITICA, unit="rule").drafts]
    assert chaves == ["regra199", "regra200", "regra201"]


def test_citacao_de_regra_no_meio_da_frase_continua_texto() -> None:
    por = linhas(["REGRA", "010 MÉTODO", "Conforme a", "Regra 009 do capítulo anterior, o método vale."])
    (draft,) = normative_chunks(por, "POR", POLITICA, unit="rule").drafts
    assert draft.section_key == "regra10" and "Regra 009 do capítulo anterior" in draft.body


def test_anexos_sem_numero_e_nome_em_duas_linhas() -> None:
    por = linhas(
        ["CAPÍTULO 4", "Orientação Patrimonial", "e Financeira", "REGRA", "026 PATRIMÔNIO", "O patrimônio é da UEB."],
        [
            "REGRA",
            "201 CONDECORAÇÕES",
            "As condecorações são concedidas.",
            "ANEXOS",
            "Cores das matilhas do Ramo Lobinho.",
        ],
    )
    resultado = normative_chunks(por, "POR", POLITICA, unit="rule")
    assert [(d.section_key, d.normative) for d in resultado.drafts] == [
        ("regra26", True),
        ("regra201", True),
        ("anexos", False),
    ]
    regra26, regra201, anexos = resultado.drafts
    assert regra26.chapter == "Capítulo 4 – Orientação Patrimonial e Financeira"
    assert "Cores das matilhas" not in regra201.body
    assert anexos.prefix == "[POR · Anexos]"


def test_nome_em_maiusculas_em_duas_linhas() -> None:
    norma = linhas(["TÍTULO IV", "DO PATRIMÔNIO, DA RECEITA E DA", "ADMINISTRAÇÃO", "Art. 60. O patrimônio é da UEB."])
    (draft,) = normative_chunks(norma, "Estatuto", POLITICA, unit="article").drafts
    assert draft.section_key == "art60"
