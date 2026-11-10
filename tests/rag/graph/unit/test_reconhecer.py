"""O reconhecimento das entidades numa pergunta (spec 106 FR-010): a regra do ADR-063 (101 R43)."""

from __future__ import annotations

from axon_scouting.rag.graph.api import reconhecer

TERMOS = {
    "tipo_orgao:conselho_consultivo_nacional": ("conselho consultivo",),
    "tipo_orgao:conselho_consultivo_regional": ("conselho consultivo regional",),
    "tipo_orgao:diretoria_regional": ("diretoria regional", "dr"),
    "tipo_orgao:assembleia_regional": ("assembleia regional",),
}


def test_o_termo_mais_longo_vence() -> None:
    assert reconhecer("qual o papel do conselho consultivo regional", TERMOS) == (
        "tipo_orgao:conselho_consultivo_regional",
    )
    assert reconhecer("como funciona o conselho consultivo", TERMOS) == ("tipo_orgao:conselho_consultivo_nacional",)


def test_ordem_da_primeira_mencao_sem_repeticao() -> None:
    texto = "a assembleia regional elege a diretoria regional e a diretoria regional presta contas"
    assert reconhecer(texto, TERMOS) == ("tipo_orgao:assembleia_regional", "tipo_orgao:diretoria_regional")


def test_palavra_inteira() -> None:
    assert reconhecer("quem dirige o drone", TERMOS) == ()
    assert reconhecer("o que faz a dr", TERMOS) == ("tipo_orgao:diretoria_regional",)


def test_sem_termo() -> None:
    assert reconhecer("qual o lema do lobinho", TERMOS) == ()
    assert reconhecer("qualquer coisa", {}) == ()
