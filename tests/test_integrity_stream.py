"""`citation_integrity` em streaming (spec 104; ADR-024): o mesmo resultado do lote, para qualquer partição.

Propriedades: `finish()` é igual a `validate()` sobre a resposta inteira; a resposta final começa com o texto que `feed`
liberou; e nenhum identificador inventado chega a sair — com qualquer partição da resposta em pedaços.
"""

from __future__ import annotations

import asyncio

import pytest
from ai_core_contracts.testing import chunk, turn_view, validator_context

from ai_scouting.validators import CitationIntegrityValidator

CONTEXTO = (
    chunk("man-pio-2026#4.2", document="Manual Pioneiro", version="2026", status="vigente"),
    chunk("por-2023#12", document="POR", version="2023", status="substituído"),
)

RESPOSTAS = [
    "A regra mudou [[man-pio-2026#4.2]]; antes era outra [[ POR-2023#12 ]] [[inventado#1]].",
    "Sem citação nenhuma.",
    "Só inventada [[man-x#9]] e fim.",
    "Repetida [[por-2023#12]], de novo [[por-2023#12]].",
    "Espaços antes   [[inventado#2]] somem.",
    "Marcador que não fecha no fim [[man-pio",
    "Colchete solto no fim [",
    "Colchetes estranhos [[a[[man-pio-2026#4.2]] e [x] e ]] fim.",
    "",
    "Termina com espaço [[man-pio-2026#4.2]]   ",
]


def particoes(texto: str) -> list[list[str]]:
    return [
        [texto],
        list(texto),
        [texto[i : i + 3] for i in range(0, len(texto), 3)],
        [texto[i : i + 7] for i in range(0, len(texto), 7)],
    ]


@pytest.mark.parametrize("resposta", RESPOSTAS)
def test_streaming_da_o_mesmo_resultado_do_lote(resposta: str) -> None:
    validator = CitationIntegrityValidator()
    ctx = validator_context("scouting")
    esperado = asyncio.run(validator.validate(turn_view(answer=resposta, generation_context=CONTEXTO), ctx))

    for partes in particoes(resposta):
        sessao = validator.open_stream(turn_view(generation_context=CONTEXTO), ctx)
        liberado = "".join(sessao.feed(parte) for parte in partes)
        resultado = sessao.finish()

        assert resultado == esperado
        assert (resultado.answer or "").startswith(liberado)
        assert "inventado" not in liberado and "man-x" not in liberado


def test_o_texto_sai_enquanto_chega() -> None:
    sessao = CitationIntegrityValidator().open_stream(
        turn_view(generation_context=CONTEXTO), validator_context("scouting")
    )
    assert sessao.feed("Primeira frase. ") == "Primeira frase."
    assert sessao.feed("Segunda") == " Segunda"
    assert sessao.feed(" [[man-pio") == ""  # marcador aberto: segura, inclusive o espaço antes dele
    assert sessao.feed("-2026#4.2]] ok") == " [1] ok"
