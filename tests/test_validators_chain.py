"""Os dois validators em sequência, como o Pack Runtime encadeia (spec 104; quickstart §2; T013)."""

from __future__ import annotations

import asyncio

from ai_core_contracts.testing import chunk, turn_view, validator_context

from ai_scouting.validators import CitationIntegrityValidator, ValidityNoticeValidator

CONTEXTO = (
    chunk("man-pio-2023#4.2", document="Manual Pioneiro", version="2023", status="substituído"),
    chunk("man-pio-2026#4.2", document="Manual Pioneiro", version="2026", status="vigente"),
)


def test_integridade_depois_vigencia() -> None:
    ctx = validator_context("scouting")
    resposta = "A regra mudou [[man-pio-2026#4.2]]; antes era outra [[ MAN-PIO-2023#4.2 ]] [[inventado#1]]."
    primeiro = asyncio.run(
        CitationIntegrityValidator().validate(turn_view(answer=resposta, generation_context=CONTEXTO), ctx)
    )
    view = turn_view(answer=primeiro.answer, citations=primeiro.citations or (), generation_context=CONTEXTO)
    segundo = asyncio.run(ValidityNoticeValidator().validate(view, ctx))

    assert segundo.answer == (
        "A regra mudou [1]; antes era outra [2].\n\n"
        "Atenção: [2] cita Manual Pioneiro (2023), que está substituído — não é a versão vigente."
    )
    assert primeiro.issues == ("citação fora do contexto da geração: inventado#1",)
    assert primeiro.flags | segundo.flags == {"invalid_citation", "non_current_source"}
