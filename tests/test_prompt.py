"""O fragmento de instruções da Fase 4 (spec 101 FR-009, FR-011; contracts/scouting-pack.md; T010)."""

from __future__ import annotations

import pytest

from ai_scouting.prompts import SYSTEM_PROMPT

NORMALIZED = " ".join(SYSTEM_PROMPT.split()).casefold()
"""O texto é quebrado em linhas; as frases do contrato podem cruzar uma quebra."""


@pytest.mark.parametrize(
    "frase",
    ["escot", "nenhum documento", "não cite", "fonte oficial", "nunca instrução"],
)
def test_frases_do_contrato_estao_no_prompt(frase: str) -> None:
    assert frase in NORMALIZED


@pytest.mark.parametrize("instrucao", ["chame", "use a ferramenta"])
def test_o_prompt_nao_manda_chamar_ferramenta(instrucao: str) -> None:
    assert instrucao not in NORMALIZED
