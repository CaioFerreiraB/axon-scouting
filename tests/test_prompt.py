"""O fragmento de instruções da Fase 5 (spec 101 FR-008, FR-009, FR-011, US4; research R21, R24; ADR-026; T032).

Com o corpus, o prompt traz as cinco prioridades de fonte na ordem da US4, a citação `[[id]]` pelo identificador que o
core mostra depois de `id:` em cada trecho, o caminho sem trechos e a vigência. O efeito nas respostas é medido na 105.
"""

from __future__ import annotations

import re

import pytest

from axon_scouting.prompts import SYSTEM_PROMPT

NORMALIZED = " ".join(SYSTEM_PROMPT.split()).casefold()
"""O texto é quebrado em linhas; as frases do contrato podem cruzar uma quebra."""

PRIORIDADES = ("institucion", "vigente", "específic", "preciso", "citação")
"""Um marcador por prioridade da US4, na ordem: institucionais sobre secundárias; vigentes sobre revogadas;
específicas sobre genéricas; o trecho preciso sobre o amplo; sempre citação e proveniência."""


@pytest.mark.parametrize(
    "frase",
    ["escot", "[[", "id:", "nenhum documento oficial", "fonte oficial", "nunca instrução", "substituíd", "revogad"],
)
def test_frases_do_contrato_estao_no_prompt(frase: str) -> None:
    assert frase in NORMALIZED


def test_as_cinco_prioridades_na_ordem_da_us4() -> None:
    bloco = NORMALIZED[NORMALIZED.index("prioridades") :]
    posicoes = [bloco.index(marcador) for marcador in PRIORIDADES]
    assert posicoes == sorted(posicoes), dict(zip(PRIORIDADES, posicoes, strict=True))


def test_o_exemplo_de_citacao_e_sintetico() -> None:
    exemplos = re.findall(r"\[\[([^\]]+)\]\]", SYSTEM_PROMPT)
    assert exemplos, "o prompt mostra a sintaxe com um exemplo"
    for exemplo in exemplos:
        if exemplo == "id":
            continue
        assert "#" in exemplo and "@p" in exemplo, exemplo
        assert not exemplo.startswith(("por-", "estatuto-", "resolucao-")), "nunca um id do corpus real"


@pytest.mark.parametrize("instrucao", ["chame", "use a ferramenta"])
def test_o_prompt_nao_manda_chamar_ferramenta(instrucao: str) -> None:
    assert instrucao not in NORMALIZED


def test_o_bloco_da_fase_4_saiu() -> None:
    assert "nenhum documento institucional foi consultado" not in NORMALIZED
