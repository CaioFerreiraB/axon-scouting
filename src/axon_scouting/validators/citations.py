"""Sintaxe de citação `[[chunk_id]]`: extração, chave de comparação e reescrita (spec 104 FR-001, FR-009; R3, R5).

Puro. O texto do modelo é dado: só vira citação o marcador cuja chave casa com um trecho entregue à geração.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Final

MARKER: Final = re.compile(r"\[\[\s*([^\[\]]+?)\s*\]\]")
"""`[[id]]`, com espaços tolerados dentro dos colchetes."""

_REMOVED: Final = re.compile(r"\s*\[\[\s*([^\[\]]+?)\s*\]\]")
"""O marcador com o espaço que o precede — usado para tirar a citação inválida sem deixar espaço sobrando."""


def key(raw: str) -> str:
    """Chave de comparação: sem nenhum espaço, sem diferença de maiúsculas (caso de borda "variação de formatação")."""
    return "".join(raw.split()).casefold()


def extract(answer: str) -> list[str]:
    """Os ids citados, como escritos, na ordem do texto (com repetições)."""
    return [match.group(1) for match in MARKER.finditer(answer)]


def rewrite(answer: str, numbers: Mapping[str, int]) -> str:
    """Troca cada marcador válido (chave em `numbers`) por `[n]` e remove os demais com o espaço anterior."""

    def replace(match: re.Match[str]) -> str:
        number = numbers.get(key(match.group(1)))
        if number is None:
            return ""
        prefix = match.group(0)[: match.start(1) - match.start(0)]
        spacing = prefix[: len(prefix) - len(prefix.lstrip())]
        return f"{spacing}[{number}]"

    return _REMOVED.sub(replace, answer)
