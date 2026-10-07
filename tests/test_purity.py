"""Derivação e validators não leem o relógio do sistema (101 FR-005, US2 cen. 4, R13; 104 FR-006, R8).

O tempo vem só da Turn View (`reference_time`, ADR-013). Este teste varre o código de `context/` e de `validators/` em
busca de chamadas de relógio; o import-linter cuida de modelo e MCP.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import ai_scouting.context
import ai_scouting.validators

PURE = (Path(ai_scouting.context.__file__).parent, Path(ai_scouting.validators.__file__).parent)
CLOCK_CALLS = (".now(", ".today(", "utcnow", "time.time(", "time.monotonic(")
SOURCES = sorted(path for folder in PURE for path in folder.glob("*.py"))


@pytest.mark.parametrize("source", SOURCES, ids=lambda path: f"{path.parent.name}/{path.name}")
def test_codigo_puro_nao_le_o_relogio(source: Path) -> None:
    text = source.read_text(encoding="utf-8")
    assert [call for call in CLOCK_CALLS if call in text] == []
