"""A derivação não lê o relógio do sistema (spec 101 FR-005, US2 cen. 4; research R13; T017).

O tempo vem só da Turn View (`reference_time`, ADR-013). Este teste varre o código de `context/` em busca de chamadas
de relógio; o import-linter cuida de modelo e MCP.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import ai_scouting.context

CONTEXT = Path(ai_scouting.context.__file__).parent
CLOCK_CALLS = (".now(", ".today(", "utcnow", "time.time(", "time.monotonic(")


@pytest.mark.parametrize("source", sorted(CONTEXT.glob("*.py")), ids=lambda path: path.name)
def test_context_nao_le_o_relogio(source: Path) -> None:
    text = source.read_text(encoding="utf-8")
    assert [call for call in CLOCK_CALLS if call in text] == []
