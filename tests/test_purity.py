"""Derivação, validators, retrieval e métricas não leem o relógio (101 FR-005; 104 FR-006; 103 FR-002; 105 R10).

O tempo vem só da Turn View ou do item avaliado (`reference_time`, ADR-013). Este teste varre o código de `context/`,
`validators/`, `rag/retrieval/` e `eval/` em busca de chamadas de relógio; o import-linter cuida de modelo e MCP.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import ai_scouting.context
import ai_scouting.eval
import ai_scouting.rag.retrieval
import ai_scouting.validators

PURE = (
    Path(ai_scouting.context.__file__).parent,
    Path(ai_scouting.validators.__file__).parent,
    Path(ai_scouting.rag.retrieval.__file__).parent,  # 103 FR-002: a vigência vem nos filtros, nunca do relógio
    Path(ai_scouting.eval.__file__).parent,  # 105 R10: o dia vem do `reference_time` do item
)
CLOCK_CALLS = (".now(", ".today(", "utcnow", "time.time(", "time.monotonic(")
SOURCES = sorted(path for folder in PURE for path in folder.rglob("*.py"))


@pytest.mark.parametrize("source", SOURCES, ids=lambda path: f"{path.parent.name}/{path.name}")
def test_codigo_puro_nao_le_o_relogio(source: Path) -> None:
    text = source.read_text(encoding="utf-8")
    assert [call for call in CLOCK_CALLS if call in text] == []
