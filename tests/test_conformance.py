"""Suíte de conformidade de pack publicada com os contratos (spec 101 FR-001, SC-003; T011).

Roda todas as etapas, inclusive imports (o pack não importa o core) e distribuição (a faixa declarada de
`ai-core-contracts` cabe na derivada da `contract_version`). Na instalação editável, a etapa de distribuição não tem o
que verificar e emite `distribution.not_found`; no ambiente limpo (job `clean-env`), o relatório vem sem nenhum item.
"""

from __future__ import annotations

from datetime import UTC, datetime
from importlib.metadata import packages_distributions

from ai_core_contracts.testing import assert_pack_conforms, check_pack, turn_view

from ai_scouting.pack import ScoutingPack

REFERENCE = datetime(2026, 10, 7, 12, tzinfo=UTC)

QUESTIONS = (
    "Quais competências do Pioneiro estão relacionadas à liderança?",
    "Como funciona a promessa?",
    "Qual era a regra em 2023?",
    "O que dizia o manual anterior?",
    "",
)

VIEWS = tuple(turn_view(question=question, reference_time=REFERENCE) for question in QUESTIONS)

DISTRIBUTION_IS_VISIBLE = "ai-scouting" in (packages_distributions().get("ai_scouting") or [])
"""Verdadeiro na instalação normal (ambiente limpo); falso na editável (`uv sync`)."""


def test_pack_conforms() -> None:
    assert_pack_conforms(ScoutingPack(), views=list(VIEWS))


def test_report_has_no_issue() -> None:
    report = check_pack(ScoutingPack(), views=list(VIEWS))
    assert report.pack_name == "scouting"
    expected = () if DISTRIBUTION_IS_VISIBLE else ("distribution.not_found",)
    assert tuple(issue.code for issue in report.issues) == expected, report.render()
