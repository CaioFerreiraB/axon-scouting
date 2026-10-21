"""Suíte de conformidade de pack publicada com os contratos (spec 101 FR-001, SC-003; T011, T035).

Na Fase 5 a carga precisa de configuração: sem ela, o default `e5` exige `embedding_dir` e o pack não carrega (fail
closed). Desde o ADR-043 a suíte carrega o artefato do índice distribuído com o pacote, em memória, com o embedder de
teste: a carga não busca, então os vetores do e5 do artefato não entram em jogo.

Roda todas as etapas, inclusive imports (o pack não importa o core) e distribuição (a faixa declarada de
`axon-ai-contracts` cabe na derivada da `contract_version`). Na instalação editável, a etapa de distribuição não tem o
que verificar e emite `distribution.not_found`; no ambiente limpo (job `clean-env`), o relatório vem sem nenhum item.
"""

from __future__ import annotations

from datetime import UTC, datetime
from importlib.metadata import packages_distributions
from pathlib import Path

from axon_ai_contracts import JSONValue
from axon_ai_contracts.testing import assert_pack_conforms, check_pack, turn_view

from axon_scouting.pack import ScoutingPack

REFERENCE = datetime(2026, 10, 7, 12, tzinfo=UTC)

QUESTIONS = (
    "Quais competências do Pioneiro estão relacionadas à liderança?",
    "Como funciona a promessa?",
    "Qual era a regra em 2023?",
    "O que dizia o manual anterior?",
    "",
)

VIEWS = tuple(turn_view(question=question, reference_time=REFERENCE) for question in QUESTIONS)

DISTRIBUTION_IS_VISIBLE = "axon-scouting" in (packages_distributions().get("axon_scouting") or [])
"""Verdadeiro na instalação normal (ambiente limpo); falso na editável (`uv sync`)."""


def config(tmp_path: Path) -> dict[str, JSONValue]:
    return {"embedder": "hashing"}


def test_pack_conforms(tmp_path: Path) -> None:
    assert_pack_conforms(ScoutingPack(), config=config(tmp_path), views=list(VIEWS))


def test_report_has_no_issue(tmp_path: Path) -> None:
    report = check_pack(ScoutingPack(), config=config(tmp_path), views=list(VIEWS))
    assert report.pack_name == "scouting"
    expected = () if DISTRIBUTION_IS_VISIBLE else ("distribution.not_found",)
    assert tuple(issue.code for issue in report.issues) == expected, report.render()
