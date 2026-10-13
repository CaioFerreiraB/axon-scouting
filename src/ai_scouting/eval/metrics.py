"""Métricas de domínio do scouting para o harness da 009 (spec 105, FR-003; research R2, R3; ADR-027).

Puras e síncronas: leem o item (`EvalItem`) e o que a execução produziu (`EvalObservation`), sem relógio — o dia vem do
`reference_time` do item — e sem I/O.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final, Literal

from ai_core_contracts import JSONValue
from ai_core_contracts.eval import EvalItem, EvalObservation

from ai_scouting.context.derivation import ScoutingContext, derive_context
from ai_scouting.context.periods import reference_day
from ai_scouting.rag.index.api import ValidityStatus

CONTEXT_KEY: Final = "context"
"""A chave de `EvalItem.extra` com o contexto esperado da derivação da 101."""

CONTEXT_FIELDS: Final = (
    "branch",
    "topics",
    "document_types",
    "effective_from",
    "effective_until",
    "include_superseded",
    "previous_version",
)
"""Os campos que `extra.context` pode declarar (data-model.md); só os declarados contam."""


def expected_context(extra: Mapping[str, JSONValue]) -> Mapping[str, JSONValue]:
    """O contexto esperado que o item declara, ou vazio."""
    context = extra.get(CONTEXT_KEY)
    return context if isinstance(context, Mapping) else {}


def is_historical(extra: Mapping[str, JSONValue]) -> bool:
    """Item histórico: `extra.context` com `include_superseded` ou `previous_version` verdadeiro."""
    context = expected_context(extra)
    return context.get("include_superseded") is True or context.get("previous_version") is True


class ContextAccuracy:
    """Acerto da derivação de contexto da 101 (research R3).

    A derivação é refeita para a pergunta e o dia de referência do item e comparada campo a campo com o que
    `extra.context` declara; valor = casados / declarados.
    """

    name = "context_accuracy"
    better: Literal["higher", "lower"] = "higher"
    deterministic = True

    def measure(self, item: EvalItem, observation: EvalObservation) -> float | None:
        declared = {key: value for key, value in expected_context(item.extra).items() if key in CONTEXT_FIELDS}
        if not declared:
            return None
        derived = _derived(derive_context(item.question, reference_day(item.reference_time)))
        matched = sum(_matches(key, value, derived[key]) for key, value in declared.items())
        return matched / len(declared)


class CurrentDocumentRate:
    """Fração dos trechos entregues à geração com status `vigente`, em item não histórico (research R2)."""

    name = "current_document_rate"
    better: Literal["higher", "lower"] = "higher"
    deterministic = True

    def measure(self, item: EvalItem, observation: EvalObservation) -> float | None:
        if is_historical(item.extra) or not observation.delivered:
            return None
        current = sum(chunk.status == ValidityStatus.VIGENTE for chunk in observation.delivered)
        return current / len(observation.delivered)


class ValidityNoticeRate:
    """Com citação não vigente na resposta final, 1 se o aviso da 104 saiu (`non_current_source`), senão 0.

    "Não vigente" como no `validity_notice`: todo status diferente de `vigente`, inclusive ausente. Depende da geração,
    por isso não barra o CI.
    """

    name = "validity_notice_rate"
    better: Literal["higher", "lower"] = "higher"
    deterministic = False

    def measure(self, item: EvalItem, observation: EvalObservation) -> float | None:
        if all(citation.get("status") == ValidityStatus.VIGENTE for citation in observation.citations):
            return None
        return 1.0 if "non_current_source" in observation.flags else 0.0


def _derived(context: ScoutingContext) -> dict[str, JSONValue]:
    """A derivação na forma de `extra.context` (a mesma da fatia publicada pelo pack)."""
    return {
        "branch": context.branch,
        "topics": list(context.topics),
        "document_types": list(context.document_types),
        "effective_from": context.validity.start.isoformat(),
        "effective_until": context.validity.end.isoformat(),
        "include_superseded": context.include_superseded,
        "previous_version": context.previous_version,
    }


def _matches(key: str, expected: JSONValue, derived: JSONValue) -> bool:
    if key in ("topics", "document_types"):
        return isinstance(expected, list | tuple) and isinstance(derived, list) and set(expected) == set(derived)
    return expected == derived
