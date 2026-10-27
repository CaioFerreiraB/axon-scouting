"""Métricas de domínio do scouting para o harness da 009 (spec 105, FR-003, FR-014; research R2, R3, R13; ADR-027).

Puras e síncronas: leem o item (`EvalItem`) e o que a execução produziu (`EvalObservation`), sem relógio — o dia vem do
`reference_time` do item — e sem I/O. O texto dos trechos entregues, que o `EvalChunk` não traz, chega à `FactRecall`
por uma função injetada pelo `pack.py`.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from typing import Final, Literal

from axon_ai_contracts import JSONValue
from axon_ai_contracts.eval import EvalItem, EvalObservation

from axon_scouting.context.derivation import ScoutingContext, derive_context
from axon_scouting.context.periods import reference_day
from axon_scouting.context.text import contains_term, normalize
from axon_scouting.rag.index.api import ValidityStatus

CONTEXT_KEY: Final = "context"
"""A chave de `EvalItem.extra` com o contexto esperado da derivação da 101."""

CONTEXT_FIELDS: Final = (
    "branch",
    "branches",
    "topics",
    "document_types",
    "suggested_types",
    "effective_from",
    "effective_until",
    "include_superseded",
    "previous_version",
)
"""Os campos que `extra.context` pode declarar (data-model.md); só os declarados contam."""

FACTS_KEY: Final = "facts"
"""A chave de `EvalItem.extra` com os fatos que precisam chegar à geração (FR-014)."""


def expected_context(extra: Mapping[str, JSONValue]) -> Mapping[str, JSONValue]:
    """O contexto esperado que o item declara, ou vazio."""
    context = extra.get(CONTEXT_KEY)
    return context if isinstance(context, Mapping) else {}


def is_historical(extra: Mapping[str, JSONValue]) -> bool:
    """Item histórico: `extra.context` com `include_superseded` ou `previous_version` verdadeiro."""
    context = expected_context(extra)
    return context.get("include_superseded") is True or context.get("previous_version") is True


def expected_facts(extra: Mapping[str, JSONValue]) -> tuple[tuple[str, ...], ...]:
    """Os fatos que o item declara em `extra.facts`, normalizados: cada fato é um texto ou uma lista de redações
    alternativas (basta uma); o contrato entrega as listas como tuplas. Entrada que não é texto nem lista de textos é
    ignorada — a validação é dos testes."""
    raw = extra.get(FACTS_KEY)
    if not isinstance(raw, list | tuple):
        return ()
    facts: list[tuple[str, ...]] = []
    for fact in raw:
        alternatives = [fact] if isinstance(fact, str) else fact if isinstance(fact, list | tuple) else []
        normalized = tuple(normalize(text) for text in alternatives if isinstance(text, str) and normalize(text))
        if normalized:
            facts.append(normalized)
    return tuple(facts)


def fact_found(fact: tuple[str, ...], texts: Sequence[str]) -> bool:
    """Alguma redação do fato aparece inteira (palavra a palavra) em algum dos textos já normalizados."""
    return any(contains_term(text, alternative) for alternative in fact for text in texts)


class FactRecall:
    """Fração dos fatos de `extra.facts` presentes no texto dos trechos entregues à geração (FR-014; research R13).

    Mede a recuperação pelo conteúdo, não pelo `chunk_id`: sobrevive a mudança de chunking e diz se a informação
    chegou, qualquer que seja o documento que a trouxe. Trecho cujo texto a função não conhece conta como vazio.
    """

    name = "fact_recall"
    better: Literal["higher", "lower"] = "higher"
    deterministic = True

    def __init__(self, text_of: Callable[[str], str | None]) -> None:
        self._text_of = text_of

    def measure(self, item: EvalItem, observation: EvalObservation) -> float | None:
        facts = expected_facts(item.extra)
        if not facts:
            return None
        texts = [normalize(self._text_of(chunk.chunk_id) or "") for chunk in observation.delivered]
        return sum(fact_found(fact, texts) for fact in facts) / len(facts)


_SOURCES: Final = r"(?:documento|documentos|norma|normas|fonte|fontes|acervo|material|materiais)"
ABSENCE_CLAIMS: Final = (
    re.compile(
        rf"\bnao (?:existe|existem|ha|consta|constam)\b.{{0,60}}\b(?:nos|nas|em nenhum|em nenhuma) {_SOURCES}\b"
    ),
    re.compile(
        rf"\b{_SOURCES}(?: oficiais| oficial)? (?:nao|nada) (?:traz|trazem|contem|menciona|mencionam|trata|tratam|"
        r"define|definem|apresenta|apresentam|informa|informam)\b"
    ),
    re.compile(r"\bnenhum documento (?:oficial )?(?:foi encontrado|trata|menciona|traz|define)\b"),
)
"""Afirmações de que o acervo não tem a informação, sobre a resposta normalizada pela 101. "Não apareceu nos trechos"
não casa: os trechos são uma amostra da busca, e dizer isso é o que o prompt pede (FR-022)."""


class AbsenceClaimRate:
    """Em item cujos fatos existem no corpus (`extra.facts`), 1 se a resposta afirma que os documentos não trazem a
    informação, senão 0 (FR-017; research R17). Depende da geração: não barra o CI — é para a rodada com modelo real."""

    name = "absence_claim_rate"
    better: Literal["higher", "lower"] = "lower"
    deterministic = False

    def measure(self, item: EvalItem, observation: EvalObservation) -> float | None:
        if not expected_facts(item.extra):
            return None
        answer = normalize(observation.answer)
        return 1.0 if any(pattern.search(answer) for pattern in ABSENCE_CLAIMS) else 0.0


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
        "branches": list(context.branches),
        "topics": list(context.topics),
        "document_types": list(context.document_types),
        "suggested_types": list(context.suggested_types),
        "effective_from": context.validity.start.isoformat(),
        "effective_until": context.validity.end.isoformat(),
        "include_superseded": context.include_superseded,
        "previous_version": context.previous_version,
    }


def _matches(key: str, expected: JSONValue, derived: JSONValue) -> bool:
    if key in ("branches", "topics", "document_types", "suggested_types"):
        return isinstance(expected, list | tuple) and isinstance(derived, list) and set(expected) == set(derived)
    return expected == derived
