"""Derivação de contexto da pergunta: ramo, temas, tipos de documento e vigência (spec 101 FR-004 a FR-006).

Função pura de (pergunta, dia de referência, léxico) — sem modelo, sem relógio, sem I/O (princípio V, SC-002). Na
dúvida, a derivação não filtra: ramo ambíguo ou mais de um ramo resultam em nenhum filtro de ramo (research R8).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from axon_ai_contracts import RETRIEVAL_FILTERS_KEY, JSONValue

from axon_scouting.context.lexicon import LEXICON, Lexicon, Topic
from axon_scouting.context.periods import Validity, find_period
from axon_scouting.context.text import contains_term, normalize

CONTEXT_KEY = "context"
"""Chave da fatia com o contexto derivado inteiro — o que a 104 e a 105 leem."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ScoutingContext:
    """O contexto derivado da pergunta (Key Entity "Scouting Context")."""

    branch: str | None
    topics: tuple[str, ...]
    document_types: tuple[str, ...]
    validity: Validity
    include_superseded: bool
    previous_version: bool


def derive_context(question: str, reference_day: date, lexicon: Lexicon = LEXICON) -> ScoutingContext:
    text = normalize(question)
    topics = tuple(topic for topic in lexicon.topics if _matches(text, topic.terms))
    period = find_period(text, reference_day)
    previous = _matches(text, lexicon.previous_markers)
    return ScoutingContext(
        branch=_branch(text, lexicon),
        topics=tuple(topic.id for topic in topics),
        document_types=_document_types(text, lexicon, topics),
        # Consulta histórica (US3): o período citado, ou o dia de referência quando a pergunta mira a versão anterior.
        validity=period or Validity.day(reference_day),
        include_superseded=period is not None or previous,
        previous_version=previous,
    )


def to_slice(context: ScoutingContext) -> dict[str, JSONValue]:
    """A atualização da fatia do pack (contracts/retrieval-filters.md): o contexto inteiro e os filtros de
    recuperação, que omitem o que não restringe."""
    full: dict[str, JSONValue] = {
        "branch": context.branch,
        "topics": list(context.topics),
        "document_types": list(context.document_types),
        "effective_from": context.validity.start.isoformat(),
        "effective_until": context.validity.end.isoformat(),
        "include_superseded": context.include_superseded,
        "previous_version": context.previous_version,
    }
    filters = {key: value for key, value in full.items() if value is not None and value != []}
    return {CONTEXT_KEY: full, RETRIEVAL_FILTERS_KEY: filters}


def _matches(text: str, terms: tuple[str, ...]) -> bool:
    return any(contains_term(text, term) for term in terms)


def _branch(text: str, lexicon: Lexicon) -> str | None:
    """Exatamente um ramo casado; nenhum, mais de um ou só termo ambíguo → sem filtro."""
    found = [branch for branch, terms in lexicon.branches.items() if _matches(text, terms)]
    return found[0] if len(found) == 1 else None


def _document_types(text: str, lexicon: Lexicon, topics: tuple[Topic, ...]) -> tuple[str, ...]:
    """Os tipos citados; sem citação, os sugeridos pelos sinais e pelos temas. Sempre na ordem do léxico."""
    explicit = {kind for kind, terms in lexicon.document_types.items() if _matches(text, terms)}
    if not explicit:
        for term, kinds in lexicon.type_signals.items():
            if contains_term(text, term):
                explicit.update(kinds)
        for topic in topics:
            explicit.update(topic.document_types)
    return tuple(kind for kind in lexicon.document_types if kind in explicit)
