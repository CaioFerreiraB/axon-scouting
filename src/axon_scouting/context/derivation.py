"""Derivação de contexto da pergunta: ramo, temas, tipos de documento e vigência (spec 101 FR-004 a FR-006).

Função pura de (pergunta, dia de referência, léxico) — sem modelo, sem relógio, sem I/O (princípio V, SC-002). Só o
tipo de documento citado filtra; o sugerido pelos temas só ordena (ADR-046). Os tipos, os órgãos e a abrangência são os
da taxonomia do acervo (ADR-049); órgão e abrangência filtram só junto com um tipo citado ou um pedido de lista ("as
atas da assembleia regional") — sozinhos ("como funciona o CAN?"), o órgão só desempata. Na dúvida, a derivação não
filtra: ramo ambíguo resulta em nenhum filtro de ramo (research R8). Mais de um ramo, ou um marcador de totalidade
("cada ramo"), resulta em facetas — `branches` —, uma busca por ramo na 103 (ADR-045).
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
    branches: tuple[str, ...]
    """Facetas: dois ramos ou mais, na ordem do léxico (ADR-045); com elas, `branch` é `None`."""
    topics: tuple[str, ...]
    document_types: tuple[str, ...]
    """Os tipos que a pergunta cita ("segundo o estatuto"): filtram a busca."""
    suggested_types: tuple[str, ...]
    """Os tipos que os temas e os sinais sugerem: só desempatam a ordem, nunca excluem (ADR-046)."""
    validity: Validity
    include_superseded: bool
    previous_version: bool
    bodies: tuple[str, ...] = ()
    """Os órgãos nomeados (ids da taxonomia), quando filtram: com um tipo citado ou um pedido de lista (ADR-049)."""
    scopes: tuple[str, ...] = ()
    """A abrangência nomeada (`nacional`, `regional`…), nas mesmas condições dos órgãos."""
    suggested_bodies: tuple[str, ...] = ()
    """Os órgãos nomeados sem tipo citado nem pedido de lista: só desempatam a ordem."""
    catalog: CatalogRequest | None = None
    """Pedido de lista de documentos (ADR-047); `None` quando a pergunta não pede uma."""
    enumeration: EnumerationRequest | None = None
    """Pedido de enumeração que não é de documentos (ADR-048); `None` quando a pergunta não pede uma."""


@dataclass(frozen=True, slots=True, kw_only=True)
class CatalogRequest:
    """A lista pedida: a espécie, pela palavra do título no singular (`documento` = qualquer título), e as palavras
    da pergunta que a qualificam, normalizadas, na ordem em que aparecem."""

    kind: str
    terms: tuple[str, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class EnumerationRequest:
    """A enumeração pedida: o tipo de unidade cujo plural a pergunta usa (ou `None`) e as palavras que a qualificam,
    sem as de ligação, o plural do tipo e o ramo, na ordem em que aparecem."""

    unit_kind: str | None
    terms: tuple[str, ...]


def derive_context(question: str, reference_day: date, lexicon: Lexicon = LEXICON) -> ScoutingContext:
    text = normalize(question)
    topics = tuple(topic for topic in lexicon.topics if _matches(text, topic.terms))
    period = find_period(text, reference_day)
    previous = _matches(text, lexicon.previous_markers)
    branch, branches = _branches(text, lexicon)
    catalog = _catalog(text, lexicon)
    cited = _cited_types(text, lexicon)
    named = tuple(body for body, terms in lexicon.bodies.items() if _matches(text, terms))
    scoped = tuple(scope for scope, terms in lexicon.scopes.items() if _matches(text, terms))
    filtering = bool(cited) or catalog is not None  # órgão e abrangência são fonte só quando a pergunta cita um tipo
    return ScoutingContext(
        branch=branch,
        branches=branches,
        topics=tuple(topic.id for topic in topics),
        document_types=cited,
        suggested_types=_suggested_types(text, lexicon, topics),
        # Consulta histórica (US3): o período citado, ou o dia de referência quando a pergunta mira a versão anterior.
        validity=period or Validity.day(reference_day),
        include_superseded=period is not None or previous,
        previous_version=previous,
        catalog=catalog,
        enumeration=None if catalog is not None else _enumeration(text, lexicon),
        bodies=named if filtering else (),
        scopes=scoped if filtering else (),
        suggested_bodies=() if filtering else named,
    )


def to_slice(context: ScoutingContext) -> dict[str, JSONValue]:
    """A atualização da fatia do pack (contracts/retrieval-filters.md): o contexto inteiro e os filtros de
    recuperação, que omitem o que não restringe."""
    full: dict[str, JSONValue] = {
        "branch": context.branch,
        "branches": list(context.branches),
        "topics": list(context.topics),
        "document_types": list(context.document_types),
        "suggested_types": list(context.suggested_types),
        "effective_from": context.validity.start.isoformat(),
        "effective_until": context.validity.end.isoformat(),
        "include_superseded": context.include_superseded,
        "previous_version": context.previous_version,
        "bodies": list(context.bodies),
        "scopes": list(context.scopes),
        "suggested_bodies": list(context.suggested_bodies),
        "catalog": None
        if context.catalog is None
        else {"kind": context.catalog.kind, "terms": list(context.catalog.terms)},
        "enumeration": None
        if context.enumeration is None
        else {"unit_kind": context.enumeration.unit_kind, "terms": list(context.enumeration.terms)},
    }
    filters = {key: value for key, value in full.items() if value is not None and value != []}
    return {CONTEXT_KEY: full, RETRIEVAL_FILTERS_KEY: filters}


def _matches(text: str, terms: tuple[str, ...]) -> bool:
    return any(contains_term(text, term) for term in terms)


def _branches(text: str, lexicon: Lexicon) -> tuple[str | None, tuple[str, ...]]:
    """Exatamente um ramo casado → filtro; marcador de totalidade → os cinco ramos como facetas; dois ou mais casados →
    eles como facetas; nenhum ou só termo ambíguo → nada."""
    if _matches(text, lexicon.all_branches):
        return None, tuple(lexicon.branches)
    found = tuple(branch for branch, terms in lexicon.branches.items() if _matches(text, terms))
    if len(found) == 1:
        return found[0], ()
    return None, found if len(found) > 1 else ()


def _cited_types(text: str, lexicon: Lexicon) -> tuple[str, ...]:
    """Os tipos que a pergunta nomeia, na ordem do léxico."""
    return tuple(kind for kind, terms in lexicon.document_types.items() if _matches(text, terms))


def _suggested_types(text: str, lexicon: Lexicon, topics: tuple[Topic, ...]) -> tuple[str, ...]:
    """Sem tipo citado, os que os sinais e os temas sugerem, na ordem do léxico; com tipo citado, nenhum — o filtro já
    decide (ADR-046)."""
    if _cited_types(text, lexicon):
        return ()
    suggested: set[str] = set()
    for term, kinds in lexicon.type_signals.items():
        if contains_term(text, term):
            suggested.update(kinds)
    for topic in topics:
        suggested.update(topic.document_types)
    return tuple(kind for kind in lexicon.document_types if kind in suggested)


def _catalog(text: str, lexicon: Lexicon) -> CatalogRequest | None:
    """Um marcador de enumeração e o plural de uma espécie pedem a lista (ADR-047); a primeira espécie citada vence.
    Sem palavra que qualifique, `documento` não pede nada — seria o acervo inteiro."""
    if not _matches(text, lexicon.list_markers):
        return None
    kind = next(
        (kind for kind, plurals in lexicon.catalog_kinds.items() if _matches(text, plurals)),
        None,
    )
    if kind is None:
        return None
    named = {kind for kind in lexicon.catalog_kinds} | {p for ps in lexicon.catalog_kinds.values() for p in ps}
    ignored = lexicon.catalog_stopwords | named
    terms: list[str] = []
    for word in text.split():
        if word not in ignored and word not in terms and (len(word) > 2 or word.isdigit()):
            terms.append(word)
    if kind == "documento" and not terms:
        return None
    return CatalogRequest(kind=kind, terms=tuple(terms))


def _enumeration(text: str, lexicon: Lexicon) -> EnumerationRequest | None:
    """Um marcador de enumeração que não pede documentos (esse é o `catalog`): a lista sai da estrutura dos documentos
    (ADR-048). O tipo de unidade é o primeiro cujo plural a pergunta usa; sem nenhum, `None` — a 103 busca então os
    trechos com forma de lista."""
    if not _matches(text, lexicon.list_markers):
        return None
    unit_kind = next((kind for kind, plurals in lexicon.unit_kinds.items() if _matches(text, plurals)), None)
    ignored = set(lexicon.catalog_stopwords)
    ignored.update(word for plurals in lexicon.unit_kinds.values() for plural in plurals for word in plural.split())
    ignored.update(word for terms in lexicon.branches.values() for term in terms for word in term.split())
    ignored.update(word for marker in lexicon.list_markers for word in marker.split())
    terms: list[str] = []
    for word in text.split():
        if word not in ignored and word not in terms and (len(word) > 2 or word.isdigit()):
            terms.append(word)
    return EnumerationRequest(unit_kind=unit_kind, terms=tuple(terms))
