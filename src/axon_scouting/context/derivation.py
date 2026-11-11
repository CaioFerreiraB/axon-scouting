"""Derivação de contexto da pergunta: ramo, temas, tipos de documento e vigência (spec 101 FR-004 a FR-006).

Função pura de (pergunta, perguntas anteriores, dia de referência, léxico) — sem modelo, sem relógio, sem I/O
(princípio V, SC-002). Só o tipo de documento citado filtra; o sugerido pelos temas só ordena (ADR-046). Os tipos, os
órgãos e a abrangência são os da taxonomia do acervo (ADR-049). O órgão filtra só quando é o **emissor** do tipo citado
ou da lista pedida ("as atas **da** assembleia regional"); quando é o **assunto** ("o que o regulamento diz sobre a
diretoria regional"), só desempata, e o nome dele não vira abrangência (ADR-061). Na dúvida, a derivação não filtra:
ramo ambíguo resulta em nenhum filtro de ramo (research R8). Mais de um ramo, ou um marcador de totalidade ("cada
ramo"), resulta em facetas — `branches` —, uma busca por ramo na 103 (ADR-045).

A pergunta de seguimento ("e no estatuto nacional?", "qual o prazo de gestão?") não tem assunto próprio: herda o da
última pergunta do usuário que tinha um — a **âncora**, publicada em `anchor` para a 103 buscar com ela (ADR-061). O
tipo citado, o período e a abrangência continuam sendo os da pergunta atual.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Final

from axon_ai_contracts import RETRIEVAL_FILTERS_KEY, JSONValue, TurnView

from axon_scouting.context.lexicon import LEXICON, Lexicon, Topic
from axon_scouting.context.periods import Validity, find_period
from axon_scouting.context.text import contains_term, normalize
from axon_scouting.rag.graph.api import EACH, QueryPlan, QueryStep, reconhecer

CONTEXT_KEY = "context"
"""Chave da fatia com o contexto derivado inteiro — o que a 104 e a 105 leem."""

ANCHOR_MAX_CHARS: Final = 400
"""A âncora vai inteira para o texto da busca; uma pergunta maior que isso é cortada."""
ANCHOR_MAX_TURNS: Final = 3
"""Quantas perguntas anteriores do usuário a âncora pode recuar."""


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
    anchor: str | None = None
    """A pergunta anterior que esta, de seguimento, retoma (ADR-061); `None` quando a pergunta tem assunto próprio."""
    synthesis: SynthesisRequest | None = None
    """Pedido de síntese ao longo das atas (ADR-062); `None` na pergunta pontual."""
    entities: tuple[str, ...] = ()
    """As entidades do grafo de domínio que a pergunta pontual nomeia, pelo termo mais longo, na ordem da menção
    (ADR-063, ADR-064). Vazio na síntese, na lista, na enumeração e na pergunta histórica (101 R43)."""
    plan: QueryPlan = field(default_factory=QueryPlan)
    """O plano de consultas do grafo pela forma da pergunta (ADR-064, 101 FR-037). No G1, `sobre` das entidades nas
    normas: a 103 traz as seções que as disciplinam, da norma de cima para a de baixo."""


@dataclass(frozen=True, slots=True, kw_only=True)
class SynthesisRequest:
    """A síntese pedida: os anos citados (um intervalo, ou nenhum), a abrangência e os órgãos nomeados — inclusive os
    que são assunto, e o nome do órgão vale como abrangência ("conselho consultivo **regional**")."""

    year_from: int | None
    year_until: int | None
    scopes: tuple[str, ...]
    bodies: tuple[str, ...]


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


def derive_context(
    question: str, reference_day: date, lexicon: Lexicon = LEXICON, *, previous: Sequence[str] = ()
) -> ScoutingContext:
    """`previous`: as perguntas anteriores do usuário na conversa, da mais antiga para a mais recente."""
    text = normalize(question)
    anchor = _anchor(text, previous, lexicon)
    anchored = text if anchor is None else f"{normalize(anchor)} {text}"
    topics = tuple(topic for topic in lexicon.topics if _matches(anchored, topic.terms))
    period = find_period(text, reference_day)
    if period is None and anchor is not None:
        period = find_period(normalize(anchor), reference_day)
    previous_version = _matches(text, lexicon.previous_markers)
    branch, branches = _branches(text, lexicon)
    if anchor is not None and branch is None and not branches:
        branch, branches = _branches(normalize(anchor), lexicon)
    catalog = _catalog(text, lexicon)
    listing = text  # o texto em que a lista foi pedida: é nele que o emissor se procura
    if catalog is None and anchor is not None:
        catalog = _catalog(anchored, lexicon)
        listing = anchored
    enumeration = None
    if catalog is None:
        enumeration = _enumeration(text, lexicon)
        if anchor is not None and (enumeration is None or enumeration.unit_kind is None):
            enumeration = _enumeration(anchored, lexicon) or enumeration
    cited = _cited_types(text, lexicon)
    named = tuple(body for body, terms in lexicon.bodies.items() if _matches(anchored, terms))
    filtering = bool(cited) or catalog is not None
    issuers = _issuers(listing, lexicon) if filtering else ()
    subjects = tuple(body for body in named if body not in issuers)
    scoped = tuple(
        scope for scope, terms in lexicon.scopes.items() if _matches(_without_bodies(text, subjects, lexicon), terms)
    )
    return ScoutingContext(
        branch=branch,
        branches=branches,
        topics=tuple(topic.id for topic in topics),
        document_types=cited,
        suggested_types=_suggested_types(anchored, lexicon, topics, cited=cited),
        # Consulta histórica (US3): o período citado, ou o dia de referência quando a pergunta mira a versão anterior.
        validity=period or Validity.day(reference_day),
        include_superseded=period is not None or previous_version,
        previous_version=previous_version,
        catalog=catalog,
        enumeration=enumeration,
        bodies=issuers,
        scopes=scoped if filtering else (),
        suggested_bodies=subjects,
        anchor=anchor,
        synthesis=(synthesis := _synthesis(anchored, reference_day, lexicon, named)),
        entities=(
            entities := ()
            if catalog is not None
            or (enumeration is not None and (synthesis is None or enumeration.unit_kind is not None))
            or (period is not None and synthesis is None)
            else reconhecer(anchored, lexicon.entity_terms)
        ),
        plan=(
            _plan(entities, branch, branches)
            if synthesis is None
            else _timeline(entities, synthesis, _matches(text, lexicon.election_terms))
        ),
    )


def earlier_questions(view: TurnView) -> tuple[str, ...]:
    """As perguntas anteriores do usuário na conversa, da mais antiga para a mais recente, sem a atual — a última
    mensagem `user` quando ela é a própria pergunta."""
    asked = [m["content"] for m in view.messages if m.get("role") == "user" and isinstance(m.get("content"), str)]
    if asked and asked[-1].strip() == view.question.strip():
        asked = asked[:-1]
    return tuple(question for question in asked if question.strip())


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
        "anchor": context.anchor,
        "entities": list(context.entities),
        "plan": None if context.plan.empty else context.plan.to_json(),
        "synthesis": None
        if context.synthesis is None
        else {
            "year_from": context.synthesis.year_from,
            "year_until": context.synthesis.year_until,
            "scopes": list(context.synthesis.scopes),
            "bodies": list(context.synthesis.bodies),
        },
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


def _suggested_types(
    text: str, lexicon: Lexicon, topics: tuple[Topic, ...], *, cited: tuple[str, ...]
) -> tuple[str, ...]:
    """Sem tipo citado, os que os sinais e os temas sugerem, na ordem do léxico; com tipo citado, nenhum — o filtro já
    decide (ADR-046)."""
    if cited:
        return ()
    suggested: set[str] = set()
    for term, kinds in lexicon.type_signals.items():
        if contains_term(text, term):
            suggested.update(kinds)
    for topic in topics:
        suggested.update(topic.document_types)
    return tuple(kind for kind in lexicon.document_types if kind in suggested)


def _issuers(text: str, lexicon: Lexicon) -> tuple[str, ...]:
    """Os órgãos nomeados como emissores: logo depois de um tipo ("resoluções **da** diretoria regional"), com no
    máximo uma palavra de complemento entre eles ("regimento interno **do** conselho"), na ordem do léxico (ADR-061)."""
    words = text.split()
    kinds = [term.split() for terms in lexicon.document_types.values() for term in terms]
    kinds += [plural.split() for plurals in lexicon.catalog_kinds.values() for plural in plurals]
    fillers = set(lexicon.issuer_fillers) | {
        term for terms in lexicon.scopes.values() for term in terms if " " not in term
    }
    found: list[str] = []
    for body, terms in lexicon.bodies.items():
        for term in terms:
            if any(
                _issued(words[:start], kinds, lexicon.issuer_links, fillers) for start in _starts(words, term.split())
            ):
                found.append(body)
                break
    return tuple(found)


def _issued(before: list[str], kinds: list[list[str]], links: tuple[str, ...], fillers: set[str]) -> bool:
    """`before` termina num tipo seguido de ligação ("atas da"), com uma palavra de complemento opcional antes da
    ligação ("regulamento regional da"), ou num termo de tipo que já traz a ligação ("parecer da")."""
    if any(_ends_with(before, kind) and kind[-1] in links for kind in kinds):
        return True
    if not before or before[-1] not in links:
        return False
    rest = before[:-1]
    candidates = [rest, rest[:-1]] if rest and rest[-1] in fillers else [rest]
    return any(_ends_with(candidate, kind) for candidate in candidates for kind in kinds)


def _starts(words: list[str], term: list[str]) -> list[int]:
    return [i for i in range(len(words) - len(term) + 1) if words[i : i + len(term)] == term]


def _ends_with(words: list[str], term: list[str]) -> bool:
    return len(words) >= len(term) and words[len(words) - len(term) :] == term


def _without_bodies(text: str, bodies: tuple[str, ...], lexicon: Lexicon) -> str:
    """O texto sem os nomes dos órgãos que são assunto: "diretoria **regional**" não é a abrangência pedida."""
    padded = f" {text} "
    for body in bodies:
        for term in sorted(lexicon.bodies[body], key=len, reverse=True):
            padded = padded.replace(f" {term} ", " ")
    return " ".join(padded.split())


def _anchor(text: str, previous: Sequence[str], lexicon: Lexicon) -> str | None:
    """A âncora da pergunta de seguimento: a última pergunta anterior que tem assunto próprio, recuando até
    `ANCHOR_MAX_TURNS`; nenhuma, `None`."""
    if not previous or not _follows_up(text, lexicon):
        return None
    for earlier in reversed(previous[-ANCHOR_MAX_TURNS:]):
        normalized = normalize(earlier)
        if normalized and not _follows_up(normalized, lexicon):
            return " ".join(earlier.split())[:ANCHOR_MAX_CHARS]
    return None


def _follows_up(text: str, lexicon: Lexicon) -> bool:
    """Sem assunto próprio (tema, órgão, espécie de lista ou tipo de unidade) e curta ou com marcador de retomada. O
    ramo e o tipo citado não são assunto: "e no estatuto?", "e para o sênior?" retomam a anterior."""
    if not text or _has_subject(text, lexicon):
        return False
    return (
        len(text.split()) <= lexicon.follow_up_max_words
        or _matches(text, lexicon.follow_up_markers)
        or any(text == opener or text.startswith(f"{opener} ") for opener in lexicon.follow_up_openers)
    )


def _has_subject(text: str, lexicon: Lexicon) -> bool:
    return (
        any(_matches(text, topic.terms) for topic in lexicon.topics)
        or any(_matches(text, terms) for terms in lexicon.bodies.values())
        or any(_matches(text, plurals) for plurals in lexicon.catalog_kinds.values())
        or any(_matches(text, plurals) for plurals in lexicon.unit_kinds.values())
    )


ORGAN_PREFIXES: Final = ("tipo_orgao:", "tipo_inst:", "orgao:", "inst:")
INSTITUTION_PREFIX: Final = "inst:"
DOCUMENT_PREFIX: Final = "doc:"
CONTENT_ROLES: Final = ["norma", "orientacao"]


def _plan(entities: tuple[str, ...], branch: str | None, branches: tuple[str, ...]) -> QueryPlan:
    """A forma da pergunta → o plano (101 FR-037, research R44; 106 G2, G3):
    - órgãos e tipos de instituição: um `sobre` com eles, nas normas — o do ADR-063; com uma instituição real citada
      (`inst:`), o plano é a interseção com `aplicavel_a` dela (106 US3): as normas que valem para ela e tratam do
      órgão. A instituição não entra no `sobre`: as partes dela tirariam as vagas do órgão (106 R20);
    - "cada ramo" (as facetas): `para_cada` ramo, com as outras entidades na mesma unidade (`todas`);
    - um ramo citado e outras entidades, ou duas entidades ou mais: as unidades que tratam de todas (`todas`) e, com
      duas entidades ou mais, também as de cada uma (a unidade que nomeia uma só pode ter a resposta); a instituição
      citada junto não muda esse plano — o assunto são as entidades;
    - a instituição real sozinha: a interseção do `sobre` dela (o documento dela, o tipo e as partes) com `aplicavel_a`.
    Normas e orientações. Uma entidade sozinha (um conceito, um cargo) ou o ramo sozinho não pede plano: a busca é mais
    precisa que todas as unidades sobre ela (106 R19, o s01 da semente)."""
    entities = tuple(entity for entity in entities if not entity.startswith(DOCUMENT_PREFIX))  # alvo só da síntese
    institutions = [entity for entity in entities if entity.startswith(INSTITUTION_PREFIX)]
    organs = [
        entity for entity in entities if entity.startswith(ORGAN_PREFIXES) and not entity.startswith(INSTITUTION_PREFIX)
    ]
    others = [entity for entity in entities if not entity.startswith(ORGAN_PREFIXES)]
    if organs or (institutions and not others):
        subject = QueryStep(q="sobre", args={"entidades": organs or institutions[:1], "papeis": ["norma"]})
        if not institutions:
            return QueryPlan(consultas=(subject,))
        applies = QueryStep(q="aplicavel_a", args={"instituicao": institutions[0]})
        return QueryPlan(forma="intersecao", consultas=(subject, applies))
    steps: list[QueryStep] = []
    if len(branches) >= 2:
        args: dict[str, JSONValue] = {"entidades": [EACH, *others[:3]], "papeis": list(CONTENT_ROLES)}
        if others:
            args["modo"] = "todas"
        steps.append(QueryStep(q="sobre", args=args))
        return QueryPlan(forma="para_cada", consultas=tuple(steps), para_cada=tuple(f"ramo:{b}" for b in branches))
    joint = [*([f"ramo:{branch}"] if branch else []), *others][:4]
    if others and len(joint) >= 2:
        steps.append(QueryStep(q="sobre", args={"entidades": joint, "modo": "todas", "papeis": list(CONTENT_ROLES)}))
    if len(others) >= 2:
        steps.append(QueryStep(q="sobre", args={"entidades": others[:4], "papeis": list(CONTENT_ROLES)}))
    return QueryPlan(consultas=tuple(steps))


TIMELINE_EXCLUDED: Final = ("inst:", "tipo_inst:")
"""Na síntese, a instituição ("a Região de São Paulo") situa a pergunta e não é o alvo da linha do tempo."""


def _timeline(entities: tuple[str, ...], synthesis: SynthesisRequest, elections: bool) -> QueryPlan:
    """A síntese com uma entidade reconhecida → a linha do tempo dela (106 G4, US4): os eventos das atas que tratam da
    entidade, por data. Com duas, a interseção das linhas do tempo — os eventos da primeira nas atas que também têm
    eventos da segunda ("os delegados à Assembleia Nacional que a Assembleia Regional elegeu"). Palavra de eleição →
    só os eventos de eleição. O período é o citado, do primeiro ao último ano. Sem entidade, plano vazio: a busca de
    resumos responde sozinha, como no ADR-062."""
    targets = [entity for entity in entities if not entity.startswith(TIMELINE_EXCLUDED)][:2]
    if not targets:
        return QueryPlan()
    args: dict[str, JSONValue] = {}
    if elections:
        args["eventos"] = ["eleicao"]
    if synthesis.year_from is not None:
        until = synthesis.year_until or synthesis.year_from
        args["periodo"] = {"de": f"{synthesis.year_from}-01-01", "ate": f"{until}-12-31"}
    steps = tuple(QueryStep(q="linha_do_tempo", args={"alvo": target, **args}) for target in targets)
    return QueryPlan(forma="intersecao" if len(steps) > 1 else "simples", consultas=steps)


_YEAR: Final = re.compile(r"\b(19[89]\d|20\d\d)\b")
_DECADE: Final = re.compile(r"\banos (2000|2010|2020|80|90)\b")


def _synthesis(text: str, reference_day: date, lexicon: Lexicon, bodies: tuple[str, ...]) -> SynthesisRequest | None:
    """Síntese: um marcador ("ao longo", "quem foram", "onde se reuniu") ou uma palavra de reunião com ano citado
    ("a assembleia de 2016"). Os anos: o menor e o maior citados; "desde 2016" vai até o ano de referência; "nos anos
    2000", a década."""
    years = [int(year) for year in _YEAR.findall(text)]
    for decade in _DECADE.findall(text):
        start = {"80": 1980, "90": 1990}.get(decade, int(decade))
        years += [start, start + 9]
    marked = _matches(text, lexicon.synthesis_markers)
    if not marked and not (years and _matches(text, lexicon.meeting_terms)):
        return None
    year_from = min(years) if years else None
    year_until = max(years) if years else None
    if year_from is not None and contains_term(text, f"desde {year_from}"):
        year_until = max(year_until or year_from, reference_day.year)
    return SynthesisRequest(
        year_from=year_from,
        year_until=year_until,
        scopes=tuple(scope for scope, terms in lexicon.scopes.items() if _matches(text, terms)),
        bodies=bodies,
    )


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
