"""As consultas com nome do grafo de domínio (spec 106 FR-009; research R7; design.md, seção 3).

Funções puras sobre o `DomainGraph`: nenhuma faz I/O nem lê relógio. Devolvem evidência, e não trechos. `sobre` (G1,
G2, G3), `aplicavel_a` (G3), `linha_do_tempo` (G4) e `ligados` (as relações de governança); as demais chegam nos
incrementos seguintes.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import date
from typing import Any, Final

from axon_scouting.rag.graph.api import (
    ROLE_ORDER,
    EdgeRef,
    EventRef,
    Evidence,
    PlanInvalid,
    QueryName,
    scope_rank,
)
from axon_scouting.rag.graph.domain.model import GOVERNANCE, LINKS, DomainGraph, EdgeRec, VersionRec

SCOPES: Final = ("mundial", "interamericano", "nacional", "regional", "local")
"""A norma de cima primeiro (ADR-061, ADR-063): a regra do nível superior abre a resposta."""


def sobre(
    g: DomainGraph,
    entidades: Sequence[str],
    *,
    modo: str = "alguma",
    papeis: Sequence[str] = (),
    tipos: Sequence[str] = (),
    ramo: str | None = None,
    periodo: Mapping[str, str] | None = None,
) -> list[Evidence]:
    """As unidades com `SOBRE` para as entidades, em camadas:
    0. as unidades da entidade e das seções do ramo dela (`DE_RAMO`); a instituição real inclui o tipo dela
       (`INSTANCIA_DE`) e as partes do tipo (`PARTE_DE`: o que trata da assembleia de grupo trata do grupo; 106 R20);
    1. as do termo genérico (`E_UM`: o que vale para os adultos vale para o escotista);
    2. os documentos inteiros das áreas que regem a entidade (`REGE` + `DE_AREA`; 106 R19).
    Em cada camada, a ordem é a posição da entidade em `entidades` (a da menção), o papel, o nível (de cima para baixo),
    o documento e a unidade — a ordem do `sorted` do ADR-063, estendida (contrato do seam, garantia 2).

    `modo="todas"`: só a camada 0, e só as unidades em que todas as entidades estão — nela mesma ou nas unidades acima
    dela (o documento de um ramo conta como acima); a entidade específica satisfaz o termo genérico dela. `ramo` e
    `periodo` ainda não são usados pelo plano (G3, G4)."""
    if ramo is not None or periodo is not None:
        raise NotImplementedError("sobre: ramo e período chegam com o G3 e o G4")
    found: dict[str, tuple[tuple[int, int, int, int, str, str], Evidence]] = {}
    required = {_root(g, entity) for entity in entidades}

    def add(layer: int, position: int, entity: str, unit_id: str, edge: EdgeRec) -> None:
        unit = g.units.get(unit_id)
        version = g.versions.get(unit.doc_key) if unit else None
        if unit is None or version is None:
            return
        if (papeis and version.role not in papeis) or (tipos and version.type not in tipos):
            return
        if modo == "todas" and not required <= _context(g, unit.unit_id):
            return
        ident = unit.unit_id if modo == "todas" else f"{position}|{unit.unit_id}"
        if ident in found:
            return
        key = (
            layer,
            0 if modo == "todas" else position,
            ROLE_ORDER.index(version.role),
            scope_rank(SCOPES, version.scope),
            unit.doc_key,
            unit.unit_id,
        )
        found[ident] = (
            key,
            Evidence(
                query="sobre",
                target=entity,
                doc_key=unit.doc_key,
                unit_id=unit.unit_id,
                chunk_ids=unit.chunk_ids,
                role=version.role,
                scope=version.scope,
                date=version.date,
                edge=_ref(edge),
            ),
        )

    for position, entity in enumerate(entidades):
        for target in _with_children(g, entity):
            for edge in g.by_to.get(("SOBRE", target), ()):
                add(0, position, entity, edge.from_, edge)
        if modo == "todas":
            continue
        for generic in _generics(g, entity):
            for edge in g.by_to.get(("SOBRE", generic), ()):
                add(1, position, entity, edge.from_, edge)
        for area in _areas(g, entity):
            for edge in g.by_to.get(("DE_AREA", area), ()):
                add(2, position, entity, f"{edge.from_}#documento", edge)
    ranked = [evidence for _, evidence in sorted(found.values(), key=lambda pair: pair[0])]
    return _strongest(g, ranked, required) if modo == "todas" else ranked


def _strongest(g: DomainGraph, evidence: list[Evidence], required: set[str]) -> list[Evidence]:
    """No `modo="todas"`, a unidade que nomeia todas as entidades no próprio caminho vence a que só as alcança porque
    o documento inteiro é do ramo: num documento com alguma das primeiras, ficam só elas (106 R19)."""
    strong = {e.unit_id for e in evidence if e.unit_id and required <= _context(g, e.unit_id, documents=False)}
    with_strong = {e.doc_key for e in evidence if e.unit_id in strong}
    return [e for e in evidence if e.doc_key not in with_strong or e.unit_id in strong]


def _with_children(g: DomainGraph, entity: str) -> list[str]:
    """A entidade e as seções do ramo (`DE_RAMO`): o que trata da alcateia trata do Ramo Lobinho. A instância inclui o
    tipo (`INSTANCIA_DE`); a instituição real, também as partes dela e do tipo (`PARTE_DE`) — só a instituição citada
    numa pergunta: a menção do tipo ("grupo escoteiro") fica com as unidades dele, como no G1."""
    found = [entity, *(edge.from_ for edge in g.by_to.get(("DE_RAMO", entity), ()))]
    kinds = [edge.to for edge in g.by_from.get(("INSTANCIA_DE", entity), ())]
    found.extend(kinds)
    if entity.startswith("inst:"):
        for whole in (entity, *kinds):
            found.extend(edge.from_ for edge in g.by_to.get(("PARTE_DE", whole), ()))
    return list(dict.fromkeys(found))


def _root(g: DomainGraph, entity: str) -> str:
    """A seção vale pelo ramo dela; as outras entidades, por elas mesmas."""
    edges = g.by_from.get(("DE_RAMO", entity), ())
    return edges[0].to if edges else entity


def _generics(g: DomainGraph, entity: str) -> list[str]:
    """Os termos genéricos da entidade (`E_UM`): o escotista é um adulto."""
    return [edge.to for edge in g.by_from.get(("E_UM", entity), ())]


def _areas(g: DomainGraph, entity: str) -> list[str]:
    """As áreas que regem a entidade ou o termo genérico dela (`REGE`)."""
    targets = (entity, *_generics(g, entity))
    return list(dict.fromkeys(edge.from_ for target in targets for edge in g.by_to.get(("REGE", target), ())))


def _context(g: DomainGraph, unit_id: str, *, documents: bool = True) -> set[str]:
    """As entidades de que a unidade trata, ela e as de cima (com `documents=False`, sem o documento inteiro), com os
    termos genéricos delas: a unidade sobre escotistas trata de adultos."""
    unit = g.units.get(unit_id)
    found: set[str] = set()
    for node in (unit_id, *(unit.ancestors if unit else ())):
        if not documents and node.endswith("#documento"):
            continue
        for edge in g.by_from.get(("SOBRE", node), ()):
            found.add(_root(g, edge.to))
            found.update(_generics(g, edge.to))
    return found


def aplicavel_a(g: DomainGraph, instituicao: str, *, papeis: Sequence[str] = ("norma",)) -> list[Evidence]:
    """Os documentos que valem para a instituição: os com `APLICA_SE_A` para ela e para cada uma acima dela pelo
    `PARTE_DE` (até 5 níveis), de cima para baixo (o nível do documento; depois o `doc_key`). Instituição fora do grafo:
    nenhum."""
    chain = [instituicao, *_above(g, instituicao)]
    found: list[tuple[tuple[int, str], Evidence]] = []
    for institution in chain:
        for edge in g.by_to.get(("APLICA_SE_A", institution), ()):
            version = g.versions.get(edge.from_)
            if version is None or (papeis and version.role not in papeis):
                continue
            found.append(
                (
                    (scope_rank(SCOPES, version.scope), version.doc_key),
                    Evidence(
                        query="aplicavel_a",
                        target=institution,
                        doc_key=version.doc_key,
                        unit_id=None,
                        chunk_ids=(),
                        role=version.role,
                        scope=version.scope,
                        date=version.date,
                        edge=_ref(edge),
                    ),
                )
            )
    return [evidence for _, evidence in sorted(found, key=lambda pair: pair[0])]


def _above(g: DomainGraph, start: str, depth: int = 5) -> list[str]:
    """As instituições acima pelo `PARTE_DE`, de baixo para cima, até `depth` níveis."""
    found: list[str] = []
    frontier = [start]
    for _ in range(depth):
        frontier = [
            edge.to
            for node in frontier
            for edge in g.by_from.get(("PARTE_DE", node), ())
            if edge.to not in found and edge.to != start
        ]
        if not frontier:
            break
        found.extend(dict.fromkeys(frontier))
    return found


def linha_do_tempo(
    g: DomainGraph,
    alvo: str,
    *,
    eventos: Sequence[str] = (),
    periodo: Mapping[str, str] | None = None,
) -> list[Evidence]:
    """Os eventos `DE` o alvo — e, sendo ele um tipo, `DE` as instâncias dele (a Diretoria Regional inclui a DR-SP) —,
    por data (sem data, por último). `eventos` filtra pelo tipo; `periodo` (`de`/`ate`, datas ISO, cada um opcional)
    fica com os datados dentro dele. Cada evidência traz o evento e os trechos da ata na página dele."""
    targets = [alvo, *(edge.from_ for edge in g.by_to.get(("INSTANCIA_DE", alvo), ()))]
    low = date.fromisoformat(periodo["de"]) if periodo and periodo.get("de") else None
    high = date.fromisoformat(periodo["ate"]) if periodo and periodo.get("ate") else None
    found: dict[str, tuple[tuple[date, str], Evidence]] = {}
    for target in targets:
        for edge in g.by_to.get(("DE", target), ()):
            event = g.events.get(edge.from_)
            if event is None or event.id in found or (eventos and event.type not in eventos):
                continue
            if (low or high) and (event.date is None or (low and event.date < low) or (high and event.date > high)):
                continue
            version = g.versions.get(event.doc_key)
            found[event.id] = (
                (event.date or date.max, event.id),
                Evidence(
                    query="linha_do_tempo",
                    target=alvo,
                    doc_key=event.doc_key,
                    unit_id=None,
                    chunk_ids=event.chunk_ids,
                    role=version.role if version else "registro",
                    scope=version.scope if version else None,
                    date=event.date,
                    edge=_ref(edge),
                    event=EventRef(type=event.type, text=event.text, page=event.page),
                ),
            )
    return [evidence for _, evidence in sorted(found.values(), key=lambda pair: pair[0])]


def ligados(g: DomainGraph, alvo: str, *, relacoes: Sequence[str] = ()) -> list[Evidence]:
    """As arestas de governança e as ligações (`ALTERA`, `REVOGA`, `SUBSTITUI`, `CITA`) com o alvo numa ponta, com a
    fonte (sem `relacoes`, todas). O alvo pode ser:
    - uma unidade ou uma versão (`doc_key`): ela e, na versão, as unidades dela;
    - um documento (`doc:<document_id>`): as versões dele e as unidades delas;
    - outra entidade: ela (a governança) e as unidades com `SOBRE` para ela ("o que mudou nas regras dos filhotes").
    A evidência é a outra ponta: a unidade (com os trechos dela), a versão (o trecho que escreve a ligação primeiro,
    depois os da versão) ou, na governança, o trecho da fonte. Na ordem do arquivo (a do catálogo)."""
    wanted = tuple(relacoes) if relacoes else (*sorted(GOVERNANCE), *LINKS)
    ends = _ends(g, alvo)
    position = {id(edge): index for index, edge in enumerate(g.edges)}
    found: dict[tuple[str, str, str], tuple[int, Evidence]] = {}
    for kind in wanted:
        for end in ends:
            for edge, other in (
                *((e, e.to) for e in g.by_from.get((kind, end), ())),
                *((e, e.from_) for e in g.by_to.get((kind, end), ())),
            ):
                if other in ends or (edge.type, edge.source, other) in found:
                    continue
                found[(edge.type, edge.source, other)] = (position[id(edge)], _linked(g, edge, other))
    return [evidence for _, evidence in sorted(found.values(), key=lambda pair: pair[0])]


def _ends(g: DomainGraph, alvo: str) -> set[str]:
    if alvo.startswith("doc:"):
        keys = {key for key, version in g.versions.items() if version.document_id == alvo[4:]}
    elif alvo in g.versions:
        keys = {alvo}
    else:
        keys = set()
    if keys:
        return keys | {unit_id for unit_id, unit in g.units.items() if unit.doc_key in keys}
    if alvo in g.units:
        return {alvo}
    return {alvo, *(edge.from_ for target in _with_children(g, alvo) for edge in g.by_to.get(("SOBRE", target), ()))}


def _linked(g: DomainGraph, edge: EdgeRec, other: str) -> Evidence:
    unit = g.units.get(other)
    if unit is not None:
        doc_key, unit_id, chunk_ids = unit.doc_key, unit.unit_id, unit.chunk_ids
    elif other in g.versions:
        own = (edge.source,) if edge.source.startswith(f"{other}#") else ()
        doc_key, unit_id, chunk_ids = other, None, tuple(dict.fromkeys((*own, *g.versions[other].chunk_ids)))
    else:  # governança: a entidade da outra ponta; a evidência é o trecho da fonte
        doc_key, unit_id, chunk_ids = edge.source.split("#", 1)[0], None, (edge.source,)
    version = g.versions.get(doc_key)
    return Evidence(
        query="ligados",
        target=other,
        doc_key=doc_key,
        unit_id=unit_id,
        chunk_ids=chunk_ids,
        role=version.role if version else "norma",
        scope=version.scope if version else None,
        date=version.date if version else None,
        edge=_ref(edge),
    )


def documentos(
    g: DomainGraph,
    *,
    tipo: str | None = None,
    emissor: str | None = None,
    aplica_se_a: str | None = None,
    area: str | None = None,
    ramo: str | None = None,
    periodo: Mapping[str, str] | None = None,
    vigente: bool = True,
) -> list[Evidence]:
    """Um item por documento (`unit_id=None`, com os trechos da versão), filtrado pelo tipo, pelo emissor
    (`EMITIDO_POR`: o órgão; o tipo de órgão vale pelas instâncias dele, e a instituição pelos órgãos dela), pela
    instituição a que se aplica (a de `aplicavel_a`), pela área (`DE_AREA`), pelo ramo (uma unidade com `SOBRE` para
    ele ou para uma seção dele) e pelo período da vigência. Do mais recente ao mais antigo, como a lista de documentos
    (ADR-047), e depois pelo `doc_key` (G6)."""
    chosen: dict[str, EdgeRef] = {
        key: EdgeRef(type="VERSAO", source=key, origin="regra")
        for key, version in g.versions.items()
        if (tipo is None or version.type == tipo) and (not vigente or version.status == "vigente")
    }
    if emissor is not None:
        issuers = _issuers(g, emissor)
        by_issuer = {edge.from_: _ref(edge) for issuer in issuers for edge in g.by_to.get(("EMITIDO_POR", issuer), ())}
        chosen = {key: by_issuer[key] for key in chosen if key in by_issuer}
    if aplica_se_a is not None:
        applies = {e.doc_key: e.edge for e in aplicavel_a(g, aplica_se_a, papeis=ROLE_ORDER)}
        chosen = {key: ref for key, ref in chosen.items() if key in applies}
    if area is not None:
        areas = {edge.from_: _ref(edge) for edge in g.by_to.get(("DE_AREA", area), ())}
        chosen = {key: ref for key, ref in chosen.items() if key in areas}
    if ramo is not None:
        branched = {
            g.units[edge.from_].doc_key
            for target in _with_children(g, ramo)
            for edge in g.by_to.get(("SOBRE", target), ())
            if edge.from_ in g.units
        }
        chosen = {key: ref for key, ref in chosen.items() if key in branched}
    if periodo is not None:
        start, until = periodo.get("de"), periodo.get("ate")
        chosen = {key: ref for key, ref in chosen.items() if _within(g.versions[key], start, until)}
    ordered = sorted(
        chosen,
        key=lambda key: (-(d.toordinal() if (d := g.versions[key].date) else 0), key),
    )
    return [
        Evidence(
            query="documentos",
            target=emissor or aplica_se_a or area or ramo or tipo or key,
            doc_key=key,
            unit_id=None,
            chunk_ids=g.versions[key].chunk_ids,
            role=g.versions[key].role,
            scope=g.versions[key].scope,
            date=g.versions[key].date,
            edge=chosen[key],
        )
        for key in ordered
    ]


def itens(
    g: DomainGraph, *, tipo_de_unidade: str | None = None, de: str | None = None, ramo: str | None = None
) -> list[Evidence]:
    """As unidades de um tipo com nome (ADR-048), na ordem do documento — `doc_key`, depois a posição na árvore —,
    com o ramo da unidade quando `ramo` vem (`ramo:senior` ou `senior`); ou, com `de`, as entidades filhas dele (os
    órgãos de uma instituição, pelo `PARTE_DE`), com o trecho da fonte de cada uma (G6)."""
    if de is not None:
        found: list[Evidence] = []
        for edge in g.by_to.get(("PARTE_DE", de), ()):
            child = g.entities.get(edge.from_)
            doc_key = edge.source.split("#", 1)[0]
            version = g.versions.get(doc_key)
            if child is None or version is None:
                continue
            found.append(
                Evidence(
                    query="itens",
                    target=child.id,
                    doc_key=doc_key,
                    unit_id=None,
                    chunk_ids=(edge.source,),
                    role=version.role,
                    scope=version.scope,
                    date=version.date,
                    edge=_ref(edge),
                )
            )
        return found
    branch = ramo.removeprefix("ramo:") if ramo else None
    chosen = sorted(
        (
            unit
            for unit in g.units.values()
            if unit.position is not None
            and (tipo_de_unidade is None or unit.unit_kind == tipo_de_unidade)
            and (branch is None or branch in unit.branches)
        ),
        key=lambda unit: (unit.doc_key, unit.position or 0),
    )
    contains = {edge.to: edge for edge in g.edges if edge.type == "CONTEM"}
    return [
        Evidence(
            query="itens",
            target=tipo_de_unidade or unit.unit_kind,
            doc_key=unit.doc_key,
            unit_id=unit.unit_id,
            chunk_ids=unit.chunk_ids,
            role=g.versions[unit.doc_key].role,
            scope=g.versions[unit.doc_key].scope,
            date=g.versions[unit.doc_key].date,
            edge=_ref(contains[unit.unit_id]),
        )
        for unit in chosen
        if unit.doc_key in g.versions and unit.unit_id in contains
    ]


def _issuers(g: DomainGraph, emissor: str) -> set[str]:
    """O órgão; as instâncias do tipo de órgão; os órgãos (e as instâncias deles) da instituição."""
    found = {emissor, *(edge.from_ for edge in g.by_to.get(("INSTANCIA_DE", emissor), ()))}
    if emissor.startswith(("inst:", "tipo_inst:")):
        found |= {edge.from_ for edge in g.by_to.get(("PARTE_DE", emissor), ())}
    return found


def _within(version: VersionRec, start: str | None, until: str | None) -> bool:
    if version.date is None:
        return start is None and until is None
    return (start is None or version.date >= date.fromisoformat(start)) and (
        until is None or version.date <= date.fromisoformat(until)
    )


def _ref(edge: EdgeRec) -> EdgeRef:
    return EdgeRef(type=edge.type, source=edge.source, origin=edge.origin)


QUERIES: Final[Mapping[QueryName, Callable[..., list[Evidence]]]] = {
    "sobre": sobre,
    "ligados": ligados,
    "aplicavel_a": aplicavel_a,
    "documentos": documentos,
    "itens": itens,
    "linha_do_tempo": linha_do_tempo,
}


def run_query(g: DomainGraph, name: QueryName, args: Mapping[str, Any]) -> list[Evidence]:
    if name not in QUERIES:
        raise PlanInvalid("q", f"consulta desconhecida: {name!r}")
    return QUERIES[name](g, **dict(args))
