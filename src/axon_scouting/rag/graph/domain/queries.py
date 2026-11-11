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
from axon_scouting.rag.graph.domain.model import GOVERNANCE, DomainGraph, EdgeRec

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
    """As arestas em que o alvo está numa das pontas, com a fonte: no G1, as relações de governança (o que
    `NormGraph.relations` dava). O trecho da fonte vem como `chunk_ids`."""
    wanted = set(relacoes) if relacoes else set(GOVERNANCE)
    found: list[Evidence] = []
    for edge in g.edges:
        if edge.type not in wanted or alvo not in (edge.from_, edge.to):
            continue
        doc_key = edge.source.split("#", 1)[0]
        version = g.versions.get(doc_key)
        found.append(
            Evidence(
                query="ligados",
                target=edge.to if edge.from_ == alvo else edge.from_,
                doc_key=doc_key,
                unit_id=None,
                chunk_ids=(edge.source,),
                role=version.role if version else "norma",
                scope=version.scope if version else None,
                date=version.date if version else None,
                edge=_ref(edge),
            )
        )
    return found


def _ref(edge: EdgeRec) -> EdgeRef:
    return EdgeRef(type=edge.type, source=edge.source, origin=edge.origin)


def _not_yet(name: str) -> Callable[..., list[Evidence]]:
    def query(g: DomainGraph, **_: Any) -> list[Evidence]:
        raise NotImplementedError(f"{name} chega num incremento seguinte (spec 106)")

    return query


QUERIES: Final[Mapping[QueryName, Callable[..., list[Evidence]]]] = {
    "sobre": sobre,
    "ligados": ligados,
    "aplicavel_a": aplicavel_a,
    "documentos": _not_yet("documentos"),
    "itens": _not_yet("itens"),
    "linha_do_tempo": linha_do_tempo,
}


def run_query(g: DomainGraph, name: QueryName, args: Mapping[str, Any]) -> list[Evidence]:
    if name not in QUERIES:
        raise PlanInvalid("q", f"consulta desconhecida: {name!r}")
    return QUERIES[name](g, **dict(args))
