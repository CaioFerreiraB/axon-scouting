"""As consultas com nome do grafo de domínio (spec 106 FR-009; research R7; design.md, seção 3).

Funções puras sobre o `DomainGraph`: nenhuma faz I/O nem lê relógio. Devolvem evidência, e não trechos. No G1, `sobre`
cobre as arestas que o ADR-063 já tinha (órgão e instituição → seções das normas) e `ligados`, as relações de
governança; as demais chegam nos incrementos seguintes.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Final

from axon_scouting.rag.graph.api import (
    ROLE_ORDER,
    EdgeRef,
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
    0. as unidades da entidade e das seções do ramo dela (`DE_RAMO`);
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
    """A entidade e as seções do ramo (`DE_RAMO`): o que trata da alcateia trata do Ramo Lobinho."""
    return [entity, *(edge.from_ for edge in g.by_to.get(("DE_RAMO", entity), ()))]


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
    "aplicavel_a": _not_yet("aplicavel_a"),
    "documentos": _not_yet("documentos"),
    "itens": _not_yet("itens"),
    "linha_do_tempo": _not_yet("linha_do_tempo"),
}


def run_query(g: DomainGraph, name: QueryName, args: Mapping[str, Any]) -> list[Evidence]:
    if name not in QUERIES:
        raise PlanInvalid("q", f"consulta desconhecida: {name!r}")
    return QUERIES[name](g, **dict(args))
