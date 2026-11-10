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
    """As unidades com `SOBRE` para as entidades, na ordem: a posição da entidade em `entidades` (a da menção), o papel,
    o nível (de cima para baixo), o documento e a unidade — a ordem do `sorted` do ADR-063, estendida (contrato do seam,
    garantia 2). `modo="todas"`, `ramo` e `periodo` chegam no G2."""
    if modo != "alguma" or ramo is not None or periodo is not None:
        raise NotImplementedError("sobre: modo todas, ramo e período chegam no G2")
    found: list[tuple[tuple[int, int, int, str, str], Evidence]] = []
    for position, entity in enumerate(entidades):
        for edge in g.by_to.get(("SOBRE", entity), ()):
            unit = g.units.get(edge.from_)
            version = g.versions.get(unit.doc_key) if unit else None
            if unit is None or version is None:
                continue
            if papeis and version.role not in papeis:
                continue
            if tipos and version.type not in tipos:
                continue
            key = (
                position,
                ROLE_ORDER.index(version.role),
                scope_rank(SCOPES, version.scope),
                unit.doc_key,
                unit.unit_id,
            )
            found.append(
                (
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
            )
    return [evidence for _, evidence in sorted(found, key=lambda pair: pair[0])]


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
