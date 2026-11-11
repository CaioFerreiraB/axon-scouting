"""O executor do plano (spec 106 FR-012; research R9; design.md, seção 5).

Roda as consultas do plano e devolve as frentes: a evidência de cada passo, ordenada, cortada em até 3 documentos, sem
as relações de governança (FR-007). Não escolhe trechos nem conta vagas — isso é da 103, que lida com a pergunta.
Puro.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from axon_scouting.rag.graph.api import EACH, Evidence, Frentes, FrontGroup, QueryName, QueryPlan
from axon_scouting.rag.graph.domain.model import GOVERNANCE, DomainGraph
from axon_scouting.rag.graph.domain.queries import run_query

DOCUMENTS_PER_QUERY: Final[int | None] = None
"""O executor entrega a evidência inteira, na ordem do papel e do nível; o corte em 3 documentos é da 103, que os
ordena também pela relevância à pergunta (106 R19)."""


def executar(g: DomainGraph, plano: QueryPlan) -> Frentes:
    """`simples`: as consultas na ordem; `para_cada`: cada consulta que usa `$cada` roda uma vez por entidade (um grupo
    por entidade; as outras rodam uma vez); `intersecao`: a evidência da primeira consulta fica só nos documentos que
    todas as seguintes trazem. Em `para_cada`, um `sobre` com `modo="todas"` que não acha nada cai para a entidade do
    grupo (o ramo)."""
    groups: list[FrontGroup] = []
    if plano.forma == "intersecao" and plano.consultas:
        first, *rest = plano.consultas
        evidence = _run(g, first.q, first.args)
        for step in rest:
            allowed = {e.doc_key for e in _run(g, step.q, step.args)}
            evidence = [e for e in evidence if e.doc_key in allowed]
        return (FrontGroup(step=0, query=first.q, entity=None, evidence=first_documents(evidence)),)
    for index, step in enumerate(plano.consultas):
        uses_each = plano.forma == "para_cada" and _mentions_each(step.args)
        for entity in plano.para_cada if uses_each else (None,):
            args = _replace_each(step.args, entity) if entity else step.args
            evidence = _run(g, step.q, args, fallback=uses_each)
            groups.append(FrontGroup(step=index, query=step.q, entity=entity, evidence=first_documents(evidence)))
    return tuple(groups)


def _run(g: DomainGraph, name: QueryName, args: Mapping[str, Any], *, fallback: bool = False) -> list[Evidence]:
    evidence = [e for e in run_query(g, name, args) if e.edge.type not in GOVERNANCE]
    if fallback and not evidence and name == "sobre" and args.get("modo") == "todas" and args.get("entidades"):
        first = list(args["entidades"])[:1]
        evidence = [e for e in run_query(g, name, {**args, "entidades": first, "modo": "alguma"})]
    return evidence


def _mentions_each(args: Mapping[str, Any]) -> bool:
    return any(value == EACH or (isinstance(value, list) and EACH in value) for value in args.values())


def _replace_each(args: Mapping[str, Any], entity: str) -> dict[str, Any]:
    return {
        key: (
            entity
            if value == EACH
            else [entity if v == EACH else v for v in value]
            if isinstance(value, list)
            else value
        )
        for key, value in args.items()
    }


def first_documents(evidence: list[Evidence], limit: int | None = DOCUMENTS_PER_QUERY) -> tuple[Evidence, ...]:
    """A evidência dos primeiros `limit` documentos (sem limite, todos), na ordem em que aparecem."""
    if limit is None:
        return tuple(evidence)
    kept: list[str] = []
    for item in evidence:
        if item.doc_key not in kept:
            kept.append(item.doc_key)
    chosen = set(kept[:limit])
    return tuple(item for item in evidence if item.doc_key in chosen)
