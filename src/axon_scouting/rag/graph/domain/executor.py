"""O executor do plano (spec 106 FR-012; research R9; design.md, seção 5).

Roda as consultas do plano e devolve as frentes: a evidência de cada passo, ordenada, cortada em até 3 documentos, sem
as relações de governança (FR-007). Não escolhe trechos nem conta vagas — isso é da 103, que lida com a pergunta.
Puro. No G1, só a forma `simples`; `para_cada` e `intersecao` chegam no G2.
"""

from __future__ import annotations

from typing import Final

from axon_scouting.rag.graph.api import Evidence, Frentes, FrontGroup, QueryPlan
from axon_scouting.rag.graph.domain.model import GOVERNANCE, DomainGraph
from axon_scouting.rag.graph.domain.queries import run_query

DOCUMENTS_PER_QUERY: Final = 3
"""Até 3 documentos por consulta, da norma de cima para a de baixo (ADR-063; clarify da 106)."""


def executar(g: DomainGraph, plano: QueryPlan) -> Frentes:
    if plano.forma != "simples":
        raise NotImplementedError("para_cada e intersecao chegam no G2 (spec 106)")
    groups: list[FrontGroup] = []
    for index, step in enumerate(plano.consultas):
        evidence = [e for e in run_query(g, step.q, step.args) if e.edge.type not in GOVERNANCE]
        groups.append(FrontGroup(step=index, query=step.q, entity=None, evidence=first_documents(evidence)))
    return tuple(groups)


def first_documents(evidence: list[Evidence], limit: int = DOCUMENTS_PER_QUERY) -> tuple[Evidence, ...]:
    """A evidência dos primeiros `limit` documentos, na ordem em que aparecem, sem mudar a ordem dentro deles."""
    kept: list[str] = []
    for item in evidence:
        if item.doc_key not in kept:
            kept.append(item.doc_key)
    chosen = set(kept[:limit])
    return tuple(item for item in evidence if item.doc_key in chosen)
