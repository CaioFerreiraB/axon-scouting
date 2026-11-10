"""`DomainGraphService`: implementa o seam *Grafo de domínio* sobre a porta `GraphStore` (spec 106, research R13).

Carrega o grafo uma vez por processo, com trava, e delega às consultas e ao executor puros do `domain/`.
"""

from __future__ import annotations

import threading

from axon_ai_contracts import JSONValue

from axon_scouting.rag.graph.api import Aliases, Evidence, Frentes, QueryName, QueryPlan
from axon_scouting.rag.graph.domain.executor import executar
from axon_scouting.rag.graph.domain.model import DomainGraph, from_records
from axon_scouting.rag.graph.domain.queries import run_query
from axon_scouting.rag.graph.ports import GraphStore


class DomainGraphService:
    def __init__(self, store: GraphStore) -> None:
        self._store = store
        self._graph: DomainGraph | None = None
        self._lock = threading.Lock()

    def graph(self) -> DomainGraph:
        with self._lock:
            if self._graph is None:
                self._graph = from_records(self._store.records())
            return self._graph

    async def aliases(self) -> Aliases:
        return self.graph().aliases()

    async def run(self, plan: QueryPlan) -> Frentes:
        graph = self.graph()
        if plan.empty or not graph.edges:  # sem grafo, nada à frente (contrato, garantia 6)
            return ()
        return executar(graph, plan)

    async def query(self, name: QueryName, /, **args: JSONValue) -> list[Evidence]:
        return run_query(self.graph(), name, args)
