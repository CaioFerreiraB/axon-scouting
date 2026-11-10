"""SC-008 (spec 106 T028a): com o grafo carregado, o plano de cada pergunta do acervo roda em menos de 50 ms."""

from __future__ import annotations

import asyncio
import statistics
import time
from pathlib import Path

import yaml

from axon_scouting.context.text import normalize
from axon_scouting.rag.graph.adapters.artifact import ArtifactGraphStore
from axon_scouting.rag.graph.api import QueryPlan, QueryStep, reconhecer
from axon_scouting.rag.graph.service import DomainGraphService

ROOT = Path(__file__).resolve().parents[3]
LIMIT_MS = 50.0


def test_cada_plano_em_menos_de_50_ms() -> None:
    graph = DomainGraphService(ArtifactGraphStore(ROOT / "src" / "axon_scouting" / "data" / "index"))
    aliases = asyncio.run(graph.aliases())  # carga
    items = yaml.safe_load((ROOT / "eval" / "acervo.yaml").read_text(encoding="utf-8"))["items"]
    times: list[float] = []
    for item in items:
        start = time.perf_counter()
        entities = reconhecer(normalize(item["question"]), aliases)
        if entities:
            asyncio.run(graph.run(QueryPlan(consultas=(QueryStep(q="sobre", args={"entidades": list(entities)}),))))
        times.append((time.perf_counter() - start) * 1000)
    print(f"mediana {statistics.median(times):.2f} ms, máximo {max(times):.2f} ms")
    assert max(times) < LIMIT_MS
