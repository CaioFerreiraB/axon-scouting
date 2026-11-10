"""Identidade do G1 (spec 106 T012, SC-001): com o grafo v2 e o plano, os trechos que vêm à frente da busca
(`graph_rank`) são os do `scouting` 0.11.1, na mesma ordem, para toda pergunta dos quatro conjuntos.

A referência (`fixtures/g1_reference.json`) foi gravada com o código do 0.11.1 (derivação com `organs` + `_by_organ`)
antes da migração (research R18). Os trechos do grafo não dependem do embedder denso: vêm do `get` do índice e da
escolha por BM25, então a suíte usa o embedder de teste.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest
import yaml
from axon_ai_contracts import RETRIEVAL_FILTERS_KEY

from axon_scouting.context.derivation import derive_context, to_slice
from axon_scouting.context.lexicon import LEXICON
from axon_scouting.context.periods import reference_day
from axon_scouting.context.text import normalize
from axon_scouting.rag.graph.adapters.artifact import ArtifactGraphStore
from axon_scouting.rag.graph.service import DomainGraphService
from axon_scouting.rag.index.adapters.artifact import ArtifactIndex
from axon_scouting.rag.index.adapters.bm25 import Bm25Encoder
from axon_scouting.rag.index.adapters.memory import HashingEmbedder
from axon_scouting.rag.ingestion.domain.tree import TYPED
from axon_scouting.rag.retrieval.domain.filters import facets, index_filters, plan, suggested_bodies, suggested_types
from axon_scouting.rag.retrieval.retriever import ScoutingRetriever

ROOT = Path(__file__).resolve().parents[3]
INDEX = ROOT / "src" / "axon_scouting" / "data" / "index"
REFERENCE = json.loads((Path(__file__).parent / "fixtures" / "g1_reference.json").read_text(encoding="utf-8"))
SETS = ("acervo", "regression", "sintese", "programa-adultos")


async def _now() -> dict[str, dict[str, Any]]:
    index = ArtifactIndex(INDEX, HashingEmbedder(), Bm25Encoder())
    graph = DomainGraphService(ArtifactGraphStore(INDEX))
    kinds = await index.unit_kinds()
    lexicon = dataclasses.replace(
        LEXICON,
        unit_kinds=MappingProxyType(
            {k.name: tuple(normalize(p) for p in k.plurals if normalize(p)) for k in kinds if k.name in TYPED}
        ),
        entity_terms=MappingProxyType(dict(await graph.aliases())),
    )
    retriever = ScoutingRetriever(index, graph=graph)
    result: dict[str, dict[str, Any]] = {}
    for name in SETS:
        for item in yaml.safe_load((ROOT / "eval" / f"{name}.yaml").read_text(encoding="utf-8"))["items"]:
            day = reference_day(datetime.fromisoformat(str(item["reference_time"])))
            filters = to_slice(derive_context(item["question"], day, lexicon))[RETRIEVAL_FILTERS_KEY]
            ids: list[str] = []
            if not facets(filters):
                chunks = await retriever._by_plan(
                    item["question"],
                    plan(filters),
                    index_filters(filters),
                    (suggested_types(filters), suggested_bodies(filters)),
                )
                ids = [c.chunk_id for c in chunks]
            organs = [e.split(":", 1)[1] for e in filters.get("entities", [])]  # type: ignore[union-attr]
            result[f"{name}/{item['id']}"] = {"organs": organs, "chunk_ids": ids}
    return result


@pytest.fixture(scope="module")
def agora() -> dict[str, dict[str, Any]]:
    return asyncio.run(_now())


def test_as_mesmas_perguntas(agora: dict[str, dict[str, Any]]) -> None:
    assert set(agora) == set(REFERENCE)


@pytest.mark.parametrize("item", sorted(REFERENCE))
def test_os_mesmos_trechos_a_frente(agora: dict[str, dict[str, Any]], item: str) -> None:
    assert agora[item] == REFERENCE[item]
