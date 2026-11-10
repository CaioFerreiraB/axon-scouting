"""Conformidade da porta `GraphStore` (spec 106 T010; constituição XIII): memória e artefato."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from axon_scouting.rag.graph.adapters.artifact import GRAPH, ArtifactGraphStore, write_graph_v2
from axon_scouting.rag.graph.adapters.memory import MemoryGraphStore
from axon_scouting.rag.graph.api import GraphUnavailable
from axon_scouting.rag.graph.ports import GraphStore

from ..grafo_sintetico import registros

Make = Callable[[list[dict[str, Any]], str | None, Path], GraphStore]


def _memory(records: list[dict[str, Any]], chunking: str | None, _: Path) -> GraphStore:
    return MemoryGraphStore(records, chunking=chunking)


def _artifact(records: list[dict[str, Any]], chunking: str | None, root: Path) -> GraphStore:
    if chunking is not None:
        (root / "manifest.json").write_text(json.dumps({"chunking": chunking}), encoding="utf-8")
    if records:
        write_graph_v2(root / GRAPH, records)
    return ArtifactGraphStore(root)


@pytest.fixture(params=[_memory, _artifact], ids=["memoria", "artefato"])
def make(request: pytest.FixtureRequest) -> Make:
    return request.param  # type: ignore[no-any-return]


def _key(record: dict[str, Any]) -> str:
    return json.dumps(record, sort_keys=True)


def test_ida_e_volta(make: Make, tmp_path: Path) -> None:
    original = registros()
    lidos = list(make(original, "teste", tmp_path).records())
    assert sorted(map(_key, lidos)) == sorted(map(_key, original))


def test_sem_grafo_nenhum_registro(make: Make, tmp_path: Path) -> None:
    assert list(make([], None, tmp_path).records()) == []


def test_grafo_de_outro_build_e_recusado(make: Make, tmp_path: Path) -> None:
    store = make(registros(chunking="velho"), "novo", tmp_path)
    with pytest.raises(GraphUnavailable, match=r"scripts/grafo\.py build"):
        list(store.records())
