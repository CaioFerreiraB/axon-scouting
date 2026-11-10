"""Invariantes do `graph.jsonl` commitado (spec 106 FR-019, T031; data-model, "Validação no build")."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from axon_scouting.rag.graph.api import ORIGINS, PREFIXES, ROLES
from axon_scouting.rag.graph.domain.model import from_records

ROOT = Path(__file__).resolve().parents[3]
INDEX = ROOT / "src" / "axon_scouting" / "data" / "index"


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


@pytest.fixture(scope="module")
def grafo() -> list[dict[str, Any]]:
    return _jsonl(INDEX / "graph.jsonl")


@pytest.fixture(scope="module")
def chunks() -> set[str]:
    return {r["chunk_id"] for r in _jsonl(INDEX / "chunks.jsonl")}


@pytest.fixture(scope="module")
def unidades() -> set[str]:
    return {p["unit_id"] for r in _jsonl(INDEX / "chunks.jsonl") for p in r.get("path") or []}


def test_carrega_sem_ponta_orfa(grafo: list[dict[str, Any]]) -> None:
    from_records(grafo)  # levanta ValueError com aresta de ponta inexistente


def test_o_chunking_e_o_do_indice(grafo: list[dict[str, Any]]) -> None:
    manifest = json.loads((INDEX / "manifest.json").read_text(encoding="utf-8"))
    assert grafo[0]["kind"] == "meta" and grafo[0]["chunking"] == manifest["chunking"]


def test_toda_fonte_resolve(grafo: list[dict[str, Any]], chunks: set[str], unidades: set[str]) -> None:
    organization = yaml.safe_load((ROOT / "corpus" / "organizacao.yaml").read_text(encoding="utf-8"))
    curated = {
        f"organizacao.yaml:{section}/{entry['id']}"
        for section, entries in organization.items()
        if isinstance(entries, list)
        for entry in entries
        if isinstance(entry, dict) and "id" in entry
    }
    unit_ids = {r["unit_id"] for r in grafo if r["kind"] == "unit"}
    problems = []
    for record in grafo:
        if record["kind"] not in ("entity", "edge"):
            continue
        source = record["source"]
        if record["origin"] not in ORIGINS:
            problems.append(f"origem {record['origin']}")
        if not (source in chunks or source in unidades or source in unit_ids or source in curated):
            problems.append(f"{record.get('id') or record['type']}: fonte {source}")
    assert problems == []


def test_modelo_so_conferido(grafo: list[dict[str, Any]]) -> None:
    assert [r for r in grafo if r.get("origin") == "modelo_conferido" and not r.get("checked")] == []


def test_todo_tipo_tem_papel_e_o_papel_e_o_da_tabela(grafo: list[dict[str, Any]]) -> None:
    for record in (r for r in grafo if r["kind"] == "version"):
        assert ROLES[record["type"]] == record["role"]


def test_entidades_so_dos_prefixos_e_sem_pessoas(grafo: list[dict[str, Any]]) -> None:
    for record in (r for r in grafo if r["kind"] == "entity"):
        assert record["type"] in PREFIXES and record["id"].startswith(record["type"] + ":")
