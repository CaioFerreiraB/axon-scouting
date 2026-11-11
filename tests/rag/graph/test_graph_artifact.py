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
    curated: set[str] = set()
    for name in ("organizacao.yaml", "dominio.yaml"):
        data = yaml.safe_load((ROOT / "corpus" / name).read_text(encoding="utf-8"))
        curated |= {
            f"{name}:{section}/{entry['id']}"
            for section, entries in data.items()
            if isinstance(entries, list)
            for entry in entries
            if isinstance(entry, dict) and "id" in entry
        }
    taxonomy = yaml.safe_load((ROOT / "corpus" / "taxonomia.yaml").read_text(encoding="utf-8"))
    curated |= {f"taxonomia:{entry['categoria']}" for entry in taxonomy["entries"]}
    # G3: APLICA_SE_A e EMITIDO_POR saem da pasta do arquivo de um documento do catálogo.
    curated |= {
        "pasta:" + d["source_file"].removeprefix("ocr/").rsplit("/", 1)[0] for d in _jsonl(INDEX / "documents.jsonl")
    }
    unit_ids = {r["unit_id"] for r in grafo if r["kind"] == "unit"}
    # G4: GERA e DE saem do item da ficha, `<doc_key>#<n>@p<página>`, de uma ata do catálogo.
    versions = {r["doc_key"] for r in grafo if r["kind"] == "version"}
    curated |= {
        f"{e['doc_key']}#{e['id'].rsplit('#', 1)[1]}@p{e['page']}"
        for e in grafo
        if e["kind"] == "event" and e["doc_key"] in versions
    }
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


def test_toda_versao_se_aplica_a_uma_instituicao(grafo: list[dict[str, Any]]) -> None:
    """G3 (106 US3): todo documento do acervo vale para uma instituição do catálogo; a UEL ou a região fora dele
    apareceria aqui."""
    versions = {r["doc_key"] for r in grafo if r["kind"] == "version"}
    applied = {r["from"] for r in grafo if r["kind"] == "edge" and r["type"] == "APLICA_SE_A"}
    assert versions - applied == set()


def test_instancias_tem_tipo_e_a_de_cima(grafo: list[dict[str, Any]]) -> None:
    instances = {r["id"] for r in grafo if r["kind"] == "entity" and r["type"] in ("inst", "orgao")}
    typed = {r["from"] for r in grafo if r["kind"] == "edge" and r["type"] == "INSTANCIA_DE"}
    above = {r["from"] for r in grafo if r["kind"] == "edge" and r["type"] == "PARTE_DE"}
    assert instances and instances == typed
    assert instances - above == {"inst:omme", "inst:regiao_interamericana"}


def test_todo_evento_vem_de_uma_ata_e_tem_os_trechos_da_pagina(grafo: list[dict[str, Any]], chunks: set[str]) -> None:
    """G4 (106 US4): o evento tem a ata (`GERA`), a página e os trechos dela no índice; e trata de alguma coisa."""
    events = {r["id"]: r for r in grafo if r["kind"] == "event"}
    generated = {r["to"] for r in grafo if r["kind"] == "edge" and r["type"] == "GERA"}
    assert events and set(events) == generated
    assert all(set(e["chunk_ids"]) <= chunks for e in events.values())
    assert sum(1 for e in events.values() if e["chunk_ids"]) / len(events) > 0.95
    about = {r["from"] for r in grafo if r["kind"] == "edge" and r["type"] == "DE"}
    assert len(about) / len(events) > 0.5
