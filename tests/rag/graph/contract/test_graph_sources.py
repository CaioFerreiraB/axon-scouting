"""Conformidade da porta `GraphSources` (spec 106 T010a; constituição XIII): memória e artefato."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from axon_scouting.rag.graph.adapters.memory import MemorySources
from axon_scouting.rag.graph.adapters.sources import ArtifactSources
from axon_scouting.rag.graph.ports import GraphSources

CHUNKS = [{"chunk_id": "e-1#a1@p1", "doc_key": "e-1", "document_type": "estatuto", "path": []}]
DOCUMENTS = [{"doc_key": "e-1", "fields": {"document_id": "e", "document_type": "estatuto", "status": "vigente"}}]
MANIFEST = {"chunking": "teste", "built_at": "2026-11-10T00:00:00+00:00"}
ORGANIZATION = {"version": 1, "tipos_de_orgao": [{"id": "x", "nome": "Conselho X"}]}


@pytest.fixture(params=["memoria", "artefato"])
def fontes(request: pytest.FixtureRequest, tmp_path: Path) -> tuple[GraphSources, str]:
    text = yaml.safe_dump(ORGANIZATION, allow_unicode=True)
    sha = hashlib.sha256(text.encode()).hexdigest()
    if request.param == "memoria":
        return MemorySources(
            chunks=CHUNKS, documents=DOCUMENTS, manifest=MANIFEST, organization=ORGANIZATION, organization_sha256=sha
        ), sha
    (tmp_path / "chunks.jsonl").write_text("".join(json.dumps(c) + "\n" for c in CHUNKS), encoding="utf-8")
    (tmp_path / "documents.jsonl").write_text("".join(json.dumps(d) + "\n" for d in DOCUMENTS), encoding="utf-8")
    (tmp_path / "manifest.json").write_text(json.dumps(MANIFEST), encoding="utf-8")
    (tmp_path / "organizacao.yaml").write_text(text, encoding="utf-8")
    return ArtifactSources(tmp_path, tmp_path / "organizacao.yaml"), sha


def test_le_o_que_a_construcao_precisa(fontes: tuple[GraphSources, str]) -> None:
    sources, sha = fontes
    lidos: dict[str, Any] = {
        "chunks": list(sources.chunks()),
        "documents": list(sources.documents()),
        "manifest": dict(sources.manifest()),
        "organization": dict(sources.organization()),
    }
    assert lidos == {"chunks": CHUNKS, "documents": DOCUMENTS, "manifest": MANIFEST, "organization": ORGANIZATION}
    assert sources.organization_sha256() == sha
