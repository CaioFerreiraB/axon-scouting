"""O artefato do índice espelha o `corpus.yaml`, sem os PDFs (spec 105 research R14; substitui, no CI, a reingestão da
102 research R25).

Os originais ficam fora do git desde 2026-10-26: o CI não pode reingerir. O que dá para conferir sem eles é o que o
catálogo decide — o conjunto de documentos, os metadados, a vigência derivada da linhagem e a proveniência. Uma
mudança no catálogo sem reingestão falha aqui; a reingestão com os PDFs continua sendo `axon-scouting-corpus ingest`,
na máquina de quem os tem.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from axon_scouting.rag.ingestion.adapters.catalog_yaml import load_catalog
from axon_scouting.rag.ingestion.domain.catalog import CatalogEntry, validate_catalog
from axon_scouting.rag.ingestion.domain.lineage import Validity, derive_lineages

ROOT = Path(__file__).resolve().parents[3]
CATALOG = ROOT / "corpus" / "corpus.yaml"
DOCUMENTS = ROOT / "src" / "axon_scouting" / "data" / "index" / "documents.jsonl"


def _iso(value: date | None) -> str | None:
    return None if value is None else value.isoformat()


def _expected(entry: CatalogEntry, validity: Validity) -> dict[str, Any]:
    """O registro que a ingestão gravaria para a entrada, na forma do `documents.jsonl` (sem o texto)."""
    return {
        "fields": {
            "document_id": entry.document_id,
            "version": entry.version,
            "document_type": entry.document_type.value,
            "document_title": entry.title,
            "jurisdiction": entry.jurisdiction,
            "effective_date": _iso(validity.effective_date),
            "status": validity.status.value,
            "branch": entry.branch,
            "publication_date": _iso(entry.publication_date),
            "expiry_date": _iso(validity.expiry_date),
            "superseded_by": validity.superseded_by,
            "successor_expiry_date": _iso(validity.successor_expiry_date),
        },
        "source_file": entry.file,
        "source_url": entry.source_url,
    }


@pytest.fixture(scope="module")
def esperado() -> dict[str, dict[str, Any]]:
    raw, base_dir = load_catalog(CATALOG)
    validated = validate_catalog(raw, base_dir)
    assert validated.ok, [problem.message for problem in validated.problems]
    lineages = derive_lineages(validated.entries)
    assert lineages.ok, [problem.message for problem in lineages.problems]
    return {entry.doc_key: _expected(entry, lineages.validities[entry.doc_key]) for entry in validated.entries}


@pytest.fixture(scope="module")
def gravado() -> dict[str, dict[str, Any]]:
    records = (json.loads(line) for line in DOCUMENTS.read_text(encoding="utf-8").splitlines() if line)
    return {
        record["doc_key"]: {key: record.get(key) for key in ("fields", "source_file", "source_url")}
        for record in records
    }


def test_os_mesmos_documentos(esperado: dict[str, Any], gravado: dict[str, Any]) -> None:
    assert sorted(set(esperado) - set(gravado)) == [], "no catálogo, fora do artefato: rode a ingestão"
    assert sorted(set(gravado) - set(esperado)) == [], "no artefato, fora do catálogo: rode a ingestão"


def test_metadados_vigencia_e_proveniencia(esperado: dict[str, Any], gravado: dict[str, Any]) -> None:
    divergentes = [key for key in sorted(esperado.keys() & gravado.keys()) if esperado[key] != gravado[key]]
    assert divergentes == [], f"{len(divergentes)} documento(s) com o catálogo mudado depois da ingestão"
