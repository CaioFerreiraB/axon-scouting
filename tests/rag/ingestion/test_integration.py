"""Ingestão ponta a ponta com o parser real e o Qdrant em modo local (spec 102 US1 T028; US2 T037)."""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from fixtures import pdfs
from qdrant_client import AsyncQdrantClient

from ai_scouting.rag.index.adapters.bm25 import Bm25Encoder
from ai_scouting.rag.index.adapters.memory import HashingEmbedder
from ai_scouting.rag.index.adapters.qdrant import QdrantIndex
from ai_scouting.rag.index.api import IndexFilters
from ai_scouting.rag.ingestion.adapters.catalog_yaml import load_catalog
from ai_scouting.rag.ingestion.adapters.memory import FixedClock
from ai_scouting.rag.ingestion.adapters.pdfium import PdfiumParser
from ai_scouting.rag.ingestion.service import IngestionService

HOJE = date(2026, 10, 7)
CATALOGO = """
documents:
  - file: manuais/manual-x.pdf
    document_id: manual-x
    version: "2026"
    type: manual
    title: Manual X
    branch: pioneiro
    jurisdiction: nacional
    effective_date: 2026-03-01
"""


def servico(client: AsyncQdrantClient) -> tuple[IngestionService, QdrantIndex]:
    embedder, sparse = HashingEmbedder(), Bm25Encoder()
    index = QdrantIndex(client, "scouting", embedder, sparse)
    service = IngestionService(
        parser=PdfiumParser(),
        embedder=embedder,
        sparse=sparse,
        writer=index,
        clock=FixedClock(datetime(2026, 10, 7, 15, 30, tzinfo=UTC)),
    )
    return service, index


def test_manual_ponta_a_ponta(tmp_path: Path) -> None:
    pdfs.narrative_pdf(tmp_path / "manuais" / "manual-x.pdf")
    (tmp_path / "corpus.yaml").write_text(CATALOGO, encoding="utf-8")
    catalogo, base = load_catalog(tmp_path / "corpus.yaml")

    async def cenario() -> dict[str, Any]:
        client = AsyncQdrantClient(location=":memory:")
        try:
            service, index = servico(client)
            primeiro = await service.ingest(catalogo, base)
            assert primeiro.ok, primeiro.problems
            assert (await index.published()) is not None
            filtros = IndexFilters(effective_from=HOJE, effective_until=HOJE, branch="pioneiro")
            achados = await index.search("acampamento de patrulha cardápio", filters=filtros, limit=3)
            assert achados[0].chunk_id == "manual-x-2026#4.2@p2"
            assert "cardápio" in achados[0].text
            status = await service.status()
            ids = list(status.documents[0].chunk_ids)
            assert [c.chunk_id for c in await index.get(ids)] == ids
            segundo = await service.ingest(catalogo, base)
            assert segundo.ok and [d.reused for d in segundo.documents] == [True]
            status = await service.status()
            return {"antes": ids, "depois": list(status.documents[0].chunk_ids)}
        finally:
            await client.close()

    resultado = asyncio.run(cenario())
    assert resultado["antes"] == resultado["depois"]
    assert resultado["antes"] == ["manual-x-2026#s1@p1", "manual-x-2026#4@p1", "manual-x-2026#4.2@p2"]
