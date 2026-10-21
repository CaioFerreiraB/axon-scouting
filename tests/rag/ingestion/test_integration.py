"""Ingestão ponta a ponta com o parser real e o Qdrant em modo local (spec 102 US1 T028; US2 T037)."""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from fixtures import pdfs
from qdrant_client import AsyncQdrantClient

from axon_scouting.rag.index.adapters.bm25 import Bm25Encoder
from axon_scouting.rag.index.adapters.memory import HashingEmbedder
from axon_scouting.rag.index.adapters.qdrant import QdrantIndex
from axon_scouting.rag.index.api import IndexFilters
from axon_scouting.rag.ingestion.adapters.catalog_yaml import load_catalog
from axon_scouting.rag.ingestion.adapters.memory import FixedClock
from axon_scouting.rag.ingestion.adapters.pdfium import PdfiumParser
from axon_scouting.rag.ingestion.service import IngestionService

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
        store=index,
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


def test_norma_ponta_a_ponta(tmp_path: Path) -> None:
    pdfs.normative_pdf(tmp_path / "norma.pdf")
    catalogo = {
        "documents": [
            {
                "file": "norma.pdf",
                "document_id": "norma-x",
                "version": "1",
                "type": "resolucao",
                "title": "Norma X",
                "jurisdiction": "nacional",
                "effective_date": date(2024, 1, 1),
            }
        ]
    }

    async def cenario() -> list[Any]:
        client = AsyncQdrantClient(location=":memory:")
        try:
            service, index = servico(client)
            relatorio = await service.ingest(catalogo, tmp_path)
            assert relatorio.ok, relatorio.problems
            assert relatorio.documents[0].normative_units == 5
            ids = list((await service.status()).documents[0].chunk_ids)
            return await index.get(ids)
        finally:
            await client.close()

    chunks = {c.chunk_id: c for c in asyncio.run(cenario())}
    assert list(chunks) == [
        "norma-x-1#preambulo@p1",
        "norma-x-1#art1@p1",
        "norma-x-1#art2@p1",
        "norma-x-1#art41@p1",
        "norma-x-1#art42@p1",
        "norma-x-1#art43@p2",
    ]
    art42 = chunks["norma-x-1#art42@p1"]
    assert art42.text.startswith("[Norma X · Capítulo II · Seção I]\nArt. 42.")
    assert (art42.page_start, art42.page_end) == (1, 2)
    assert art42.section == "Seção I – Do planejamento"
