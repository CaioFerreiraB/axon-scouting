"""O retriever sobre o `QdrantIndex` em modo local e, com o modelo, sobre o e5 (spec 103 US1; T010)."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest
from qdrant_client import AsyncQdrantClient

from axon_scouting.rag.index.adapters.bm25 import Bm25Encoder
from axon_scouting.rag.index.adapters.memory import HashingEmbedder
from axon_scouting.rag.index.adapters.qdrant import QdrantIndex
from axon_scouting.rag.retrieval.retriever import ScoutingRetriever

from .corpus import filtros, indice, publicar

PERGUNTA = "o que diz a Regra 33"


def test_qdrant_local_igual_a_memoria() -> None:
    async def go() -> tuple[list[str], list[str]]:
        memoria = await ScoutingRetriever(await indice()).search(PERGUNTA, top_k=5, filters=filtros())
        client = AsyncQdrantClient(location=":memory:")
        try:
            embedder, sparse = HashingEmbedder(), Bm25Encoder()
            index = QdrantIndex(client, "scouting", embedder, sparse)
            await publicar(index, embedder, sparse)
            qdrant = await ScoutingRetriever(index).search(PERGUNTA, top_k=5, filters=filtros())
        finally:
            await client.close()
        return [c.chunk_id for c in memoria], [c.chunk_id for c in qdrant]

    memoria, qdrant = asyncio.run(go())
    assert "por-2025#regra33@p38" in memoria and "por-2025#regra33@p38" in qdrant
    assert set(memoria) == set(qdrant)


@pytest.mark.skipif(not os.environ.get("AXON_SCOUTING_EMBEDDING_DIR"), reason="AXON_SCOUTING_EMBEDDING_DIR ausente")
def test_parafrase_com_e5() -> None:
    from axon_scouting.rag.index.adapters.e5 import E5Embedder

    async def go() -> list[tuple[str, float]]:
        embedder = E5Embedder(Path(os.environ["AXON_SCOUTING_EMBEDDING_DIR"]))
        retriever = ScoutingRetriever(await indice(embedder))
        achados = await retriever.search("o que é preciso para acampar com a tropa?", top_k=5, filters=filtros())
        return [(c.chunk_id, c.score) for c in achados]

    achados = asyncio.run(go())
    ids = [chunk_id for chunk_id, _ in achados[:3]]
    assert "manual-x-2026#4.2@p12" in ids  # US1 cen. 2: a paráfrase acha o trecho de acampamento
    scores = dict(achados)
    if "por-2025#regra33@p38" in scores:
        assert scores["manual-x-2026#4.2@p12"] > scores["por-2025#regra33@p38"]
