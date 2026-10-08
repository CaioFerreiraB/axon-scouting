"""Sanidade do retrieval com o e5 e o corpus real (spec 103 quickstart §2; T020). Não é teste da CI.

Uso: `AI_SCOUTING_QDRANT_URL=… AI_SCOUTING_EMBEDDING_DIR=… [AI_SCOUTING_INDEX=scouting] uv run python
tests/rag/retrieval/sanity.py` — com o índice ingerido a partir de `corpus/corpus.yaml`.
"""

from __future__ import annotations

import asyncio
import os
import random
import time
from datetime import date
from pathlib import Path

from qdrant_client import AsyncQdrantClient

from ai_scouting.rag.index.adapters.bm25 import Bm25Encoder
from ai_scouting.rag.index.adapters.e5 import E5Embedder
from ai_scouting.rag.index.adapters.qdrant import QdrantIndex
from ai_scouting.rag.retrieval.domain.ordering import FUSION_SCORE
from ai_scouting.rag.retrieval.reranker import MetadataReranker
from ai_scouting.rag.retrieval.retriever import ScoutingRetriever

HOJE = date.today().isoformat()  # o roteiro faz o papel da 101: fixa o dia de referência
PERGUNTAS: list[tuple[str, dict[str, object]]] = [
    ("o que diz a Regra 033", {}),
    ("qual é a promessa do lobinho?", {}),
    ("como conduzir a cerimônia de promessa na alcateia", {"branch": "lobinho"}),
    ("quem compõe a Assembleia Nacional?", {"document_types": ["estatuto"]}),
    ("receita de bolo de cenoura", {}),
]


async def main() -> None:
    client = AsyncQdrantClient(url=os.environ.get("AI_SCOUTING_QDRANT_URL", "http://localhost:6333"))
    index = QdrantIndex(
        client,
        os.environ.get("AI_SCOUTING_INDEX", "scouting"),
        E5Embedder(Path(os.environ["AI_SCOUTING_EMBEDDING_DIR"]).expanduser()),
        Bm25Encoder(),
    )
    retriever = ScoutingRetriever(index)
    try:
        for pergunta, extra in PERGUNTAS:
            filtros = {"effective_from": HOJE, "effective_until": HOJE, **extra}
            achados = await retriever.search(pergunta, top_k=5, filters=filtros)
            print(f"\n? {pergunta} {extra or ''}")
            for chunk in achados[:3]:
                print(f"   score {chunk.score:.3f}  fusão {chunk.metadata[FUSION_SCORE]:.3f}  {chunk.chunk_id}")
        candidatos = await retriever.search(
            "acampamento da patrulha", top_k=30, filters={"effective_from": HOJE, "effective_until": HOJE}
        )
        random.Random(0).shuffle(candidatos)
        reranker, voltas = MetadataReranker(), 1000
        inicio = time.perf_counter()
        for _ in range(voltas):
            await reranker.rerank("acampamento da patrulha", candidatos, top_k=5)
        media = (time.perf_counter() - inicio) / voltas * 1000
        print(f"\nrerank de {len(candidatos)} candidatos: {media:.3f} ms em média ({voltas} voltas)")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
