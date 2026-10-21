"""`copy_index` (ADR-043, item 4): o Qdrant — ou qualquer `IndexWriter` — carregado a partir do artefato, sem PDF nem
modelo."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from axon_scouting.rag.index.adapters.artifact import ArtifactIndex
from axon_scouting.rag.index.adapters.bm25 import Bm25Encoder
from axon_scouting.rag.index.adapters.memory import HashingEmbedder, MemoryIndex
from axon_scouting.rag.index.api import IndexFilters
from axon_scouting.rag.ingestion.service import copy_index

from ..index.kit import INSTANTE, campos, codificar, registro, trecho, versao

DOC = campos()
TRECHOS = [trecho(DOC, "4.2", 2, "[Manual X · 4.2 Uniforme]\nO uniforme escoteiro é usado em cerimônias.")]
FILTROS = IndexFilters(effective_from=INSTANTE.date(), effective_until=INSTANTE.date())


def _artefato(root: Path) -> ArtifactIndex:
    async def publicar() -> ArtifactIndex:
        index = ArtifactIndex(root, HashingEmbedder(), Bm25Encoder())
        build = await index.create(versao())
        encoded = await codificar(TRECHOS, HashingEmbedder(), Bm25Encoder())
        await index.write(build.name, registro(DOC, TRECHOS, ("p", "p")), encoded)
        await index.publish(build.name)
        return index

    return asyncio.run(publicar())


def test_copia_o_publicado(tmp_path: Path) -> None:
    fonte = _artefato(tmp_path / "index")
    destino = MemoryIndex("scouting", HashingEmbedder(), Bm25Encoder())
    build = asyncio.run(copy_index(fonte, destino))
    assert build.version == asyncio.run(fonte.version())
    achados = asyncio.run(destino.search("uniforme escoteiro", filters=FILTROS, limit=3))
    assert [c.chunk_id for c in achados] == [TRECHOS[0].chunk_id]
    assert asyncio.run(destino.original(TRECHOS[0].chunk_id)) is not None


def test_falha_descarta_o_build(tmp_path: Path) -> None:
    fonte = _artefato(tmp_path / "index")
    destino = MemoryIndex("scouting", HashingEmbedder(), Bm25Encoder())

    async def quebra(*args: object) -> None:
        raise RuntimeError("servidor caiu")

    destino.publish = quebra  # type: ignore[method-assign]
    with pytest.raises(RuntimeError):
        asyncio.run(copy_index(fonte, destino))
    assert destino.storage.builds == {} and destino.storage.aliases == {}
