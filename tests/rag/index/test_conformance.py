"""Suíte de conformidade do seam `ChunkIndex` e das portas `IndexWriter`/`DocumentStore` (spec 102
contracts/chunk-index.md; T019, T040, T045). Roda contra o `MemoryIndex`, o `QdrantIndex` em modo local e, com
`AI_SCOUTING_TEST_QDRANT_URL`, o `QdrantIndex` contra o servidor.
"""

from __future__ import annotations

import asyncio
import dataclasses
from datetime import date

import pytest

from ai_scouting.rag.index.adapters.bm25 import Bm25Encoder
from ai_scouting.rag.index.adapters.memory import HashingEmbedder
from ai_scouting.rag.index.api import IndexFilters, IndexUnavailable, IndexVersion, ValidityStatus
from ai_scouting.rag.index.ports import StoredDocument

from .kit import (
    INSTANTE,
    Index,
    Make,
    MemoryTarget,
    Scenario,
    Target,
    campos,
    codificar,
    qdrant_targets,
    registro,
    trecho,
    versao,
)

HOJE = date(2026, 10, 7)
DEFAULT = IndexFilters(effective_from=HOJE, effective_until=HOJE)

NORMA = campos("norma-x", "1", document_title="Norma X", effective_date=date(2020, 1, 1))
NORMA_TRECHOS = [
    trecho(
        NORMA,
        "art41",
        3,
        "[Norma X · Capítulo II]\nArt. 41. O grupo mantém registro das atividades.",
        article="Art. 41",
    ),
    trecho(
        NORMA,
        "art42",
        3,
        "[Norma X · Capítulo II]\nArt. 42. O responsável pela atividade deverá comunicar os pais.",
        article="Art. 42",
    ),
    trecho(
        NORMA, "art43", 4, "[Norma X · Capítulo II]\nArt. 43. A autorização dos pais é obrigatória.", article="Art. 43"
    ),
]
MANUAL = campos()
MANUAL_TRECHOS = [
    trecho(MANUAL, "4.1", 10, "[Manual X · 4.1 Acampamento]\nO acampamento de patrulha exige planejamento e cardápio."),
    trecho(MANUAL, "4.2", 11, "[Manual X · 4.2 Uniforme]\nO uniforme escoteiro é usado em cerimônias e atividades."),
]
PAGINAS = ("p1", "p2", "Art. 41 …\nArt. 42 …", "Art. 43 …")


def _targets() -> list[Target]:
    return [MemoryTarget(), *qdrant_targets()]


@pytest.fixture(params=_targets(), ids=lambda t: t.name)
def target(request: pytest.FixtureRequest) -> Target:
    return request.param  # type: ignore[no-any-return]


def run(target: Target, scenario: Scenario) -> None:
    asyncio.run(target.run(scenario))


async def publicar(index: Index, version: IndexVersion | None = None) -> str:
    embedder, sparse = HashingEmbedder(), Bm25Encoder()
    build = await index.create(version or versao())
    for fields, chunks in ((NORMA, NORMA_TRECHOS), (MANUAL, MANUAL_TRECHOS)):
        await index.write(build.name, registro(fields, chunks, PAGINAS), await codificar(chunks, embedder, sparse))
    await index.publish(build.name)
    return build.name


def padrao(make: Make) -> Index:
    return make(HashingEmbedder(), Bm25Encoder())


# --- foundational (T019) ---------------------------------------------------------------------------------------


def test_indice_novo(target: Target) -> None:
    async def scenario(make: Make) -> None:
        index = padrao(make)
        assert await index.published() is None
        with pytest.raises(IndexUnavailable):
            await index.search("uniforme", filters=DEFAULT, limit=5)
        with pytest.raises(IndexUnavailable):
            await index.version()
        with pytest.raises(IndexUnavailable):
            await index.get(["x"])

    run(target, scenario)


def test_publicar_torna_tudo_visivel_de_uma_vez(target: Target) -> None:
    async def scenario(make: Make) -> None:
        index = padrao(make)
        build = await index.create(versao())
        await index.write(
            build.name,
            registro(MANUAL, MANUAL_TRECHOS, PAGINAS),
            await codificar(MANUAL_TRECHOS, HashingEmbedder(), Bm25Encoder()),
        )
        with pytest.raises(IndexUnavailable):
            await index.search("uniforme", filters=DEFAULT, limit=5)
        await index.publish(build.name)
        publicado = await index.published()
        assert publicado is not None and publicado.name == build.name
        assert [c.chunk_id for c in await index.get([c.chunk_id for c in MANUAL_TRECHOS])] == [
            c.chunk_id for c in MANUAL_TRECHOS
        ]
        assert await index.search("uniforme", filters=DEFAULT, limit=5)

    run(target, scenario)


def test_descartar_nao_afeta_o_publicado(target: Target) -> None:
    async def scenario(make: Make) -> None:
        index = padrao(make)
        publicado = await publicar(index)
        antes = await index.search("uniforme escoteiro", filters=DEFAULT, limit=5)
        novo = await index.create(versao(built_at=INSTANTE.replace(hour=16)))
        await index.write(
            novo.name,
            registro(MANUAL, MANUAL_TRECHOS[:1], PAGINAS),
            await codificar(MANUAL_TRECHOS[:1], HashingEmbedder(), Bm25Encoder()),
        )
        await index.discard(novo.name)
        assert (await index.published()).name == publicado  # type: ignore[union-attr]
        assert await index.search("uniforme escoteiro", filters=DEFAULT, limit=5) == antes

    run(target, scenario)


def test_documentos_sem_paginas(target: Target) -> None:
    async def scenario(make: Make) -> None:
        index = padrao(make)
        build = await publicar(index)
        documentos = await index.documents(build)
        assert [d.doc_key for d in documentos] == ["manual-x-2026", "norma-x-1"]
        assert all(type(d) is StoredDocument for d in documentos)
        assert documentos[1].chunk_ids == tuple(c.chunk_id for c in NORMA_TRECHOS)
        assert documentos[1].fields == NORMA

    run(target, scenario)


def test_copiar_preserva_vetores_e_reescreve_vigencia(target: Target) -> None:
    async def scenario(make: Make) -> None:
        index = padrao(make)
        origem = await publicar(index)
        consulta = "o responsável pela atividade deverá comunicar os pais"
        antes = await index.search(consulta, filters=DEFAULT, limit=3)
        destino = await index.create(versao(built_at=INSTANTE.replace(hour=16)))
        revogada = dataclasses.replace(NORMA, status=ValidityStatus.REVOGADO, expiry_date=date(2030, 12, 31))
        await index.copy(origem, destino.name, registro(revogada, NORMA_TRECHOS, PAGINAS))
        await index.copy(origem, destino.name, registro(MANUAL, MANUAL_TRECHOS, PAGINAS))
        await index.publish(destino.name)
        depois = await index.search(consulta, filters=DEFAULT, limit=3)
        # Topo nos dois rankings (denso e esparso) dá 1/2 + 1/2: sem o vetor denso copiado, cairia para 1/2. Empates
        # abaixo do topo podem trocar de ordem no Qdrant, então só o topo é comparado por score.
        assert depois[0].chunk_id == antes[0].chunk_id == "norma-x-1#art42@p3"
        assert depois[0].score == antes[0].score == 1.0
        copiados = await index.get([c.chunk_id for c in NORMA_TRECHOS])
        assert {c.status for c in copiados} == {ValidityStatus.REVOGADO}
        assert {c.expiry_date for c in copiados} == {date(2030, 12, 31)}
        assert [c.article for c in copiados] == ["Art. 41", "Art. 42", "Art. 43"]
        assert [d.doc_key for d in await index.documents(destino.name)] == ["manual-x-2026", "norma-x-1"]
        with pytest.raises(KeyError):
            await index.documents(origem)  # publicar apaga o build anterior

    run(target, scenario)


def test_get_na_ordem_pedida(target: Target) -> None:
    async def scenario(make: Make) -> None:
        index = padrao(make)
        await publicar(index)
        pedidos = [NORMA_TRECHOS[2].chunk_id, "inexistente#x@p1", MANUAL_TRECHOS[0].chunk_id]
        obtidos = await index.get(pedidos)
        assert [c.chunk_id for c in obtidos] == [pedidos[0], pedidos[2]]
        assert all(c.score is None for c in obtidos)
        assert obtidos[0] == NORMA_TRECHOS[2]

    run(target, scenario)


def test_busca_ordenada_limitada_e_deterministica(target: Target) -> None:
    async def scenario(make: Make) -> None:
        index = padrao(make)
        await publicar(index)
        resultado = await index.search("atividade dos pais", filters=DEFAULT, limit=3)
        assert len(resultado) == 3
        chaves = [(-(c.score or 0.0), c.chunk_id) for c in resultado]
        assert chaves == sorted(chaves)
        assert all(c.score is not None and c.score > 0 for c in resultado)
        assert await index.search("atividade dos pais", filters=DEFAULT, limit=3) == resultado
        assert await index.search("de que o a", filters=DEFAULT, limit=3) == []
        assert await index.search("", filters=DEFAULT, limit=3) == []

    run(target, scenario)


def test_termo_exato_pelo_esparso(target: Target) -> None:
    async def scenario(make: Make) -> None:
        index = padrao(make)
        await publicar(index)
        resultado = await index.search("o que diz o 42", filters=DEFAULT, limit=2)
        assert resultado[0].chunk_id == "norma-x-1#art42@p3"

    run(target, scenario)


def test_versao_gravada(target: Target) -> None:
    async def scenario(make: Make) -> None:
        index = padrao(make)
        await publicar(index)
        assert await index.version() == versao()

    run(target, scenario)


def test_mesmo_instante_ganha_sufixo(target: Target) -> None:
    async def scenario(make: Make) -> None:
        index = padrao(make)
        primeiro = await index.create(versao())
        segundo = await index.create(versao())
        assert primeiro.name == "scouting__20261007T153000Z" or primeiro.name.endswith("__20261007T153000Z")
        assert segundo.name == f"{primeiro.name}-2"

    run(target, scenario)
