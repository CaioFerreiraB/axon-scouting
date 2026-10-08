"""Suíte de conformidade do seam `ChunkIndex` e das portas `IndexWriter`/`DocumentStore` (spec 102
contracts/chunk-index.md; T019, T040, T045). Roda contra o `MemoryIndex`, o `QdrantIndex` em modo local e, com
`AI_SCOUTING_TEST_QDRANT_URL`, o `QdrantIndex` contra o servidor.
"""

from __future__ import annotations

import asyncio
import dataclasses
from datetime import date
from typing import Any

import pytest

from ai_scouting.rag.index.adapters.bm25 import Bm25Encoder
from ai_scouting.rag.index.adapters.memory import HashingEmbedder
from ai_scouting.rag.index.api import (
    DocumentType,
    IndexFilters,
    IndexMismatch,
    IndexUnavailable,
    IndexVersion,
    ValidityStatus,
)
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
    outro_embedder,
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
    embedder = (
        HashingEmbedder(dimensions=version.embedding.dimensions, fingerprint=version.embedding.fingerprint)
        if version
        else HashingEmbedder()
    )
    sparse = Bm25Encoder()
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


# --- vigência (US3, T040) ----------------------------------------------------------------------------------------

M2023 = campos(
    "manual-v",
    "2023",
    document_title="Manual V",
    effective_date=date(2023, 1, 1),
    expiry_date=date(2026, 2, 28),
    superseded_by="manual-v-2026",
    status=ValidityStatus.SUBSTITUIDO,
)
M2026 = campos("manual-v", "2026", document_title="Manual V", effective_date=date(2026, 3, 1))
RASCUNHO = campos(
    "manual-v", "2027", document_title="Manual V", effective_date=date(2026, 3, 1), status=ValidityStatus.RASCUNHO
)
GERAL = campos("por-v", "1", document_title="POR V", document_type=DocumentType.POR, effective_date=date(2020, 1, 1))
ESCOTEIRO = campos("manual-esc", "1", document_title="Manual Esc", branch="escoteiro", effective_date=date(2020, 1, 1))
PIONEIRO = campos("manual-pio", "1", document_title="Manual Pio", branch="pioneiro", effective_date=date(2020, 1, 1))
TEXTO = "As regras do acampamento da tropa valem para todos."
VIGENCIA = [
    (f, [trecho(f, "s1", 1, f"[{f.document_title}]\n{TEXTO}")])
    for f in (M2023, M2026, RASCUNHO, GERAL, ESCOTEIRO, PIONEIRO)
]


async def publicar_vigencia(index: Index) -> None:
    embedder, sparse = HashingEmbedder(), Bm25Encoder()
    build = await index.create(versao())
    for fields, chunks in VIGENCIA:
        await index.write(build.name, registro(fields, chunks, ("p1",)), await codificar(chunks, embedder, sparse))
    await index.publish(build.name)


def _docs(chunks: list[Any]) -> set[str]:
    return {c.doc_key for c in chunks}


@pytest.mark.parametrize(
    ("filtros", "esperado"),
    [
        (
            IndexFilters(effective_from=HOJE, effective_until=HOJE),
            {"manual-v-2026", "por-v-1", "manual-esc-1", "manual-pio-1"},
        ),
        (
            IndexFilters(effective_from=date(2023, 1, 1), effective_until=date(2023, 12, 31), include_superseded=True),
            {"manual-v-2023", "por-v-1", "manual-esc-1", "manual-pio-1"},
        ),
        (
            IndexFilters(effective_from=date(2026, 1, 1), effective_until=date(2026, 12, 31)),
            {"manual-v-2026", "por-v-1", "manual-esc-1", "manual-pio-1"},
        ),
        (IndexFilters(effective_from=HOJE, effective_until=HOJE, previous_version=True), {"manual-v-2023"}),
        (
            IndexFilters(effective_from=HOJE, effective_until=HOJE, branch="pioneiro"),
            {"manual-v-2026", "por-v-1", "manual-pio-1"},
        ),
        (IndexFilters(effective_from=HOJE, effective_until=HOJE, document_types=(DocumentType.POR,)), {"por-v-1"}),
        (
            IndexFilters(effective_from=date(2000, 1, 1), effective_until=date(2100, 1, 1), include_superseded=True),
            {"manual-v-2023", "manual-v-2026", "por-v-1", "manual-esc-1", "manual-pio-1"},
        ),
    ],
)
def test_vigencia(target: Target, filtros: IndexFilters, esperado: set[str]) -> None:
    async def scenario(make: Make) -> None:
        index = padrao(make)
        await publicar_vigencia(index)
        achados = await index.search("regras do acampamento", filters=filtros, limit=20)
        assert _docs(achados) == esperado
        assert "manual-v-2027" not in _docs(achados)  # rascunho nunca

    run(target, scenario)


def test_historico_informa_o_status_atual(target: Target) -> None:
    async def scenario(make: Make) -> None:
        index = padrao(make)
        await publicar_vigencia(index)
        filtros = IndexFilters(
            effective_from=date(2023, 1, 1), effective_until=date(2023, 12, 31), include_superseded=True
        )
        (antigo,) = [
            c
            for c in await index.search("regras do acampamento", filters=filtros, limit=20)
            if c.document_id == "manual-v"
        ]
        assert antigo.status is ValidityStatus.SUBSTITUIDO
        assert antigo.to_chunk().status == "substituido"
        assert antigo.superseded_by == "manual-v-2026"

    run(target, scenario)


# --- original e versão (US5, T045) -------------------------------------------------------------------------------


def test_original_das_paginas_do_chunk(target: Target) -> None:
    async def scenario(make: Make) -> None:
        index = padrao(make)
        await publicar(index)
        original = await index.original("norma-x-1#art42@p3")
        assert original is not None
        assert (original.doc_key, original.document_title, original.version) == ("norma-x-1", "Norma X", "1")
        assert (original.page_start, original.page_end) == (3, 3)
        assert original.pages == ("Art. 41 …\nArt. 42 …",)
        assert (original.source_file, original.source_sha256) == ("norma-x-1.pdf", "0" * 64)
        assert original.source_url == "https://exemplo.org/norma-x-1.pdf"
        assert await index.original("inexistente#x@p1") is None

    run(target, scenario)


def test_embedder_diferente_recusa_a_busca(target: Target) -> None:
    async def scenario(make: Make) -> None:
        await publicar(padrao(make))
        outro = make(outro_embedder("fp-novo"), Bm25Encoder())
        with pytest.raises(IndexMismatch) as erro:
            await outro.search("uniforme", filters=DEFAULT, limit=3)
        assert "hashing-v1" in str(erro.value) and "fp-novo" in str(erro.value)

    run(target, scenario)


class _OutroBm25(Bm25Encoder):
    def identity(self) -> str:
        return "bm25-pt-v2"


def test_codificador_esparso_diferente_recusa_a_busca(target: Target) -> None:
    async def scenario(make: Make) -> None:
        await publicar(padrao(make))
        with pytest.raises(IndexMismatch, match="bm25-pt-v2"):
            await make(HashingEmbedder(), _OutroBm25()).search("uniforme", filters=DEFAULT, limit=3)

    run(target, scenario)


def test_mesma_instancia_ve_a_versao_nova(target: Target) -> None:
    async def scenario(make: Make) -> None:
        antiga = padrao(make)
        await publicar(antiga)
        assert await antiga.search("uniforme", filters=DEFAULT, limit=3)
        novo_embedder = outro_embedder("v2", dimensions=32)
        nova = make(novo_embedder, Bm25Encoder())
        await publicar(nova, versao(novo_embedder, built_at=INSTANTE.replace(hour=16)))
        assert (await antiga.version()).embedding.fingerprint == "v2"
        with pytest.raises(IndexMismatch):
            await antiga.search("uniforme", filters=DEFAULT, limit=3)
        assert await nova.search("uniforme", filters=DEFAULT, limit=3)

    run(target, scenario)


# --- similaridade densa (ADR-025; spec 103 T003) ------------------------------------------------------------------


def test_similaridade_em_cada_resultado(target: Target) -> None:
    async def scenario(make: Make) -> None:
        embedder = HashingEmbedder()
        index = make(embedder, Bm25Encoder())
        await publicar(index)
        alvo = NORMA_TRECHOS[1]
        achados = await index.search(alvo.text, filters=DEFAULT, limit=10)
        assert achados and all(c.similarity is not None and -1.0 <= c.similarity <= 1.0 for c in achados)
        maior = max(achados, key=lambda c: c.similarity or -2.0)
        assert maior.chunk_id == alvo.chunk_id  # o trecho com o mesmo texto da pergunta é o mais próximo
        pergunta = await embedder.embed_query(alvo.text)
        for chunk in achados:  # o mesmo cosseno em todo adapter, para os mesmos vetores
            (vetor,) = await embedder.embed_passages([chunk.text])
            esperado = sum(a * b for a, b in zip(pergunta, vetor, strict=True))
            assert chunk.similarity == pytest.approx(esperado, abs=1e-5)
        assert all(c.similarity is None for c in await index.get([alvo.chunk_id]))

    run(target, scenario)
