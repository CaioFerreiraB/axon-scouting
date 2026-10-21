"""Aceitação do serviço de ingestão com adapters em memória (spec 102: US1 T025; US2 T037; US3 T041; US5 T046)."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from axon_scouting.rag.index.adapters.bm25 import Bm25Encoder
from axon_scouting.rag.index.adapters.memory import HashingEmbedder, MemoryIndex, MemoryStorage
from axon_scouting.rag.index.api import DocumentType, IndexFilters, ValidityStatus
from axon_scouting.rag.ingestion.adapters.memory import FixedClock, MemoryParser
from axon_scouting.rag.ingestion.domain.chunking import ChunkingPolicy
from axon_scouting.rag.ingestion.service import IngestionService

BASE = Path("/corpus")
INSTANTE = datetime(2026, 10, 7, 15, 30, tzinfo=UTC)

MANUAL_PAGINAS = [
    "APRESENTAÇÃO\nEste manual orienta a vida em patrulha.\n4 Programa de Atividades\nO programa tem ciclos.",
    "4.2 Acampamento\nO acampamento de patrulha exige cardápio e autorização dos responsáveis.",
]


class CountingEmbedder(HashingEmbedder):
    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        self.passages = 0

    async def embed_passages(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        self.passages += len(texts)
        return await super().embed_passages(texts)


def entrada(**changes: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "file": "manual.pdf",
        "document_id": "manual-x",
        "version": "2026",
        "type": "manual",
        "title": "Manual X",
        "branch": "pioneiro",
        "jurisdiction": "nacional",
        "effective_date": date(2026, 3, 1),
    }
    base.update(changes)
    return {k: v for k, v in base.items() if v is not ...}


def catalogo(*entradas: dict[str, Any]) -> dict[str, Any]:
    return {"documents": list(entradas)}


class Ambiente:
    def __init__(self, **embedder: Any) -> None:
        self.parser = MemoryParser({BASE / "manual.pdf": MANUAL_PAGINAS})
        self.storage = MemoryStorage()
        self.embedder = CountingEmbedder(**embedder)
        self.index = MemoryIndex("scouting", self.embedder, Bm25Encoder(), storage=self.storage)
        self.service = self.servico(self.embedder)

    def servico(self, embedder: HashingEmbedder) -> IngestionService:
        index = MemoryIndex("scouting", embedder, Bm25Encoder(), storage=self.storage)
        return IngestionService(
            parser=self.parser,
            embedder=embedder,
            sparse=Bm25Encoder(),
            writer=index,
            store=index,
            clock=FixedClock(INSTANTE),
        )

    def ingest(self, raw: object, service: IngestionService | None = None) -> Any:
        return asyncio.run((service or self.service).ingest(raw, BASE))

    def publicado(self) -> str | None:
        return self.storage.aliases.get("scouting")


# --- US1 -------------------------------------------------------------------------------------------------------


def test_us1_cen1_chunks_com_id_metadados_e_duas_representacoes() -> None:
    amb = Ambiente()
    relatorio = amb.ingest(catalogo(entrada()))
    assert relatorio.ok, relatorio.problems
    (doc,) = relatorio.documents
    assert (doc.doc_key, doc.reused, doc.status) == ("manual-x-2026", False, ValidityStatus.VIGENTE)
    assert doc.chunks == 3
    assert relatorio.build == amb.publicado()
    build = amb.storage.builds[relatorio.build]
    assert len(build.chunks) == 3
    for chunk_id, encoded in build.chunks.items():
        chunk = encoded.chunk
        assert chunk_id.startswith("manual-x-2026#")
        assert chunk.text.startswith("[Manual X")
        assert (chunk.document_type, chunk.document_title, chunk.branch, chunk.jurisdiction) == (
            DocumentType.MANUAL,
            "Manual X",
            "pioneiro",
            "nacional",
        )
        assert (chunk.effective_date, chunk.status, chunk.version) == (date(2026, 3, 1), ValidityStatus.VIGENTE, "2026")
        assert len(encoded.dense) == 64 and any(encoded.dense)
        assert encoded.sparse.indices
    ids = sorted(build.chunks)
    assert ids == ["manual-x-2026#4.2@p2", "manual-x-2026#4@p1", "manual-x-2026#s1@p1"]
    registro = build.documents["manual-x-2026"]
    assert registro.pages == tuple(MANUAL_PAGINAS)
    assert relatorio.version is not None and relatorio.version.chunking == "estrutural-v1;janela=300-600"


def test_us1_cen2_reingestao_reaproveita_com_os_mesmos_ids() -> None:
    amb = Ambiente()
    primeiro = amb.ingest(catalogo(entrada()))
    ids = sorted(amb.storage.builds[primeiro.build].chunks)
    amb.embedder.passages = 0
    segundo = amb.ingest(catalogo(entrada()))
    assert segundo.ok
    assert segundo.build != primeiro.build and segundo.build == amb.publicado()
    assert list(amb.storage.builds) == [segundo.build]  # o build anterior foi apagado
    assert sorted(amb.storage.builds[segundo.build].chunks) == ids
    assert amb.embedder.passages == 0
    assert [d.reused for d in segundo.documents] == [True]


def test_metadado_de_documento_muda_sem_recodificar() -> None:
    amb = Ambiente()
    amb.ingest(catalogo(entrada()))
    amb.embedder.passages = 0
    relatorio = amb.ingest(catalogo(entrada(branch="senior", jurisdiction="regional")))
    assert relatorio.ok and amb.embedder.passages == 0
    chunks = [e.chunk for e in amb.storage.builds[relatorio.build].chunks.values()]
    assert {(c.branch, c.jurisdiction) for c in chunks} == {("senior", "regional")}


def test_titulo_novo_recodifica() -> None:
    amb = Ambiente()
    amb.ingest(catalogo(entrada()))
    amb.embedder.passages = 0
    relatorio = amb.ingest(catalogo(entrada(title="Manual X, 2ª edição")))
    assert relatorio.ok and amb.embedder.passages == 3
    assert [d.reused for d in relatorio.documents] == [False]


def test_tudo_ou_nada() -> None:
    amb = Ambiente()
    publicado = amb.ingest(catalogo(entrada())).build
    amb.parser.put(BASE / "outro.pdf", ["Texto."])
    amb.embedder.passages = 0
    relatorio = amb.ingest(catalogo(entrada(), entrada(file="outro.pdf", document_id="outro", title=...)))
    assert not relatorio.ok
    assert any("title" in p for p in relatorio.problems)
    assert relatorio.build is None
    assert amb.publicado() == publicado
    assert list(amb.storage.builds) == [publicado]
    assert amb.embedder.passages == 0


def test_pdf_sem_camada_de_texto_e_pagina_vazia() -> None:
    amb = Ambiente()
    amb.parser.put(BASE / "scan.pdf", ["", "  \n "])
    relatorio = amb.ingest(catalogo(entrada(file="scan.pdf")))
    assert not relatorio.ok
    assert any("sem camada de texto (OCR fora do escopo)" in p for p in relatorio.problems)

    amb.parser.put(BASE / "capa.pdf", ["", *MANUAL_PAGINAS])
    relatorio = amb.ingest(catalogo(entrada(file="capa.pdf")))
    assert relatorio.ok
    assert relatorio.documents[0].warnings == ("página 1 sem texto",)


def test_arquivo_inexistente() -> None:
    relatorio = Ambiente().ingest(catalogo(entrada(file="sumiu.pdf")))
    assert not relatorio.ok
    assert any("sumiu.pdf" in p and "não encontrado" in p for p in relatorio.problems)


def test_documento_que_sai_do_catalogo_sai_do_indice() -> None:
    amb = Ambiente()
    amb.parser.put(BASE / "y.pdf", ["Outro manual.\nCom texto."])
    amb.ingest(catalogo(entrada(), entrada(file="y.pdf", document_id="manual-y", version="1", branch=...)))
    relatorio = amb.ingest(catalogo(entrada()))
    assert relatorio.ok
    assert relatorio.removed == ("manual-y-1",)
    assert set(amb.storage.builds[relatorio.build].documents) == {"manual-x-2026"}


def test_versao_incompativel_recusa_sem_escrever() -> None:
    amb = Ambiente(fingerprint="a")
    publicado = amb.ingest(catalogo(entrada())).build
    relatorio = amb.ingest(catalogo(entrada()), amb.servico(CountingEmbedder(fingerprint="b")))
    assert not relatorio.ok
    assert any("reindex" in p for p in relatorio.problems)
    assert amb.publicado() == publicado and list(amb.storage.builds) == [publicado]


def test_status() -> None:
    amb = Ambiente()
    vazio = asyncio.run(amb.service.status())
    assert vazio.build is None and vazio.documents == ()
    amb.ingest(catalogo(entrada()))
    status = asyncio.run(amb.service.status())
    assert status.build is not None and status.build.name == amb.publicado()
    assert [(d.doc_key, d.fields.status, len(d.chunk_ids)) for d in status.documents] == [
        ("manual-x-2026", ValidityStatus.VIGENTE, 3)
    ]


# --- US2 -------------------------------------------------------------------------------------------------------

NORMA_PAGINAS = [
    "O Conselho resolve:\nCAPÍTULO II – Das Atividades\nSEÇÃO I\nDo planejamento\n"
    "Art. 41. O grupo mantém registro.\nArt. 42. O responsável deverá:\nI – comunicar os pais;",
    "Parágrafo único. O registro fica com o grupo.\nArt. 43. A autorização é obrigatória.",
]


def test_us2_por_divide_por_regra() -> None:
    amb = Ambiente()
    amb.parser.put(
        BASE / "por.pdf",
        [
            "CAPÍTULO 1\nFundamentos\nREGRA\n001 DEFINIÇÃO\nO Escotismo é um movimento.\n"
            "REGRA\n002 PROPÓSITO\nContribuir."
        ],
    )
    relatorio = amb.ingest(
        catalogo(entrada(file="por.pdf", document_id="por", version="2025", type="por", title="POR", branch=...))
    )
    assert relatorio.ok and relatorio.documents[0].normative_units == 2
    artigos = sorted(e.chunk.article for e in amb.storage.builds[relatorio.build].chunks.values())
    assert artigos == ["Regra 001", "Regra 002"]


def test_us2_norma_um_chunk_por_unidade() -> None:
    for tipo in ("resolucao", "estatuto"):
        amb = Ambiente()
        amb.parser.put(BASE / "norma.pdf", NORMA_PAGINAS)
        relatorio = amb.ingest(
            catalogo(
                entrada(file="norma.pdf", document_id="norma-x", version="1", type=tipo, title="Norma X", branch=...)
            )
        )
        assert relatorio.ok, relatorio.problems
        assert relatorio.documents[0].normative_units == 3
        chunks = {e.chunk.chunk_id: e.chunk for e in amb.storage.builds[relatorio.build].chunks.values()}
        normativos = [c for c in chunks.values() if c.article]
        assert sorted(c.article for c in normativos) == ["Art. 41", "Art. 42", "Art. 43"]
        for chunk in normativos:  # SC-001: caminho ancestral completo e citação derivável
            assert chunk.text.startswith("[Norma X · Capítulo II · Seção I]\n")
            assert chunk.to_chunk().section == chunk.article
        art42 = chunks["norma-x-1#art42@p1"]
        assert (art42.page_start, art42.page_end) == (1, 2)
        assert "Parágrafo único" in art42.text
        assert art42.chapter == "Capítulo II – Das Atividades"
        assert "norma-x-1#preambulo@p1" in chunks


def test_us2_manual_com_artigo_no_texto_continua_narrativo() -> None:
    amb = Ambiente()
    amb.parser.put(BASE / "m.pdf", ["4.2 Regras\nConforme o Art. 3 do estatuto, a patrulha decide.\nArt. 3 é citado."])
    relatorio = amb.ingest(catalogo(entrada(file="m.pdf")))
    assert relatorio.ok and relatorio.documents[0].normative_units == 0
    assert all(e.chunk.article is None for e in amb.storage.builds[relatorio.build].chunks.values())


def test_us2_normativo_sem_unidade_vira_janelas_com_aviso() -> None:
    amb = Ambiente()
    amb.parser.put(BASE / "r.pdf", ["Resolução sem artigos numerados.\nApenas texto corrido."])
    relatorio = amb.ingest(catalogo(entrada(file="r.pdf", type="resolucao", branch=...)))
    assert relatorio.ok
    (doc,) = relatorio.documents
    assert doc.normative_units == 0 and doc.chunks == 1
    assert doc.warnings == ("normativo sem unidade reconhecida: dividido em janelas",)


def test_us2_unidade_longa_avisa() -> None:
    amb = Ambiente()
    longo = "Art. 1º " + " ".join(["A atividade exige cuidado."] * 90)
    amb.parser.put(BASE / "e.pdf", [longo])
    relatorio = amb.ingest(catalogo(entrada(file="e.pdf", type="estatuto", branch=...)))
    assert relatorio.ok
    (aviso,) = relatorio.documents[0].warnings
    assert aviso.startswith("unidade art1 com ~") and "trunca em 512" in aviso


# --- US3 -------------------------------------------------------------------------------------------------------

HOJE = date(2026, 10, 7)
V2023: dict[str, Any] = {"file": "m2023.pdf", "version": "2023", "effective_date": date(2023, 1, 1)}
V2026: dict[str, Any] = {
    "file": "m2026.pdf",
    "version": "2026",
    "effective_date": date(2026, 3, 1),
    "supersedes": "2023",
}


def ambiente_com_versoes() -> Ambiente:
    amb = Ambiente()
    amb.parser.put(BASE / "m2023.pdf", MANUAL_PAGINAS)
    amb.parser.put(BASE / "m2026.pdf", ["APRESENTAÇÃO\nEdição revista do manual.", "4.2 Acampamento\nNovo cardápio."])
    return amb


def chunks_de(amb: Ambiente, build: str, doc_key: str) -> list[Any]:
    return [e.chunk for e in amb.storage.builds[build].chunks.values() if e.chunk.doc_key == doc_key]


def test_us3_sucessor_marca_o_antecessor_sem_recodificar() -> None:
    amb = ambiente_com_versoes()
    primeiro = amb.ingest(catalogo(entrada(**V2023)))
    assert [(d.doc_key, d.status) for d in primeiro.documents] == [("manual-x-2023", ValidityStatus.VIGENTE)]
    amb.embedder.passages = 0

    segundo = amb.ingest(catalogo(entrada(**V2023), entrada(**V2026)))
    assert segundo.ok, segundo.problems
    assert [(d.doc_key, d.status, d.reused) for d in segundo.documents] == [
        ("manual-x-2023", ValidityStatus.SUBSTITUIDO, True),
        ("manual-x-2026", ValidityStatus.VIGENTE, False),
    ]
    assert amb.embedder.passages == segundo.documents[1].chunks  # só a 2026 foi codificada
    for chunk in chunks_de(amb, segundo.build, "manual-x-2023"):
        assert chunk.status is ValidityStatus.SUBSTITUIDO
        assert chunk.superseded_by == "manual-x-2026"
        assert chunk.expiry_date == date(2026, 2, 28)

    filtros = IndexFilters(effective_from=HOJE, effective_until=HOJE)
    achados = asyncio.run(amb.index.search("manual acampamento cardápio", filters=filtros, limit=10))
    assert achados and {c.doc_key for c in achados} == {"manual-x-2026"}


def test_us3_duas_vigentes_na_linhagem_rejeita() -> None:
    amb = ambiente_com_versoes()
    publicado = amb.ingest(catalogo(entrada(**V2023))).build
    relatorio = amb.ingest(catalogo(entrada(**V2023), entrada(**{**V2026, "supersedes": ...})))
    assert not relatorio.ok
    (problema,) = relatorio.problems
    assert "manual-x" in problema and "2023" in problema and "2026" in problema and "vigente" in problema
    assert amb.publicado() == publicado


def test_us3_rascunho_indexado_mas_fora_da_busca() -> None:
    amb = ambiente_com_versoes()
    relatorio = amb.ingest(catalogo(entrada(**V2023), entrada(**{**V2026, "status": "rascunho"})))
    assert relatorio.ok
    assert [(d.doc_key, d.status) for d in relatorio.documents] == [
        ("manual-x-2023", ValidityStatus.VIGENTE),
        ("manual-x-2026", ValidityStatus.RASCUNHO),
    ]
    assert chunks_de(amb, relatorio.build, "manual-x-2026")
    amplo = IndexFilters(effective_from=date(2000, 1, 1), effective_until=date(2100, 1, 1), include_superseded=True)
    achados = asyncio.run(amb.index.search("manual edição revista cardápio", filters=amplo, limit=20))
    assert achados and "manual-x-2026" not in {c.doc_key for c in achados}


# --- US4 -------------------------------------------------------------------------------------------------------


def test_us4_politica_diferente_exige_reindex() -> None:
    amb = Ambiente()
    publicado = amb.ingest(catalogo(entrada())).build
    index = MemoryIndex("scouting", amb.embedder, Bm25Encoder(), storage=amb.storage)
    outra = IngestionService(
        parser=amb.parser,
        embedder=amb.embedder,
        sparse=Bm25Encoder(),
        writer=index,
        store=index,
        clock=FixedClock(INSTANTE),
        policy=ChunkingPolicy(window_min=200, window_max=400),
    )
    relatorio = amb.ingest(catalogo(entrada()), outra)
    assert not relatorio.ok and any("janela=200-400" in p and "reindex" in p for p in relatorio.problems)
    assert amb.publicado() == publicado


# --- US5 -------------------------------------------------------------------------------------------------------


def test_us5_reindex_com_outro_embedder() -> None:
    amb = ambiente_com_versoes()
    primeiro = amb.ingest(catalogo(entrada(**V2023), entrada(**V2026)))
    antes = {k: sorted(b.chunks) for k, b in amb.storage.builds.items()}[primeiro.build]
    novo = CountingEmbedder(fingerprint="novo", dimensions=32)
    servico = amb.servico(novo)
    relatorio = asyncio.run(servico.reindex(catalogo(entrada(**V2023), entrada(**V2026)), BASE))
    assert relatorio.ok, relatorio.problems
    assert relatorio.build == amb.publicado() and list(amb.storage.builds) == [relatorio.build]
    build = amb.storage.builds[relatorio.build]
    assert sorted(build.chunks) == antes  # mesmos chunk_ids
    assert sorted(build.documents) == ["manual-x-2023", "manual-x-2026"]  # mesmos documentos
    assert not any(d.reused for d in relatorio.documents)
    assert novo.passages == len(antes)
    assert build.version.embedding.fingerprint == "novo" and build.version.built_at == INSTANTE
    assert {len(e.dense) for e in build.chunks.values()} == {32}

    # reindexar de novo, com a mesma configuração, dá os mesmos ids e contagens (FR-010)
    outra_vez = asyncio.run(servico.reindex(catalogo(entrada(**V2023), entrada(**V2026)), BASE))
    assert sorted(amb.storage.builds[outra_vez.build].chunks) == antes
    assert [(d.doc_key, d.chunks) for d in outra_vez.documents] == [(d.doc_key, d.chunks) for d in relatorio.documents]

    # o ingest com a configuração antiga agora recusa e manda reindexar
    recusado = amb.ingest(catalogo(entrada(**V2023), entrada(**V2026)))
    assert not recusado.ok and any("reindex" in p for p in recusado.problems)


def test_us5_reindex_com_catalogo_rejeitado_nao_escreve() -> None:
    amb = ambiente_com_versoes()
    publicado = amb.ingest(catalogo(entrada(**V2023))).build
    relatorio = asyncio.run(
        amb.servico(CountingEmbedder(fingerprint="novo")).reindex(catalogo(entrada(**V2023, title=...)), BASE)
    )
    assert not relatorio.ok
    assert amb.publicado() == publicado and list(amb.storage.builds) == [publicado]


def test_us5_original_de_qualquer_chunk() -> None:
    amb = Ambiente()
    amb.ingest(catalogo(entrada()))
    original = asyncio.run(amb.service.original("manual-x-2026#4.2@p2"))
    assert original is not None
    assert original.pages == (MANUAL_PAGINAS[1],)
    assert asyncio.run(amb.service.original("nada#x@p1")) is None


def test_problemas_de_catalogo_e_de_linhagem_juntos() -> None:
    """O curador vê tudo de uma vez: campo ausente numa linhagem e duas vigentes em outra (quickstart §4)."""
    amb = ambiente_com_versoes()
    amb.parser.put(BASE / "norma.pdf", ["Art. 1º Texto."])
    norma = entrada(file="norma.pdf", document_id="norma-x", version="1", type="resolucao", title=..., branch=...)
    relatorio = amb.ingest(catalogo(entrada(**V2023), entrada(**{**V2026, "supersedes": ...}), norma))
    assert not relatorio.ok
    assert any("title" in p for p in relatorio.problems)
    assert any("manual-x" in p and "vigente" in p for p in relatorio.problems)
    assert not any("norma-x" in p and "linhagem" in p for p in relatorio.problems)  # linhagem com entrada inválida
    assert amb.publicado() is None


def test_link_do_arquivo_vai_para_o_registro_e_para_o_original() -> None:
    amb = Ambiente()
    url = "https://www.escoteiros.org.br/manual-x.pdf"
    relatorio = amb.ingest(catalogo(entrada(source_url=url)))
    assert amb.storage.builds[relatorio.build].documents["manual-x-2026"].source_url == url
    original = asyncio.run(amb.service.original("manual-x-2026#4.2@p2"))
    assert original is not None and original.source_url == url
    amb.embedder.passages = 0
    novo = amb.ingest(catalogo(entrada(source_url="https://www.escoteiros.org.br/outro.pdf")))
    assert amb.embedder.passages == 0  # trocar o link não recodifica
    assert amb.storage.builds[novo.build].documents["manual-x-2026"].source_url.endswith("outro.pdf")
