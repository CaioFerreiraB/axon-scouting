"""Aceitação do serviço de ingestão com adapters em memória (spec 102: US1 T025; US2 T037; US3 T041; US5 T046)."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from ai_scouting.rag.index.adapters.bm25 import Bm25Encoder
from ai_scouting.rag.index.adapters.memory import HashingEmbedder, MemoryIndex, MemoryStorage
from ai_scouting.rag.index.api import DocumentType, ValidityStatus
from ai_scouting.rag.ingestion.adapters.memory import FixedClock, MemoryParser
from ai_scouting.rag.ingestion.service import IngestionService

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
