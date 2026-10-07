"""CLI `ai-scouting-corpus` (spec 102 contracts/ingestion-cli.md; T026, T041, T046)."""

from __future__ import annotations

import io
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from ai_scouting.rag.index.adapters.bm25 import Bm25Encoder
from ai_scouting.rag.index.adapters.memory import HashingEmbedder, MemoryIndex, MemoryStorage
from ai_scouting.rag.index.api import IndexUnavailable
from ai_scouting.rag.ingestion.adapters.catalog_yaml import CatalogUnreadable
from ai_scouting.rag.ingestion.adapters.cli import Composition, ConfigurationError, main
from ai_scouting.rag.ingestion.adapters.memory import FixedClock, MemoryParser
from ai_scouting.rag.ingestion.service import IngestionService

BASE = Path("/corpus")
SEGREDO = "s3cr3t-qdrant-key"
ENTRADA = {
    "file": "manual.pdf",
    "document_id": "manual-x",
    "version": "2026",
    "type": "manual",
    "title": "Manual X",
    "jurisdiction": "nacional",
    "effective_date": date(2026, 3, 1),
}


class Saida:
    def __init__(self, catalogo: object, *, falha: Exception | None = None) -> None:
        self.out, self.err = io.StringIO(), io.StringIO()
        self.storage = MemoryStorage()
        self.catalogo = catalogo
        self.falha = falha
        self.parser = MemoryParser({BASE / "manual.pdf": ["APRESENTAÇÃO\nTexto do manual.", "4.2 Acampamento\nMais."]})

    def composition(self) -> Composition:
        @asynccontextmanager
        async def open_service(needs_model: bool) -> AsyncIterator[IngestionService]:
            if self.falha is not None:
                raise self.falha
            index = MemoryIndex("scouting", HashingEmbedder(), Bm25Encoder(), storage=self.storage)
            yield IngestionService(
                parser=self.parser,
                embedder=HashingEmbedder(),
                sparse=Bm25Encoder(),
                writer=index,
                clock=FixedClock(datetime(2026, 10, 7, 15, 30, tzinfo=UTC)),
            )

        def load_catalog(path: Path) -> tuple[object, Path]:
            if isinstance(self.catalogo, Exception):
                raise self.catalogo
            return self.catalogo, BASE

        def fetch_model(directory: Path) -> Sequence[tuple[Path, str]]:
            return [(directory / "model.onnx", "ca456c06")]

        return Composition(
            index_name="scouting",
            open_service=open_service,
            load_catalog=load_catalog,
            fetch_model=fetch_model,
            secrets=(SEGREDO,),
            out=self.out,
            err=self.err,
        )

    def run(self, *argv: str) -> int:
        return main(list(argv), self.composition())

    @property
    def texto(self) -> str:
        return self.out.getvalue() + self.err.getvalue()


def test_ingest_com_sucesso() -> None:
    saida = Saida({"documents": [ENTRADA]})
    assert saida.run("ingest", "/corpus/corpus.yaml") == 0
    texto = saida.out.getvalue()
    assert "índice scouting — build scouting__20261007T153000Z publicado" in texto
    assert "versão: estrutural-v1;janela=300-600 · hashing-test" in texto
    assert "manual-x-2026" in texto and "codificado" in texto
    assert "duração: 0m00s" in texto
    assert saida.run("ingest", "/corpus/corpus.yaml") == 0
    assert "reaproveitado" in saida.out.getvalue()


def test_rejeicao() -> None:
    saida = Saida({"documents": [{**ENTRADA, "title": None}]})
    assert saida.run("ingest", "/corpus/corpus.yaml") == 1
    assert "title" in saida.out.getvalue() and "nada foi aplicado" in saida.out.getvalue()
    assert not saida.storage.aliases


def test_catalogo_ilegivel_e_inexistente() -> None:
    assert Saida(CatalogUnreadable("YAML inválido")).run("ingest", "x.yaml") == 1
    assert Saida(FileNotFoundError("x.yaml")).run("ingest", "x.yaml") == 2


@pytest.mark.parametrize(
    "falha", [ConfigurationError("AI_SCOUTING_EMBEDDING_DIR ausente"), IndexUnavailable("índice fora do ar")]
)
def test_configuracao_ou_infraestrutura(falha: Exception) -> None:
    saida = Saida({"documents": [ENTRADA]}, falha=falha)
    assert saida.run("ingest", "/corpus/corpus.yaml") == 2
    assert str(falha) in saida.err.getvalue()


def test_segredo_nunca_aparece() -> None:
    saida = Saida({"documents": [ENTRADA]}, falha=IndexUnavailable(f"falha com api-key={SEGREDO}"))
    assert saida.run("status") == 2
    assert SEGREDO not in saida.texto
    assert "***" in saida.texto


def test_status() -> None:
    saida = Saida({"documents": [ENTRADA]})
    assert saida.run("status") == 0
    assert "nenhum build publicado" in saida.out.getvalue()
    saida.run("ingest", "/corpus/corpus.yaml")
    assert saida.run("status") == 0
    texto = saida.out.getvalue()
    assert "manual-x" in texto and "vigente" in texto and "1 documentos, 2 chunks" in texto


def test_model_fetch() -> None:
    saida = Saida({})
    assert saida.run("model", "fetch", "/tmp/e5") == 0
    assert "model.onnx  sha256 ca456c06" in saida.out.getvalue()


def test_uso_invalido() -> None:
    with pytest.raises(SystemExit) as exc:
        Saida({}).run("desconhecido")
    assert exc.value.code == 2
