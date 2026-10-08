"""`ScoutingPack`: o pack de domínio escoteiro (spec 101).

Composition root do pack (ADR-012): aqui só se compõe — as políticas vivem em `context/` e `prompts/`. A 101 não faz
I/O, então não declara porta secundária (princípio XIII). Na Fase 4 o manifest declara só nome e versão: flow
preferido, coleção exigida e esquema de configuração chegam com a US5, na Fase 5 (clarificação de 2026-10-07).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from ai_core_contracts import JSONValue, PackBase, PackContext, PackManifest, TurnView

from ai_scouting.context.derivation import derive_context, to_slice
from ai_scouting.context.periods import reference_day
from ai_scouting.prompts import SYSTEM_PROMPT

VERSION: Final = "0.1.0"
"""Versão do manifest; igual à da distribuição `ai-scouting` (verificado em tests/test_pack.py)."""


class ScoutingPack(PackBase):
    """Instruções de domínio e a derivação determinística de contexto da pergunta."""

    name = "scouting"
    contract_version = "0.1"

    def manifest(self) -> PackManifest:
        return PackManifest(name=self.name, version=VERSION)

    def system_prompt(self, ctx: PackContext) -> str | None:
        return SYSTEM_PROMPT

    async def prepare_context(self, view: TurnView, ctx: PackContext) -> Mapping[str, JSONValue]:
        """Contexto e filtros de recuperação derivados da pergunta, sem modelo e sem relógio (FR-004, FR-005)."""
        return to_slice(derive_context(view.question, reference_day(view.reference_time)))


def corpus_cli() -> None:
    """`ai-scouting-corpus` (spec 102, research R17): compõe a CLI offline da ingestão.

    O `pack.py` é a composition root do pack também para o processo offline. Os imports de infraestrutura ficam aqui
    dentro, para a carga do pack pelo core não pagar por eles.
    """
    import os
    import sys
    from collections.abc import AsyncIterator
    from contextlib import asynccontextmanager
    from pathlib import Path

    from qdrant_client import AsyncQdrantClient

    from ai_scouting.rag.index.adapters.bm25 import Bm25Encoder
    from ai_scouting.rag.index.adapters.e5 import E5Embedder, fetch
    from ai_scouting.rag.index.adapters.qdrant import QdrantIndex
    from ai_scouting.rag.ingestion.adapters import cli
    from ai_scouting.rag.ingestion.adapters.catalog_yaml import load_catalog
    from ai_scouting.rag.ingestion.adapters.clock import SystemClock
    from ai_scouting.rag.ingestion.adapters.pdfium import PdfiumParser
    from ai_scouting.rag.ingestion.service import IngestionService

    url = os.environ.get("AI_SCOUTING_QDRANT_URL", "http://localhost:6333")
    api_key = os.environ.get("AI_SCOUTING_QDRANT_API_KEY") or None
    index_name = os.environ.get("AI_SCOUTING_INDEX", "scouting")

    def model_dir(needed: bool) -> Path:
        directory = os.environ.get("AI_SCOUTING_EMBEDDING_DIR")
        if not directory:
            if needed:
                raise cli.ConfigurationError(
                    "AI_SCOUTING_EMBEDDING_DIR ausente: rode `ai-scouting-corpus model fetch <dir>` e aponte a "
                    "variável para esse diretório"
                )
            return Path()
        path = Path(directory).expanduser()
        missing = [name for name in ("model.onnx", "tokenizer.json") if not (path / name).is_file()]
        if needed and missing:
            raise cli.ConfigurationError(
                f"{path} sem {', '.join(missing)}: rode `ai-scouting-corpus model fetch {path}`"
            )
        return path

    @asynccontextmanager
    async def open_service(needs_model: bool) -> AsyncIterator[IngestionService]:
        embedder = E5Embedder(model_dir(needs_model))
        sparse = Bm25Encoder()
        client = AsyncQdrantClient(url=url, api_key=api_key)
        try:
            index = QdrantIndex(client, index_name, embedder, sparse)
            yield IngestionService(
                parser=PdfiumParser(),
                embedder=embedder,
                sparse=sparse,
                writer=index,
                store=index,
                clock=SystemClock(),
            )
        finally:
            await client.close()

    composition = cli.Composition(
        index_name=index_name,
        open_service=open_service,
        load_catalog=load_catalog,
        fetch_model=fetch,
        secrets=(api_key,) if api_key else (),
    )
    sys.exit(cli.main(sys.argv[1:], composition))
