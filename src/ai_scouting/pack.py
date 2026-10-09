"""`ScoutingPack`: o pack de domínio escoteiro (spec 101).

Composition root do pack (ADR-012): aqui só se compõe — as políticas vivem em `context/`, `prompts/`, `rag/` e
`validators/`. A 101 não faz I/O, então não declara porta secundária (princípio XIII). Na Fase 5 (research R16–R20) o
manifest declara o flow preferido `rag_first`, a coleção `scouting` e o esquema de configuração, e o pack publica a
recuperação da 103 — o índice da 102 no Qdrant, pelo `config` do `packs.yaml` — e os validators da 104. A chave do
Qdrant vem só do ambiente (`AI_SCOUTING_QDRANT_API_KEY`, FR-012). Os imports de infraestrutura ficam dentro dos hooks:
ler a identidade do pack (002) continua barato.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

from ai_core_contracts import (
    AnswerValidator,
    JSONValue,
    NamedRetriever,
    PackBase,
    PackContext,
    PackManifest,
    Reranker,
    TurnView,
)

from ai_scouting.context.derivation import derive_context, to_slice
from ai_scouting.context.periods import reference_day
from ai_scouting.prompts import SYSTEM_PROMPT

VERSION: Final = "0.2.0"
"""Versão do manifest; igual à da distribuição `ai-scouting` (verificado em tests/test_pack.py)."""

COLLECTION: Final = "scouting"
"""A coleção lógica que o Assistant concede (research R17): fixa no pack, porque o manifest não recebe configuração. O
que a configuração escolhe é o índice — o alias no Qdrant."""

DEFAULT_QDRANT_URL: Final = "http://localhost:6333"
DEFAULT_INDEX: Final = "scouting"
DEFAULT_LIMIT: Final = 5
MAX_LIMIT: Final = 30
MODEL_FILES: Final = ("model.onnx", "tokenizer.json")
API_KEY_VARIABLE: Final = "AI_SCOUTING_QDRANT_API_KEY"
"""A única fonte da chave do Qdrant (FR-012): nunca o esquema, o `packs.yaml` ou uma mensagem."""

SCOUTING_CONFIG_SCHEMA: Final[Mapping[str, JSONValue]] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "qdrant_url": {"type": "string", "pattern": "^https?://"},
        "qdrant_path": {"type": "string", "minLength": 1},
        "index": {"type": "string", "pattern": "^[a-z0-9][a-z0-9_-]*$"},
        "embedder": {"enum": ["e5", "hashing"]},
        "embedding_dir": {"type": "string", "minLength": 1},
        "default_limit": {"type": "integer", "minimum": 1},
        "max_limit": {"type": "integer", "minimum": 1},
    },
}
"""O `config` do pack no `packs.yaml` (research R18), validado pelo core no boot. Os defaults são do pack
(`read_config`): o core valida o esquema, mas não os aplica."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ScoutingConfig:
    """A configuração lida, com os defaults aplicados (data-model "Configuração do pack")."""

    qdrant_url: str | None
    qdrant_path: str | None
    index: str
    embedder: Literal["e5", "hashing"]
    embedding_dir: str | None
    default_limit: int
    max_limit: int


def read_config(config: Mapping[str, JSONValue]) -> ScoutingConfig:
    """Defaults e as combinações que o esquema sozinho não pega; cada recusa nomeia o campo (princípio VII).

    Não toca a rede — Qdrant fora, índice ausente ou modelo divergente aparecem na primeira busca (103 FR-007). Só
    confere que o diretório do e5 tem os arquivos, para o pack não carregar com um modelo que não existe.
    """
    url, path = _text(config, "qdrant_url"), _text(config, "qdrant_path")
    if url is not None and path is not None:
        raise ValueError("config do scouting: qdrant_url é exclusivo com qdrant_path — escolha um dos dois")
    embedder = config.get("embedder", "e5")
    if embedder not in ("e5", "hashing"):
        raise ValueError(f"config do scouting: embedder precisa ser 'e5' ou 'hashing': {embedder!r}")
    embedding_dir = _text(config, "embedding_dir")
    if embedder == "e5":
        _require_model(embedding_dir)
    default_limit = _count(config, "default_limit", DEFAULT_LIMIT)
    max_limit = _count(config, "max_limit", MAX_LIMIT)
    if default_limit > max_limit:
        raise ValueError(
            f"config do scouting: precisa valer 1 ≤ default_limit ≤ max_limit (default_limit={default_limit}, "
            f"max_limit={max_limit})"
        )
    return ScoutingConfig(
        qdrant_url=url if url is not None or path is not None else DEFAULT_QDRANT_URL,
        qdrant_path=path,
        index=_text(config, "index") or DEFAULT_INDEX,
        embedder="e5" if embedder == "e5" else "hashing",
        embedding_dir=embedding_dir,
        default_limit=default_limit,
        max_limit=max_limit,
    )


def _text(config: Mapping[str, JSONValue], name: str) -> str | None:
    value = config.get(name)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"config do scouting: {name} precisa ser texto não vazio")
    return value


def _count(config: Mapping[str, JSONValue], name: str, default: int) -> int:
    value = config.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"config do scouting: {name} precisa ser inteiro ≥ 1 (1 ≤ default_limit ≤ max_limit)")
    return value


def _require_model(directory: str | None) -> None:
    if directory is None:
        raise ValueError(
            "config do scouting: embedding_dir é obrigatório com o embedder 'e5' — rode `ai-scouting-corpus model "
            "fetch <dir>` e aponte o diretório"
        )
    path = Path(directory).expanduser()
    missing = [name for name in MODEL_FILES if not (path / name).is_file()]
    if missing:
        raise ValueError(
            f"config do scouting: embedding_dir sem {', '.join(missing)} — rode `ai-scouting-corpus model fetch "
            f"{directory}`"
        )


class ScoutingPack(PackBase):
    """Instruções de domínio, a derivação determinística de contexto, a recuperação da 103 e os validators da 104."""

    name = "scouting"
    contract_version = "0.1"

    def manifest(self) -> PackManifest:
        return PackManifest(
            name=self.name,
            version=VERSION,
            preferred_flow="rag_first",
            requires_collections=(COLLECTION,),
            config_schema=SCOUTING_CONFIG_SCHEMA,
        )

    def system_prompt(self, ctx: PackContext) -> str | None:
        return SYSTEM_PROMPT

    def retrievers(self, ctx: PackContext) -> Sequence[NamedRetriever]:
        """A tool `scouting.search_docs` sobre o índice configurado (R19). A carga não toca a rede: Qdrant fora, índice
        ausente ou modelo divergente aparecem na busca, como erro explícito (103 FR-007)."""
        import os

        from qdrant_client import AsyncQdrantClient

        from ai_scouting.rag.index.adapters.bm25 import Bm25Encoder
        from ai_scouting.rag.index.adapters.e5 import E5Embedder
        from ai_scouting.rag.index.adapters.memory import HashingEmbedder
        from ai_scouting.rag.index.adapters.qdrant import QdrantIndex
        from ai_scouting.rag.index.ports import Embedder
        from ai_scouting.rag.retrieval.domain.config import RetrievalConfig
        from ai_scouting.rag.retrieval.retriever import ScoutingRetriever, search_docs

        config = read_config(ctx.config)
        if config.qdrant_path is not None:
            client = AsyncQdrantClient(path=config.qdrant_path)
        else:
            # Sem a checagem de versão do cliente, que iria à rede na carga; a compatibilidade com o servidor é da suíte
            # do índice (102), e o servidor fora aparece na primeira busca.
            client = AsyncQdrantClient(
                url=config.qdrant_url, api_key=os.environ.get(API_KEY_VARIABLE) or None, check_compatibility=False
            )
        embedder: Embedder
        if config.embedder == "e5" and config.embedding_dir is not None:
            embedder = E5Embedder(Path(config.embedding_dir).expanduser())
        else:
            embedder = HashingEmbedder()
        index = QdrantIndex(client, config.index, embedder, Bm25Encoder())
        retrieval = RetrievalConfig(default_limit=config.default_limit, max_limit=config.max_limit)
        return [search_docs(ScoutingRetriever(index, retrieval), collection=COLLECTION)]

    def reranker(self, ctx: PackContext) -> Reranker | None:
        from ai_scouting.rag.retrieval.reranker import MetadataReranker

        return MetadataReranker()

    def validators(self, ctx: PackContext) -> Sequence[AnswerValidator]:
        """A integridade das citações primeiro: o aviso de vigência lê as citações que ela validou (104; R19)."""
        from ai_scouting.validators.integrity import CitationIntegrityValidator
        from ai_scouting.validators.notice import ValidityNoticeValidator

        return (CitationIntegrityValidator(), ValidityNoticeValidator())

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
