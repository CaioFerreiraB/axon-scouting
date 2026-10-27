"""`ScoutingPack`: o pack de domínio escoteiro (spec 101).

Composition root do pack (ADR-012): aqui só se compõe — as políticas vivem em `context/`, `prompts/`, `rag/`,
`validators/` e `eval/`. A 101 não faz I/O, então não declara porta secundária (princípio XIII). Na Fase 5 (research
R16–R20) o manifest declara o flow preferido `rag_first`, a coleção `scouting` e o esquema de configuração, e o pack
publica a recuperação da 103 — o índice da 102 no Qdrant, pelo `config` do `packs.yaml` — e os validators da 104; na
Fase 6, as métricas de domínio da 105 (hook `eval_metrics`, ADR-027). A chave do Qdrant vem só do ambiente
(`AXON_SCOUTING_QDRANT_API_KEY`, FR-012). Os imports de infraestrutura ficam dentro dos hooks: ler a identidade do pack
(002) continua barato. Na Fase 9 (US6, ADR-037) o pack se chama `axon-scouting` e recusa as variáveis `AI_SCOUTING_*`
(research R26). Desde o ADR-043 o índice de produção é o artefato distribuído com o pacote, carregado em memória
(`store: memory`; o Qdrant fica como opção), com o modelo denso escolhido em `embedder` (`e5`, `pplx` ou `hashing`); e,
pelo ADR-041, o hook `prepare` baixa o e5 no build, em `data_dir/e5-small`, só quando ele é o escolhido.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal

from axon_ai_contracts import (
    AnswerValidator,
    JSONValue,
    NamedRetriever,
    PackBase,
    PackContext,
    PackManifest,
    Reranker,
    TurnView,
)
from axon_ai_contracts.eval import EvalMetric

from axon_scouting.context.derivation import derive_context, to_slice
from axon_scouting.context.periods import reference_day
from axon_scouting.prompts import SYSTEM_PROMPT

if TYPE_CHECKING:
    from axon_scouting.rag.index.api import ChunkIndex
    from axon_scouting.rag.index.ports import Embedder

VERSION: Final = "0.5.0"
"""Versão do manifest; igual à da distribuição `axon-scouting` (verificado em tests/test_pack.py)."""

COLLECTION: Final = "scouting"
"""A coleção lógica que o Assistant concede (research R17): fixa no pack, porque o manifest não recebe configuração. O
que a configuração escolhe é o índice — o alias no Qdrant."""

DEFAULT_QDRANT_URL: Final = "http://localhost:6333"
DEFAULT_INDEX: Final = "scouting"
PACKAGED_INDEX: Final = Path(__file__).parent / "data" / "index"
"""O artefato do índice gerado no desenvolvimento (ADR-043), distribuído como dado do pacote."""
E5_SUBDIR: Final = "e5-small"
"""Onde o `prepare` baixa o e5 dentro do `data_dir` (ADR-041)."""
EMBEDDERS: Final = ("e5", "pplx", "hashing")
STORES: Final = ("memory", "qdrant")
DEFAULT_LIMIT: Final = 5
MAX_LIMIT: Final = 30
MODEL_FILES: Final = ("model.onnx", "tokenizer.json")
API_KEY_VARIABLE: Final = "AXON_SCOUTING_QDRANT_API_KEY"
"""A única fonte da chave do Qdrant (FR-012): nunca o esquema, o `packs.yaml` ou uma mensagem."""
EMBEDDING_KEY_VARIABLE: Final = "AXON_SCOUTING_EMBEDDING_API_KEY"
"""A chave do embedder remoto (ADR-043, item 5): a do pack, nunca a do core."""

PREFIX: Final = "AXON_SCOUTING_"
LEGACY_PREFIX: Final = "AI_SCOUTING_"
"""O prefixo de antes do ADR-037: corte seco, sem leitura de compatibilidade (FR-014)."""

SCOUTING_CONFIG_SCHEMA: Final[Mapping[str, JSONValue]] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "store": {"enum": list(STORES)},
        "index_dir": {"type": "string", "minLength": 1},
        "qdrant_url": {"type": "string", "pattern": "^https?://"},
        "qdrant_path": {"type": "string", "minLength": 1},
        "index": {"type": "string", "pattern": "^[a-z0-9][a-z0-9_-]*$"},
        "embedder": {"enum": list(EMBEDDERS)},
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

    store: Literal["memory", "qdrant"]
    index_dir: str | None
    qdrant_url: str | None
    qdrant_path: str | None
    index: str
    embedder: Literal["e5", "pplx", "hashing"]
    embedding_dir: str | None
    default_limit: int
    max_limit: int


def read_config(
    config: Mapping[str, JSONValue], data_dir: Path | None = None, *, require_model: bool = True
) -> ScoutingConfig:
    """Defaults e as combinações que o esquema sozinho não pega; cada recusa nomeia o campo (princípio VII).

    Não toca a rede — Qdrant fora, índice ausente no servidor, chave do embedder remoto ausente ou modelo divergente
    aparecem na primeira busca (103 FR-007). Com o e5, o diretório do modelo é o `embedding_dir` ou, sem ele,
    `data_dir/e5-small` (ADR-041); `require_model=False` é o `prepare`, que vai baixá-lo.
    """
    store = config.get("store", "memory")
    if store not in STORES:
        raise ValueError(f"config do scouting: store precisa ser 'memory' ou 'qdrant': {store!r}")
    url, path = _text(config, "qdrant_url"), _text(config, "qdrant_path")
    index_dir = _text(config, "index_dir")
    if store == "memory":
        extra = [name for name in ("qdrant_url", "qdrant_path", "index") if config.get(name) is not None]
        if extra:
            raise ValueError(f"config do scouting: {', '.join(extra)} só valem com store: qdrant")
    elif index_dir is not None:
        raise ValueError("config do scouting: index_dir só vale com store: memory (o Qdrant tem o próprio índice)")
    if url is not None and path is not None:
        raise ValueError("config do scouting: qdrant_url é exclusivo com qdrant_path — escolha um dos dois")
    embedder = config.get("embedder", "e5")
    if embedder not in EMBEDDERS:
        raise ValueError(f"config do scouting: embedder precisa ser 'e5', 'pplx' ou 'hashing': {embedder!r}")
    embedding_dir = _text(config, "embedding_dir")
    if embedder == "e5":
        if embedding_dir is None and data_dir is not None:
            embedding_dir = str(data_dir / E5_SUBDIR)
        if require_model:
            _require_model(embedding_dir)
        elif embedding_dir is None:
            raise ValueError(
                "config do scouting: com o embedder 'e5', embedding_dir é obrigatório quando a plataforma não entrega "
                "data_dir (AXON_PACKS_DATA_DIR)"
            )
    default_limit = _count(config, "default_limit", DEFAULT_LIMIT)
    max_limit = _count(config, "max_limit", MAX_LIMIT)
    if default_limit > max_limit:
        raise ValueError(
            f"config do scouting: precisa valer 1 ≤ default_limit ≤ max_limit (default_limit={default_limit}, "
            f"max_limit={max_limit})"
        )
    qdrant = store == "qdrant"
    return ScoutingConfig(
        store="qdrant" if qdrant else "memory",
        index_dir=index_dir if not qdrant else None,
        qdrant_url=(url if url is not None or path is not None else DEFAULT_QDRANT_URL) if qdrant else None,
        qdrant_path=path if qdrant else None,
        index=_text(config, "index") or DEFAULT_INDEX,
        embedder="e5" if embedder == "e5" else "pplx" if embedder == "pplx" else "hashing",
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


def _artifact_root(config: ScoutingConfig) -> Path:
    """O diretório do artefato do índice: o configurado ou o que vem no pacote (ADR-043)."""
    root = Path(config.index_dir).expanduser() if config.index_dir is not None else PACKAGED_INDEX
    if not (root / "manifest.json").is_file():
        raise ValueError(
            f"config do scouting: {root} não tem um artefato do índice (manifest.json) — gere com "
            "`axon-scouting-corpus ingest`"
        )
    return root


def _require_model(directory: str | None) -> None:
    if directory is None:
        raise ValueError(
            "config do scouting: embedding_dir é obrigatório com o embedder 'e5' quando a plataforma não entrega "
            "data_dir — rode `axon packs prepare` com AXON_PACKS_DATA_DIR, ou `axon-scouting-corpus model fetch "
            "<dir>` e aponte o diretório"
        )
    path = Path(directory).expanduser()
    missing = [name for name in MODEL_FILES if not (path / name).is_file()]
    if missing:
        raise ValueError(
            f"config do scouting: embedding_dir sem {', '.join(missing)} — rode `axon-scouting-corpus model fetch "
            f"{directory}`"
        )


def _refuse_legacy() -> None:
    import os

    legacy = legacy_environment(os.environ)
    if legacy is not None:
        raise ValueError(legacy)


def _embedder(config: ScoutingConfig, environ: Mapping[str, str]) -> Embedder:
    """O modelo denso da configuração; a pergunta é codificada com o mesmo modelo dos vetores (ADR-043, item 3)."""
    if config.embedder == "e5":
        from axon_scouting.rag.index.adapters.e5 import E5Embedder

        assert config.embedding_dir is not None
        return E5Embedder(Path(config.embedding_dir).expanduser())
    if config.embedder == "pplx":
        from axon_scouting.rag.index.adapters.openrouter import OpenRouterEmbedder

        return OpenRouterEmbedder(environ.get(EMBEDDING_KEY_VARIABLE) or None)
    from axon_scouting.rag.index.adapters.memory import HashingEmbedder

    return HashingEmbedder()


def legacy_environment(environ: Mapping[str, str]) -> str | None:
    """A recusa das variáveis com o nome antigo (research R26), ou `None` sem nenhuma.

    Vale o prefixo exato, com qualquer valor (vazio inclusive), conhecida ou não, e mesmo com a nova definida. A
    mensagem nomeia a variável nova e nunca o valor, que pode ser a chave do Qdrant.
    """
    legacy = sorted(name for name in environ if name.startswith(LEGACY_PREFIX))
    if not legacy:
        return None
    lines = [f"  {name} → {PREFIX}{name.removeprefix(LEGACY_PREFIX)}" for name in legacy]
    return "\n".join(["variáveis com o nome antigo (ADR-037): renomeie", *lines])


class ScoutingPack(PackBase):
    """Instruções de domínio, a derivação determinística de contexto, a recuperação da 103, os validators da 104 e as
    métricas de domínio da 105."""

    name = "scouting"
    contract_version = "0.2.1"
    """0.2.1: o hook `prepare` e o `PackContext.data_dir` (ADR-041)."""

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
        """A tool `scouting.search_docs` sobre o índice configurado (R19): o artefato em memória (ADR-043) ou o
        Qdrant. A carga não toca a rede: Qdrant fora, chave do embedder remoto ausente ou modelo divergente aparecem
        na busca, como erro explícito (103 FR-007). Variável `AI_SCOUTING_*` no ambiente deixa o pack em erro (R26)."""
        import os

        _refuse_legacy()

        from axon_scouting.rag.index.adapters.bm25 import Bm25Encoder
        from axon_scouting.rag.retrieval.domain.config import RetrievalConfig
        from axon_scouting.rag.retrieval.retriever import ScoutingRetriever, search_docs

        config = read_config(ctx.config, ctx.data_dir)
        embedder = _embedder(config, os.environ)
        index: ChunkIndex
        if config.store == "memory":
            from axon_scouting.rag.index.adapters.artifact import ArtifactIndex

            index = ArtifactIndex(_artifact_root(config), embedder, Bm25Encoder())
        else:
            from qdrant_client import AsyncQdrantClient

            from axon_scouting.rag.index.adapters.qdrant import QdrantIndex

            if config.qdrant_path is not None:
                client = AsyncQdrantClient(path=config.qdrant_path)
            else:
                # Sem a checagem de versão do cliente, que iria à rede na carga; a compatibilidade com o servidor é da
                # suíte do índice (102), e o servidor fora aparece na primeira busca.
                client = AsyncQdrantClient(
                    url=config.qdrant_url, api_key=os.environ.get(API_KEY_VARIABLE) or None, check_compatibility=False
                )
            index = QdrantIndex(client, config.index, embedder, Bm25Encoder())
        retrieval = RetrievalConfig(default_limit=config.default_limit, max_limit=config.max_limit)
        return [search_docs(ScoutingRetriever(index, retrieval), collection=COLLECTION)]

    def prepare(self, ctx: PackContext) -> None:
        """No build (ADR-041): baixa o e5 da revisão fixada, conferindo os sha256, em `embedding_dir` ou
        `data_dir/e5-small` — só quando ele é o embedder escolhido. O que já está presente e íntegro não é baixado de
        novo. Com `pplx` ou `hashing`, nada."""
        _refuse_legacy()
        config = read_config(ctx.config, ctx.data_dir, require_model=False)
        if config.embedder != "e5" or config.embedding_dir is None:
            return
        from axon_scouting.rag.index.adapters.e5 import fetch

        fetch(Path(config.embedding_dir).expanduser())

    def reranker(self, ctx: PackContext) -> Reranker | None:
        from axon_scouting.rag.retrieval.reranker import MetadataReranker

        return MetadataReranker()

    def validators(self, ctx: PackContext) -> Sequence[AnswerValidator]:
        """A integridade das citações primeiro: o aviso de vigência lê as citações que ela validou (104; R19)."""
        from axon_scouting.validators.integrity import CitationIntegrityValidator
        from axon_scouting.validators.notice import ValidityNoticeValidator

        return (CitationIntegrityValidator(), ValidityNoticeValidator())

    def eval_metrics(self, ctx: PackContext) -> Sequence[EvalMetric]:
        """As métricas de domínio que o harness da 009 reporta como `scouting:<nome>` (spec 105, ADR-027). A de fatos
        (FR-014) lê o texto dos trechos entregues no artefato, e só existe com o índice em memória — o da avaliação
        (FR-013); o artefato só é lido quando uma rodada a consulta."""
        from axon_scouting.eval import (
            AbsenceClaimRate,
            ContextAccuracy,
            CurrentDocumentRate,
            FactRecall,
            ValidityNoticeRate,
        )

        metrics: list[EvalMetric] = [ContextAccuracy(), CurrentDocumentRate(), ValidityNoticeRate(), AbsenceClaimRate()]
        config = read_config(ctx.config, ctx.data_dir, require_model=False)
        if config.store == "memory":
            from axon_scouting.rag.index.adapters.artifact import chunk_texts

            metrics.append(FactRecall(chunk_texts(_artifact_root(config))))
        return tuple(metrics)

    async def prepare_context(self, view: TurnView, ctx: PackContext) -> Mapping[str, JSONValue]:
        """Contexto e filtros de recuperação derivados da pergunta, sem modelo e sem relógio (FR-004, FR-005)."""
        return to_slice(derive_context(view.question, reference_day(view.reference_time)))


def corpus_cli() -> None:
    """`axon-scouting-corpus` (spec 102, research R17; ADR-043): compõe a CLI offline da ingestão.

    O `pack.py` é a composition root do pack também para o processo offline. A ingestão grava o artefato do índice
    (`--dir`; default, o do pacote — no clone do repositório, o que se commita), com os vetores do modelo de
    `--embedder`; `load-qdrant` o carrega no Qdrant de `AXON_SCOUTING_QDRANT_URL`. Os imports de infraestrutura ficam
    aqui dentro, para a carga do pack pelo core não pagar por eles. Variável `AI_SCOUTING_*` no ambiente encerra antes
    de compor qualquer coisa (R26).
    """
    import os
    import sys

    legacy = legacy_environment(os.environ)
    if legacy is not None:
        from axon_scouting.rag.ingestion.adapters.cli import FAILED

        print(f"erro: {legacy}", file=sys.stderr)
        sys.exit(FAILED)
    from collections.abc import AsyncIterator
    from contextlib import asynccontextmanager

    from axon_scouting.rag.index.adapters.artifact import ArtifactIndex
    from axon_scouting.rag.index.adapters.bm25 import Bm25Encoder
    from axon_scouting.rag.index.adapters.e5 import E5Embedder, fetch
    from axon_scouting.rag.index.adapters.openrouter import OpenRouterEmbedder
    from axon_scouting.rag.ingestion.adapters import cli
    from axon_scouting.rag.ingestion.adapters.catalog_yaml import load_catalog
    from axon_scouting.rag.ingestion.adapters.clock import SystemClock
    from axon_scouting.rag.ingestion.adapters.pdfium import PdfiumParser
    from axon_scouting.rag.ingestion.service import IngestionService, copy_index

    qdrant_key = os.environ.get(API_KEY_VARIABLE) or None
    embedding_key = os.environ.get(EMBEDDING_KEY_VARIABLE) or None

    def model_dir(needed: bool) -> Path:
        directory = os.environ.get("AXON_SCOUTING_EMBEDDING_DIR")
        if not directory:
            if needed:
                raise cli.ConfigurationError(
                    "AXON_SCOUTING_EMBEDDING_DIR ausente: rode `axon-scouting-corpus model fetch <dir>` e aponte a "
                    "variável para esse diretório"
                )
            return Path()
        path = Path(directory).expanduser()
        missing = [name for name in MODEL_FILES if not (path / name).is_file()]
        if needed and missing:
            raise cli.ConfigurationError(
                f"{path} sem {', '.join(missing)}: rode `axon-scouting-corpus model fetch {path}`"
            )
        return path

    def embedder_for(options: cli.ServiceOptions) -> Embedder:
        if options.embedder == "pplx":
            if options.needs_model and embedding_key is None:
                raise cli.ConfigurationError(
                    f"{EMBEDDING_KEY_VARIABLE} ausente: o embedder pplx codifica pelo OpenRouter"
                )
            return OpenRouterEmbedder(embedding_key)
        return E5Embedder(model_dir(options.needs_model))

    def artifact(options: cli.ServiceOptions, embedder: Embedder) -> ArtifactIndex:
        root = options.directory.expanduser() if options.directory is not None else PACKAGED_INDEX
        return ArtifactIndex(root, embedder, Bm25Encoder())

    @asynccontextmanager
    async def open_service(options: cli.ServiceOptions) -> AsyncIterator[IngestionService]:
        embedder = embedder_for(options)
        index = artifact(options, embedder)
        yield IngestionService(
            parser=PdfiumParser(),
            embedder=embedder,
            sparse=Bm25Encoder(),
            writer=index,
            store=index,
            clock=SystemClock(),
        )

    async def load_qdrant(options: cli.ServiceOptions) -> str:
        from qdrant_client import AsyncQdrantClient

        from axon_scouting.rag.index.adapters.qdrant import QdrantIndex

        embedder = embedder_for(options)
        source = artifact(options, embedder)
        client = AsyncQdrantClient(
            url=os.environ.get("AXON_SCOUTING_QDRANT_URL", DEFAULT_QDRANT_URL), api_key=qdrant_key
        )
        try:
            target = QdrantIndex(client, os.environ.get("AXON_SCOUTING_INDEX", DEFAULT_INDEX), embedder, Bm25Encoder())
            return (await copy_index(source, target)).name
        finally:
            await client.close()

    composition = cli.Composition(
        index_name=DEFAULT_INDEX,
        open_service=open_service,
        load_qdrant=load_qdrant,
        load_catalog=load_catalog,
        fetch_model=fetch,
        secrets=tuple(secret for secret in (qdrant_key, embedding_key) if secret),
    )
    sys.exit(cli.main(sys.argv[1:], composition))
