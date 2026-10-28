"""Identidade, manifest e capacidades do pack scouting (spec 101 FR-003, FR-010, FR-012; research R19, R20; T034; e as
métricas de domínio da 105, T010).

Só `axon_ai_contracts`, o kit de testes e o próprio pack — nada do core (SC-003). Cada carga no Qdrant usa um
`qdrant_path` novo: o Qdrant embutido trava o diretório por cliente aberto. O índice em memória (ADR-043) carrega um
artefato montado no `tmp_path`, e o `prepare` (ADR-041) roda com o download trocado por um falso.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import pytest
from axon_ai_contracts import NamedRetriever, PackManifest
from axon_ai_contracts.eval import EvalChunk, EvalItem, EvalObservation
from axon_ai_contracts.testing import pack_context, turn_view
from rag.index.kit import campos, codificar, registro, trecho, versao

from axon_scouting.pack import SCOUTING_CONFIG_SCHEMA, ScoutingPack
from axon_scouting.rag.index.adapters import e5
from axon_scouting.rag.index.adapters.artifact import ArtifactIndex
from axon_scouting.rag.index.adapters.bm25 import Bm25Encoder
from axon_scouting.rag.index.adapters.memory import HashingEmbedder
from axon_scouting.rag.retrieval.reranker import MetadataReranker


def config(tmp_path: Path, **extra: object) -> dict[str, object]:
    return {"store": "qdrant", "qdrant_path": str(tmp_path / "qdrant"), "embedder": "hashing", **extra}


def artefato(root: Path) -> Path:
    """Um artefato pequeno, com vetores do embedder de teste."""

    async def publicar() -> None:
        fields = campos()
        chunks = [trecho(fields, "4.1", 1, "[Manual X · 4.1]\nO acampamento de patrulha exige planejamento.")]
        index = ArtifactIndex(root, HashingEmbedder(), Bm25Encoder())
        build = await index.create(versao())
        await index.write(
            build.name, registro(fields, chunks, ("p",)), await codificar(chunks, HashingEmbedder(), Bm25Encoder())
        )
        await index.publish(build.name)

    asyncio.run(publicar())
    return root


def test_identity_is_readable_on_the_class() -> None:
    """A 002 lê `name` e `contract_version` da classe, antes de instanciar."""
    assert ScoutingPack.name == "scouting"
    assert ScoutingPack.contract_version == "0.2.1"


def test_manifest_da_fase_5() -> None:
    assert ScoutingPack().manifest() == PackManifest(
        name="scouting",
        version="0.6.0",
        preferred_flow="rag_first",
        requires_collections=("scouting",),
        config_schema=SCOUTING_CONFIG_SCHEMA,
    )


def test_manifest_version_matches_the_distribution() -> None:
    assert ScoutingPack().manifest().version == version("axon-scouting")


def test_system_prompt_is_stable() -> None:
    pack = ScoutingPack()
    prompt = pack.system_prompt(pack_context("scouting"))
    assert prompt is not None and prompt.strip()
    assert prompt == pack.system_prompt(pack_context("scouting", {"embedder": "hashing"}))


def test_capacidades_publicadas(tmp_path: Path) -> None:
    pack = ScoutingPack()
    ctx = pack_context("scouting", config(tmp_path))
    assert tuple(pack.tools(ctx)) == ()
    retriever, lista, unidades = pack.retrievers(ctx)
    assert isinstance(retriever, NamedRetriever) and isinstance(lista, NamedRetriever)
    assert retriever.name == "search_docs" and retriever.collections == frozenset({"scouting"})
    assert lista.name == "list_documents" and lista.collections == frozenset({"scouting"})
    assert unidades.name == "list_units" and unidades.collections == frozenset({"scouting"})
    assert isinstance(pack.reranker(ctx), MetadataReranker)
    assert [validator.name for validator in pack.validators(ctx)] == ["citation_integrity", "validity_notice"]
    metrics = pack.eval_metrics(ctx)
    assert [metric.name for metric in metrics] == [
        "context_accuracy",
        "current_document_rate",
        "validity_notice_rate",
        "absence_claim_rate",
    ]
    assert [metric.deterministic for metric in metrics] == [True, True, False, False]


@pytest.mark.parametrize(
    ("extra", "campo"),
    [({"embedder": "e5"}, "embedding_dir"), ({"qdrant_url": "https://q.example"}, "qdrant_url")],
)
def test_configuracao_invalida_impede_a_carga(tmp_path: Path, extra: dict[str, object], campo: str) -> None:
    with pytest.raises(ValueError, match=campo):
        ScoutingPack().retrievers(pack_context("scouting", config(tmp_path, **extra)))


def test_servidor_por_url_com_a_chave_do_ambiente_nao_toca_a_rede(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AXON_SCOUTING_QDRANT_API_KEY", "chave-de-teste")
    ctx = pack_context(
        "scouting", {"store": "qdrant", "qdrant_url": "https://qdrant.invalid:6333", "embedder": "hashing"}
    )
    retriever, *_ = ScoutingPack().retrievers(ctx)
    assert retriever.name == "search_docs"
    assert "chave-de-teste" not in repr(retriever)


def test_indice_em_memoria_pelo_artefato(tmp_path: Path) -> None:
    """ADR-043, item 4: o artefato carregado no processo, sem serviço; a busca usa o mesmo embedder dos vetores."""
    root = artefato(tmp_path / "index")
    pack = ScoutingPack()
    ctx = pack_context("scouting", {"embedder": "hashing", "index_dir": str(root)})
    retriever, *_ = pack.retrievers(ctx)
    pergunta = "acampamento de patrulha"
    depois_da_vigencia = datetime(2026, 10, 7, 12, tzinfo=UTC)
    fatia = asyncio.run(pack.prepare_context(turn_view(question=pergunta, reference_time=depois_da_vigencia), ctx))
    filtros = fatia["retrieval_filters"]
    assert isinstance(filtros, dict)
    resultado = asyncio.run(retriever.retriever.search(pergunta, top_k=3, filters=filtros))
    assert [chunk.document for chunk in resultado] == ["Manual X"]


def test_metrica_de_fatos_le_o_texto_do_artefato(tmp_path: Path) -> None:
    """105 FR-014: com o índice em memória, a `fact_recall` acha o fato no texto do trecho entregue, pelo id."""
    root = artefato(tmp_path / "index")
    metrics = ScoutingPack().eval_metrics(pack_context("scouting", {"embedder": "hashing", "index_dir": str(root)}))
    assert [metric.name for metric in metrics][-1] == "fact_recall"
    item = EvalItem(
        id="f1",
        question="acampamento",
        reference_time=datetime(2026, 10, 7, 12, tzinfo=UTC),
        extra={"facts": ["acampamento de patrulha exige planejamento", "fato que não está no trecho"]},
    )
    entregue = EvalChunk(chunk_id="manual-x-2026#4.1@p1", score=0.9, document="Manual X")
    assert metrics[-1].measure(item, EvalObservation(answer="", delivered=[entregue])) == 0.5


def test_artefato_ausente_impede_a_carga(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"manifest\.json"):
        ScoutingPack().retrievers(pack_context("scouting", {"embedder": "hashing", "index_dir": str(tmp_path)}))


def test_pplx_carrega_sem_a_chave(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """O `packs check` do build não tem segredos: a falta da chave aparece na busca, não na carga."""
    monkeypatch.delenv("AXON_SCOUTING_EMBEDDING_API_KEY", raising=False)
    root = artefato(tmp_path / "index")
    retriever, *_ = ScoutingPack().retrievers(pack_context("scouting", {"embedder": "pplx", "index_dir": str(root)}))
    assert retriever.name == "search_docs"


@pytest.fixture
def downloads(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    feitos: list[Path] = []
    monkeypatch.setattr(e5, "fetch", lambda directory: feitos.append(directory) or [])
    return feitos


def test_prepare_baixa_o_e5_no_data_dir(tmp_path: Path, downloads: list[Path]) -> None:
    """ADR-041: o e5 é o default, e vai para data_dir/e5-small."""
    ScoutingPack().prepare(pack_context("scouting", {}, data_dir=tmp_path))
    assert downloads == [tmp_path / "e5-small"]


def test_prepare_respeita_o_embedding_dir(tmp_path: Path, downloads: list[Path]) -> None:
    ScoutingPack().prepare(pack_context("scouting", {"embedding_dir": str(tmp_path / "modelo")}, data_dir=tmp_path))
    assert downloads == [tmp_path / "modelo"]


@pytest.mark.parametrize("embedder", ["pplx", "hashing"])
def test_prepare_sem_e5_nao_baixa_nada(tmp_path: Path, downloads: list[Path], embedder: str) -> None:
    """ADR-043, item 6: com pplx, a imagem não carrega modelo nenhum."""
    ScoutingPack().prepare(pack_context("scouting", {"embedder": embedder}, data_dir=tmp_path))
    assert downloads == []


def test_prepare_sem_data_dir_nem_diretorio(downloads: list[Path]) -> None:
    with pytest.raises(ValueError, match="AXON_PACKS_DATA_DIR"):
        ScoutingPack().prepare(pack_context("scouting", {}))
    assert downloads == []
