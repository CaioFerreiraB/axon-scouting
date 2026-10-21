"""Identidade, manifest e capacidades do pack scouting (spec 101 FR-003, FR-010, FR-012; research R19, R20; T034; e as
métricas de domínio da 105, T010).

Só `axon_ai_contracts`, o kit de testes e o próprio pack — nada do core (SC-003). Cada carga usa um `qdrant_path` novo:
o Qdrant embutido trava o diretório por cliente aberto.
"""

from __future__ import annotations

from importlib.metadata import version
from pathlib import Path

import pytest
from axon_ai_contracts import NamedRetriever, PackManifest
from axon_ai_contracts.testing import pack_context

from axon_scouting.pack import SCOUTING_CONFIG_SCHEMA, ScoutingPack
from axon_scouting.rag.retrieval.reranker import MetadataReranker


def config(tmp_path: Path, **extra: object) -> dict[str, object]:
    return {"qdrant_path": str(tmp_path / "qdrant"), "embedder": "hashing", **extra}


def test_identity_is_readable_on_the_class() -> None:
    """A 002 lê `name` e `contract_version` da classe, antes de instanciar."""
    assert ScoutingPack.name == "scouting"
    assert ScoutingPack.contract_version == "0.2"


def test_manifest_da_fase_5() -> None:
    assert ScoutingPack().manifest() == PackManifest(
        name="scouting",
        version="0.4.0",
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
    (retriever,) = pack.retrievers(ctx)
    assert isinstance(retriever, NamedRetriever)
    assert retriever.name == "search_docs" and retriever.collections == frozenset({"scouting"})
    assert isinstance(pack.reranker(ctx), MetadataReranker)
    assert [validator.name for validator in pack.validators(ctx)] == ["citation_integrity", "validity_notice"]
    metrics = pack.eval_metrics(ctx)
    assert [metric.name for metric in metrics] == ["context_accuracy", "current_document_rate", "validity_notice_rate"]
    assert [metric.deterministic for metric in metrics] == [True, True, False]


@pytest.mark.parametrize(
    ("extra", "campo"),
    [({"embedder": "e5"}, "embedding_dir"), ({"qdrant_url": "https://q.example"}, "qdrant_url")],
)
def test_configuracao_invalida_impede_a_carga(tmp_path: Path, extra: dict[str, object], campo: str) -> None:
    with pytest.raises(ValueError, match=campo):
        ScoutingPack().retrievers(pack_context("scouting", config(tmp_path, **extra)))


def test_servidor_por_url_com_a_chave_do_ambiente_nao_toca_a_rede(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AXON_SCOUTING_QDRANT_API_KEY", "chave-de-teste")
    ctx = pack_context("scouting", {"qdrant_url": "https://qdrant.invalid:6333", "embedder": "hashing"})
    (retriever,) = ScoutingPack().retrievers(ctx)
    assert retriever.name == "search_docs"
    assert "chave-de-teste" not in repr(retriever)
