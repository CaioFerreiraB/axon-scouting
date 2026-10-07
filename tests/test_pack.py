"""Identidade, manifest e catálogo do pack scouting na Fase 4 (spec 101, T009).

Só `ai_core_contracts` e o kit de testes — nada do core (SC-003).
"""

from __future__ import annotations

from importlib.metadata import version

from ai_core_contracts import PackManifest
from ai_core_contracts.testing import pack_context

from ai_scouting.pack import ScoutingPack


def test_identity_is_readable_on_the_class() -> None:
    """A 002 lê `name` e `contract_version` da classe, antes de instanciar."""
    assert ScoutingPack.name == "scouting"
    assert ScoutingPack.contract_version == "0.1"


def test_manifest_is_minimal_in_phase_4() -> None:
    manifest = ScoutingPack().manifest()
    assert manifest == PackManifest(name="scouting", version="0.1.0")
    assert manifest.preferred_flow is None  # clarificação 1: rag_first na Fase 5, com a US5
    assert manifest.requires_collections == ()
    assert manifest.depends_on == ()
    assert manifest.config_schema is None


def test_manifest_version_matches_the_distribution() -> None:
    assert ScoutingPack().manifest().version == version("ai-scouting")


def test_system_prompt_is_stable() -> None:
    pack = ScoutingPack()
    prompt = pack.system_prompt(pack_context("scouting"))
    assert prompt is not None and prompt.strip()
    assert prompt == pack.system_prompt(pack_context("scouting", {"qualquer": "coisa"}))


def test_catalog_is_empty_in_phase_4() -> None:
    pack = ScoutingPack()
    ctx = pack_context("scouting")
    assert tuple(pack.tools(ctx)) == ()
    assert tuple(pack.retrievers(ctx)) == ()
    assert pack.reranker(ctx) is None
    assert tuple(pack.validators(ctx)) == ()
