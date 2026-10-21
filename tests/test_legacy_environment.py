"""Recusa dos nomes antigos (spec 101 US6, FR-014; research R26, R30; ADR-037, item 3).

O pack recusa qualquer `AI_SCOUTING_*` no ambiente, na carga e na CLI do corpus, nomeando a variável nova e nunca o
valor.
"""

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path

import pytest
from axon_ai_contracts.testing import pack_context

from axon_scouting.pack import ScoutingPack, corpus_cli, legacy_environment


def test_sem_variaveis_antigas_nao_ha_recusa() -> None:
    assert legacy_environment({}) is None
    assert legacy_environment({"AXON_SCOUTING_INDEX": "scouting", "PATH": "/usr/bin"}) is None


def test_uma_linha_por_variavel_antiga_em_ordem_sem_valor() -> None:
    message = legacy_environment(
        {"AI_SCOUTING_QDRANT_URL": "", "AI_SCOUTING_INDEX": "segredo-do-indice", "AI_SCOUTING_FOO": "x"}
    )
    assert message is not None
    assert message.splitlines() == [
        "variáveis com o nome antigo (ADR-037): renomeie",
        "  AI_SCOUTING_FOO → AXON_SCOUTING_FOO",
        "  AI_SCOUTING_INDEX → AXON_SCOUTING_INDEX",
        "  AI_SCOUTING_QDRANT_URL → AXON_SCOUTING_QDRANT_URL",
    ]
    assert "segredo-do-indice" not in message


def test_so_o_prefixo_exato() -> None:
    assert legacy_environment({"MY_AI_SCOUTING_X": "1", "AI_SCOUTINGX": "1"}) is None


def test_a_nova_definida_nao_libera_a_antiga() -> None:
    message = legacy_environment({"AI_SCOUTING_INDEX": "a", "AXON_SCOUTING_INDEX": "b"})
    assert message is not None and "AXON_SCOUTING_INDEX" in message


def test_a_carga_recusa(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AI_SCOUTING_INDEX", "scouting")
    ctx = pack_context("scouting", {"qdrant_path": str(tmp_path / "qdrant"), "embedder": "hashing"})
    with pytest.raises(ValueError, match="AXON_SCOUTING_INDEX"):
        ScoutingPack().retrievers(ctx)


def test_a_cli_do_corpus_recusa(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("AI_SCOUTING_QDRANT_API_KEY", "chave-antiga")
    monkeypatch.setattr("sys.argv", ["axon-scouting-corpus", "status"])
    with pytest.raises(SystemExit) as exit_info:
        corpus_cli()
    assert exit_info.value.code == 2
    err = capsys.readouterr().err
    assert err.startswith("erro: variáveis com o nome antigo (ADR-037)")
    assert "AI_SCOUTING_QDRANT_API_KEY → AXON_SCOUTING_QDRANT_API_KEY" in err
    assert "chave-antiga" not in err


def test_o_import_antigo_nao_existe() -> None:
    assert find_spec("ai_scouting") is None
