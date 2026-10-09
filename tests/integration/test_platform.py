"""US1 cen. 1 e 2 na plataforma real, pelo perfil `test` (spec 101, research R9; T012).

Grupo opcional `platform`: pulado quando o `ai-core` não está instalado — a suíte padrão roda só com os contratos
(SC-003). Usa apenas a superfície que a 011 publica para suítes de packs externos: a configuração pelas variáveis do
operador (`AI_CORE_*`), a montagem por `platform(profile="test")` e as credenciais exportadas do perfil. Nenhuma porta é
trocada e nenhum módulo interno ou adapter do core é importado (princípio II).
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("ai_core", reason="ai-core não instalado: rode com o grupo platform (Python 3.14)")

from ai_core.bootstrap.profiles import TEST_OIDC_TOKEN
from ai_core.bootstrap.wiring import platform
from ai_core_contracts.testing import pack_context
from fastapi.testclient import TestClient

from ai_scouting.pack import ScoutingPack

PACKS = """
packs:
  scouting:
    enabled: true
    config:
      qdrant_path: {qdrant}
      embedder: hashing
"""
"""Na Fase 5 a carga precisa de configuração (T037): o default `e5` exige o diretório do modelo."""

ASSISTANTS = """
assistants:
  scouting:
    visibility: public
    auth:
      schemes: [oidc, api_key]
      allow: ["*"]
    flow: chat
    packs: [scouting]
    models:
      generation: ollama/qwen3:1.7b
      allow: ["ollama/*"]
api_keys:
  admin_group: admin
"""


@pytest.fixture
def plataforma(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    for name in [name for name in os.environ if name.startswith("AI_CORE_")]:
        monkeypatch.delenv(name)
    packs = tmp_path / "packs.yaml"
    assistants = tmp_path / "assistants.yaml"
    packs.write_text(PACKS.format(qdrant=tmp_path / "qdrant"), encoding="utf-8")
    assistants.write_text(ASSISTANTS, encoding="utf-8")
    monkeypatch.setenv("AI_CORE_PACKS_FILE", str(packs))
    monkeypatch.setenv("AI_CORE_ASSISTANTS_FILE", str(assistants))
    montada = platform(profile="test")
    assert montada.ok, montada.report.render()
    yield montada
    montada.aclose()


def test_pack_carrega_pelo_entry_point(plataforma: Any) -> None:
    entry = plataforma.packs.entry("scouting")
    assert entry is not None
    assert entry.status == "enabled", entry.error
    assert entry.version == "0.2.0"
    assert entry.contract_version == "0.1"


def test_plataforma_compoe_as_instrucoes_de_dominio(plataforma: Any) -> None:
    prompts = plataforma.packs.system_prompts(plataforma.packs.compose(["scouting"]))
    assert ScoutingPack().system_prompt(pack_context("scouting")) in prompts

    with TestClient(plataforma.app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {TEST_OIDC_TOKEN}"},
            json={"model": "scouting", "messages": [{"role": "user", "content": "O que é a promessa escoteira?"}]},
        )

    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["message"]["content"]


@pytest.mark.parametrize(
    ("config", "esperado"),
    [("{embedder: e5}", "embedding_dir"), ("{embedder: hashing, api_key: segredo}", "api_key")],
    ids=["e5-sem-diretorio", "chave-no-arquivo"],
)
def test_configuracao_invalida_deixa_o_pack_em_erro(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config: str, esperado: str
) -> None:
    """Edge Case "configuração inválida" e FR-012: o pack não carrega, e o erro aparece na introspecção."""
    for name in [name for name in os.environ if name.startswith("AI_CORE_")]:
        monkeypatch.delenv(name)
    packs = tmp_path / "packs.yaml"
    packs.write_text(f"packs:\n  scouting:\n    enabled: true\n    config: {config}\n", encoding="utf-8")
    monkeypatch.setenv("AI_CORE_PACKS_FILE", str(packs))
    montada = platform(profile="test")
    try:
        entry = montada.packs.entry("scouting")
        assert entry is not None and entry.status == "error"
        assert esperado in (entry.error or "")
        assert "segredo" not in (entry.error or "")
    finally:
        montada.aclose()
