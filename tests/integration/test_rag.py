"""US5 cen. 1 ponta a ponta na plataforma, pelo perfil `test` e sem rede (spec 101 SC-004, FR-011; research R23; T036).

Grupo opcional `platform` (core `v0.3.0`; o `rag_first` e o ADR-026 vêm da `v0.2.0`). O corpus sintético da 103 é
publicado num artefato do índice (ADR-043: o índice de produção, em memória) com o embedder de teste; o pack é
configurado pelo `packs.yaml` e o Assistant `scouting`
não declara `flow` — roda no preferido, `rag_first`. A geração é um provider escrito aqui, só com tipos do contrato 001,
trocado na montagem pela porta `providers` da 011: ele cita o primeiro identificador que vir depois de `id:` no
contexto.
"""

from __future__ import annotations

import asyncio
import os
import re
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("axon_ai", reason="axon-ai não instalado: rode com o grupo platform (Python 3.14)")

from axon_ai.bootstrap.profiles import TEST_OIDC_TOKEN
from axon_ai.bootstrap.wiring import platform
from axon_ai_contracts import ChatChunk, ChatMessage, ChatResponse, JSONValue
from fastapi.testclient import TestClient
from rag.retrieval.corpus import publicar

from axon_scouting.rag.index.adapters.artifact import ArtifactIndex
from axon_scouting.rag.index.adapters.bm25 import Bm25Encoder
from axon_scouting.rag.index.adapters.memory import HashingEmbedder

ID = re.compile(r"id: ([^\]\s]+)\]")
SEM_TRECHOS = "Nenhum documento oficial foi encontrado para a pergunta."

ASSISTANTS = """
assistants:
  scouting:
    visibility: public
    auth:
      schemes: [oidc, api_key]
      allow: ["*"]
    packs: [scouting]
    collections: [scouting]
    models:
      generation: ollama/qwen3:1.7b
      allow: ["ollama/*"]
api_keys:
  admin_group: admin
"""


class CitingProvider:
    """`LLMProvider` (001) da geração: cita o primeiro `id:` da mensagem `tool`; sem trecho, responde sem citar."""

    name = "ollama"

    def __init__(self) -> None:
        self.calls: list[list[ChatMessage]] = []

    def _answer(self, messages: Sequence[ChatMessage]) -> str:
        for message in messages:
            if message["role"] == "tool":
                found = ID.search(message.get("content") or "")
                if found:
                    return f"Conforme o documento [[{found.group(1)}]]."
        return SEM_TRECHOS

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        model: str,
        *,
        tools: Any = None,
        response_format: Mapping[str, JSONValue] | None = None,
        stream: bool = False,
    ) -> ChatResponse | AsyncIterator[ChatChunk]:
        self.calls.append(list(messages))
        answer = self._answer(messages)
        if not stream:
            return {"message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}

        async def chunks() -> AsyncIterator[ChatChunk]:
            yield {"delta": answer}
            yield {"delta": "", "finish_reason": "stop"}

        return chunks()

    async def probe(self, model: str) -> Any:
        return SimpleNamespace(available=True, detail=None)


def _publish(path: Path) -> None:
    embedder, sparse = HashingEmbedder(), Bm25Encoder()
    asyncio.run(publicar(ArtifactIndex(path, embedder, sparse), embedder, sparse))


@pytest.fixture
def montar(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    for name in [name for name in os.environ if name.startswith(("AXON_", "AI_CORE_", "AI_SCOUTING_"))]:
        monkeypatch.delenv(name)
    index = tmp_path / "index"
    _publish(index)
    packs = tmp_path / "packs.yaml"
    packs.write_text(
        f"packs:\n  scouting:\n    enabled: true\n    config:\n      index_dir: {index}\n      embedder: hashing\n",
        encoding="utf-8",
    )
    assistants = tmp_path / "assistants.yaml"
    assistants.write_text(ASSISTANTS, encoding="utf-8")
    monkeypatch.setenv("AXON_PACKS_FILE", str(packs))
    monkeypatch.setenv("AXON_ASSISTANTS_FILE", str(assistants))
    montadas: list[Any] = []

    def montar(**env: str) -> tuple[Any, CitingProvider]:
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        geracao = CitingProvider()
        plataforma = platform(profile="test", overrides={"providers": {"ollama": geracao}})
        assert plataforma.ok, plataforma.report.render()
        montadas.append(plataforma)
        return plataforma, geracao

    yield montar
    for plataforma in montadas:
        plataforma.aclose()


def _ask(plataforma: Any, question: str) -> dict[str, Any]:
    with TestClient(plataforma.app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {TEST_OIDC_TOKEN}"},
            json={"model": "scouting", "messages": [{"role": "user", "content": question}]},
        )
    assert response.status_code == 200, response.text
    return response.json()


def test_pergunta_coberta_responde_com_uma_chamada_e_citacao_resolvivel(montar: Any) -> None:
    plataforma, geracao = montar()
    payload = _ask(plataforma, "O que diz a Regra 33 do POR sobre o nome do grupo?")

    (mensagens,) = geracao.calls
    (resultado,) = [m["content"] for m in mensagens if m["role"] == "tool"]
    citado = ID.search(resultado or "")
    assert citado is not None, "a geração vê `· id: <chunk_id>` em cada trecho (ADR-026)"

    texto = payload["choices"][0]["message"]["content"]
    assert "[1]" in texto and "[[" not in texto
    (citacao,) = payload["x_axon"]["citations"]
    assert citacao["chunk_id"] == citado.group(1)
    assert citacao["document"] and citacao["page"]

    tipos = [r.type for r in plataforma.trace.trace(payload["x_axon"]["request_id"]).records]
    assert tipos.count("model_call") == 1, "SC-004: uma chamada de modelo no caminho coberto"
    gates = [r for r in plataforma.trace.trace(payload["x_axon"]["request_id"]).records if r.type == "coverage_gate"]
    assert [g.attributes["route"] for g in gates] == ["generate"]


def test_sem_trechos_responde_sem_citar_e_marca_nao_fundamentada(montar: Any) -> None:
    plataforma, geracao = montar(AXON_RAG_MIN_SCORE="1")
    payload = _ask(plataforma, "Qual é a receita de bolo de cenoura?")

    (mensagens,) = geracao.calls
    assert [m for m in mensagens if m["role"] == "tool"] == []
    assert payload["choices"][0]["message"]["content"].startswith(SEM_TRECHOS)
    assert "ungrounded" in payload["x_axon"]["flags"]
    assert {e["kind"] for e in payload["x_axon"]["errors"]} == {"CoverageInsufficient", "ValidatorIssue"}
    assert payload["choices"][0]["finish_reason"] == "stop", "registros `notice`, nenhuma falha"
