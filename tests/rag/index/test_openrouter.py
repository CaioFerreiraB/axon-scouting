"""`OpenRouterEmbedder` (ADR-043, itens 2 e 5), com transporte falso: pedido, normalização, ordem e falhas."""

from __future__ import annotations

import asyncio
import io
import json
import math
import urllib.error
import urllib.request
from typing import IO, Any

import pytest

from axon_scouting.rag.index.adapters.openrouter import PPLX, EmbeddingUnavailable, OpenRouterEmbedder

CHAVE = "sk-or-chave-sentinela"


class Transporte:
    def __init__(self, resposta: Any = None, erro: BaseException | None = None, dims: int = 4) -> None:
        self.pedidos: list[urllib.request.Request] = []
        self.resposta = resposta
        self.erro = erro
        self.dims = dims

    def __call__(self, request: urllib.request.Request, timeout: float) -> IO[bytes]:
        self.pedidos.append(request)
        if self.erro is not None:
            raise self.erro
        if self.resposta is not None:
            return io.BytesIO(json.dumps(self.resposta).encode())
        textos = json.loads(request.data)["input"]  # type: ignore[arg-type]
        # Índices na ordem inversa: o embedder precisa reordenar pelo `index`.
        data = [{"index": i, "embedding": [float(i + 1)] * self.dims} for i in range(len(textos))][::-1]
        return io.BytesIO(json.dumps({"data": data}).encode())


def embedder(transporte: Transporte, chave: str | None = CHAVE) -> OpenRouterEmbedder:
    return OpenRouterEmbedder(chave, dimensions=transporte.dims, opener=transporte)


def test_identidade_do_modelo() -> None:
    model = OpenRouterEmbedder(None).model()
    assert (model.name, model.fingerprint, model.dimensions) == (f"openrouter/{PPLX}", f"{PPLX}:1024:l2", 1024)


def test_pedido_e_normalizacao() -> None:
    transporte = Transporte()
    vetores = asyncio.run(embedder(transporte).embed_passages(["a", "b"]))
    (pedido,) = transporte.pedidos
    assert pedido.full_url == "https://openrouter.ai/api/v1/embeddings"
    assert pedido.get_header("Authorization") == f"Bearer {CHAVE}"
    assert json.loads(pedido.data) == {"model": PPLX, "input": ["a", "b"], "encoding_format": "float"}  # type: ignore[arg-type]
    assert len(vetores) == 2 and all(math.isclose(sum(x * x for x in v), 1.0) for v in vetores)
    assert vetores[0] == vetores[1], "ordenados pelo index e normalizados, os dois apontam para o mesmo lado"


def test_pergunta_sem_prefixo() -> None:
    transporte = Transporte()
    asyncio.run(embedder(transporte).embed_query("o que é a promessa?"))
    assert json.loads(transporte.pedidos[0].data)["input"] == ["o que é a promessa?"]  # type: ignore[arg-type]


def test_lotes() -> None:
    transporte = Transporte()
    vetores = asyncio.run(embedder(transporte).embed_passages([str(i) for i in range(130)]))
    assert len(vetores) == 130 and len(transporte.pedidos) == 3


def test_sem_chave_falha_na_codificacao_e_nao_na_construcao() -> None:
    transporte = Transporte()
    sem_chave = embedder(transporte, chave=None)
    with pytest.raises(EmbeddingUnavailable, match="AXON_SCOUTING_EMBEDDING_API_KEY"):
        asyncio.run(sem_chave.embed_query("x"))
    assert transporte.pedidos == []


@pytest.mark.parametrize(
    ("transporte", "mensagem"),
    [
        (Transporte(erro=urllib.error.HTTPError("u", 401, "não", {}, None)), "respondeu 401"),  # type: ignore[arg-type]
        (Transporte(erro=urllib.error.URLError("dns")), "inacessível"),
        (Transporte(resposta={"data": []}), "sem 1 vetores"),
        (Transporte(resposta={"data": [{"index": 0, "embedding": [1.0]}]}), "dimensões"),
    ],
)
def test_falhas_nunca_mostram_a_chave(transporte: Transporte, mensagem: str) -> None:
    with pytest.raises(EmbeddingUnavailable, match=mensagem) as erro:
        asyncio.run(embedder(transporte).embed_query("x"))
    assert CHAVE not in str(erro.value)
