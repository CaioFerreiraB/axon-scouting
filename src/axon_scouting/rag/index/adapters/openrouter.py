"""`OpenRouterEmbedder`: embedding remoto pela API compatível com a OpenAI do OpenRouter (spec 102; ADR-043, 2 e 5).

O primeiro modelo é o `perplexity/pplx-embed-v1-0.6b` (pesos abertos, 1024 dimensões), sem prefixo de pergunta ou de
passagem. Os vetores voltam normalizados (L2), na ingestão e na pergunta, para o cosseno do índice valer igual ao do e5.
A chave é a do próprio pack (`AXON_SCOUTING_EMBEDDING_API_KEY`), lida na construção e nunca posta em mensagem; sem ela,
o embedder existe — o `packs check` do build não tem segredos — e a falta aparece na primeira codificação. Falha do
provedor é exceção: a recuperação falha, e o `rag_first` encerra sem gerar (005). Só biblioteca padrão; a chamada roda
em outra thread para não travar o laço do core.
"""

from __future__ import annotations

import asyncio
import json
import math
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from typing import IO, Final

from axon_scouting.rag.index.api import EmbeddingModel

PPLX: Final = "perplexity/pplx-embed-v1-0.6b"
PPLX_DIMENSIONS: Final = 1024
BASE_URL: Final = "https://openrouter.ai/api/v1"
BATCH: Final = 64
TIMEOUT: Final = 60.0

Opener = Callable[[urllib.request.Request, float], IO[bytes]]


class EmbeddingUnavailable(RuntimeError):
    """O embedder remoto não codificou: chave ausente, provedor fora ou resposta inesperada. Nunca traz a chave."""


def _open(request: urllib.request.Request, timeout: float) -> IO[bytes]:
    response: IO[bytes] = urllib.request.urlopen(request, timeout=timeout)
    return response


class OpenRouterEmbedder:
    def __init__(
        self,
        api_key: str | None,
        *,
        model: str = PPLX,
        dimensions: int = PPLX_DIMENSIONS,
        base_url: str = BASE_URL,
        opener: Opener = _open,
    ) -> None:
        self._key = api_key or None
        self._name = model
        self._url = f"{base_url.rstrip('/')}/embeddings"
        self._opener = opener
        # O modelo remoto não tem arquivo para hashear: a impressão digital é o id no provedor, as dimensões e a
        # normalização — o que precisa ser igual na ingestão e na pergunta.
        self._model = EmbeddingModel(
            name=f"openrouter/{model}", fingerprint=f"{model}:{dimensions}:l2", dimensions=dimensions
        )

    def model(self) -> EmbeddingModel:
        return self._model

    async def embed_passages(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        vectors: list[tuple[float, ...]] = []
        for start in range(0, len(texts), BATCH):
            vectors.extend(await asyncio.to_thread(self._encode, list(texts[start : start + BATCH])))
        return vectors

    async def embed_query(self, text: str) -> tuple[float, ...]:
        (vector,) = await asyncio.to_thread(self._encode, [text])
        return vector

    def _encode(self, texts: list[str]) -> list[tuple[float, ...]]:
        if self._key is None:
            raise EmbeddingUnavailable(
                f"embedder {self._name}: AXON_SCOUTING_EMBEDDING_API_KEY ausente — sem ela, o pack não codifica"
            )
        body = json.dumps({"model": self._name, "input": texts, "encoding_format": "float"}).encode()
        request = urllib.request.Request(
            self._url,
            data=body,
            method="POST",
            headers={"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"},
        )
        try:
            with self._opener(request, TIMEOUT) as response:
                payload = json.loads(response.read())
        except urllib.error.HTTPError as error:
            raise EmbeddingUnavailable(f"embedder {self._name}: o provedor respondeu {error.code}") from None
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise EmbeddingUnavailable(
                f"embedder {self._name}: provedor inacessível ({type(error).__name__})"
            ) from None
        return self._vectors(payload, len(texts))

    def _vectors(self, payload: object, count: int) -> list[tuple[float, ...]]:
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list) or len(data) != count:
            raise EmbeddingUnavailable(f"embedder {self._name}: resposta sem {count} vetores")
        rows = sorted(data, key=lambda item: int(item.get("index", 0)) if isinstance(item, dict) else 0)
        vectors: list[tuple[float, ...]] = []
        for row in rows:
            values = row.get("embedding") if isinstance(row, dict) else None
            if not isinstance(values, list) or len(values) != self._model.dimensions:
                raise EmbeddingUnavailable(
                    f"embedder {self._name}: vetor com dimensões diferentes de {self._model.dimensions}"
                )
            norm = math.sqrt(sum(float(x) * float(x) for x in values)) or 1.0
            vectors.append(tuple(float(x) / norm for x in values))
        return vectors
