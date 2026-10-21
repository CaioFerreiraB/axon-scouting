"""`E5Embedder`: `intfloat/multilingual-e5-small` em ONNX, no processo (spec 102 clarificação 2; research R10).

Provisório até a medição da Fase 6 (princípio X). Lê `model.onnx` e `tokenizer.json` de um diretório e nunca baixa
nada em runtime; `fetch` baixa a revisão fixada e confere os sha256. Configuração do modelo: truncamento em 512, *mean
pooling* com a máscara de atenção, normalização L2 e os prefixos `passage: `/`query: ` que o e5 exige. A inferência
roda em outra thread para não travar o laço do core.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import threading
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Final

# Antes do import: o onnxruntime não coleta telemetria nem tenta gravar um id de dispositivo — num sistema de arquivos
# somente leitura, isso viraria um aviso a cada processo. Era uma linha da imagem da instalação; desde o ADR-042 a
# instalação não conhece pack, e quem sabe que usa o onnxruntime é o pack.
os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from axon_scouting.rag.index.api import EmbeddingModel

NAME: Final = "intfloat/multilingual-e5-small"
REVISION: Final = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
BASE_URL: Final = f"https://huggingface.co/{NAME}/resolve/{REVISION}"
DIMENSIONS: Final = 384
MAX_TOKENS: Final = 512
BATCH: Final = 8
"""Lote de inferência: o pico de memória cresce com lote vezes tokens ao quadrado; 8 cabe folgado nos 12 GB do alvo."""


@dataclass(frozen=True, slots=True)
class ModelFile:
    remote: str
    local: str
    sha256: str


FILES: Final = (
    ModelFile("onnx/model.onnx", "model.onnx", "ca456c06b3a9505ddfd9131408916dd79290368331e7d76bb621f1cba6bc8665"),
    ModelFile(
        "onnx/tokenizer.json", "tokenizer.json", "0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39"
    ),
)


class E5Embedder:
    def __init__(self, model_dir: Path) -> None:
        self._dir = model_dir
        self._lock = threading.Lock()
        self._session: ort.InferenceSession | None = None
        self._tokenizer: Tokenizer | None = None
        self._model: EmbeddingModel | None = None

    def model(self) -> EmbeddingModel:
        if self._model is None:
            self._model = EmbeddingModel(
                name=NAME, fingerprint=_sha256(self._dir / "model.onnx"), dimensions=DIMENSIONS
            )
        return self._model

    async def embed_passages(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        if not texts:
            return []
        return await asyncio.to_thread(self._encode, [f"passage: {t}" for t in texts])

    async def embed_query(self, text: str) -> tuple[float, ...]:
        (vector,) = await asyncio.to_thread(self._encode, [f"query: {text}"])
        return vector

    def _load(self) -> tuple[ort.InferenceSession, Tokenizer]:
        with self._lock:
            if self._session is None or self._tokenizer is None:
                tokenizer = Tokenizer.from_file(str(self._dir / "tokenizer.json"))
                tokenizer.enable_truncation(max_length=MAX_TOKENS)
                pad = tokenizer.token_to_id("<pad>")
                tokenizer.enable_padding(pad_id=1 if pad is None else pad, pad_token="<pad>")
                self._tokenizer = tokenizer
                options = ort.SessionOptions()
                options.enable_cpu_mem_arena = False  # a arena reteria o pico do maior lote pelo resto do processo
                self._session = ort.InferenceSession(
                    str(self._dir / "model.onnx"), sess_options=options, providers=["CPUExecutionProvider"]
                )
            return self._session, self._tokenizer

    def _encode(self, texts: list[str]) -> list[tuple[float, ...]]:
        session, tokenizer = self._load()
        names = {i.name for i in session.get_inputs()}
        vectors: list[tuple[float, ...]] = []
        for start in range(0, len(texts), BATCH):
            encodings = tokenizer.encode_batch(texts[start : start + BATCH])
            ids = np.array([e.ids for e in encodings], dtype=np.int64)
            mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
            feeds = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in names:
                feeds["token_type_ids"] = np.zeros_like(ids)
            hidden = session.run(None, feeds)[0]
            weights = mask[..., None].astype(np.float32)
            pooled = (hidden * weights).sum(axis=1) / np.clip(weights.sum(axis=1), 1e-9, None)
            pooled /= np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12, None)
            vectors.extend(tuple(float(x) for x in row) for row in pooled)
        return vectors


def fetch(
    model_dir: Path,
    *,
    files: Sequence[ModelFile] = FILES,
    base_url: str = BASE_URL,
    opener: Callable[[str], IO[bytes]] = urllib.request.urlopen,
) -> list[tuple[Path, str]]:
    """Baixa os arquivos da revisão fixada, conferindo o sha256; arquivo presente e íntegro não é baixado de novo."""
    model_dir.mkdir(parents=True, exist_ok=True)
    fetched: list[tuple[Path, str]] = []
    for file in files:
        target = model_dir / file.local
        if not (target.exists() and _sha256(target) == file.sha256):
            partial = target.with_suffix(target.suffix + ".part")
            digest = hashlib.sha256()
            with opener(f"{base_url}/{file.remote}") as response, partial.open("wb") as out:
                while chunk := response.read(1 << 20):
                    digest.update(chunk)
                    out.write(chunk)
            if digest.hexdigest() != file.sha256:
                partial.unlink()
                raise ValueError(
                    f"{file.local}: sha256 {digest.hexdigest()} diferente do fixado {file.sha256} (revisão {REVISION})"
                )
            partial.replace(target)
        fetched.append((target, file.sha256))
    return fetched


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()
