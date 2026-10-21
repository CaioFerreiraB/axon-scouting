"""O formato do artefato do índice (spec 102; ADR-043): manifesto, vetores densos por modelo e a escolha do modelo.

Puro: o adapter `ArtifactIndex` lê e grava os arquivos; aqui só se decide o que vai neles. Os trechos e os documentos
usam o mesmo JSON do payload do Qdrant (`points.py`); os vetores densos ficam um arquivo por modelo, em float32
little-endian, uma linha por trecho na ordem de `chunks.jsonl`.
"""

from __future__ import annotations

import re
import sys
from array import array
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Final

from axon_scouting.rag.index.api import EmbeddingModel, IndexVersion

FORMAT: Final = 1
"""Versão do formato dos arquivos; um formato que este código não lê é recusado na carga."""

MANIFEST: Final = "manifest.json"
CHUNKS: Final = "chunks.jsonl"
DOCUMENTS: Final = "documents.jsonl"
VECTORS: Final = "vectors"

_SLUG: Final = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelVectors:
    """Um conjunto de vetores densos do artefato: o modelo, o arquivo e o sha256 dele."""

    model: EmbeddingModel
    file: str
    sha256: str


@dataclass(frozen=True, slots=True, kw_only=True)
class Manifest:
    """O que identifica um artefato: o build, a codificação comum a todos os modelos e os conjuntos de vetores."""

    index: str
    build: str
    chunking: str
    sparse: str
    built_at: datetime
    chunks_sha256: str
    documents_sha256: str
    models: tuple[ModelVectors, ...]

    def vectors_of(self, model: EmbeddingModel) -> ModelVectors | None:
        """O conjunto do modelo pedido (mesmo nome, impressão digital e dimensões), ou `None`."""
        return next((entry for entry in self.models if entry.model == model), None)

    def version_for(self, model: EmbeddingModel) -> IndexVersion:
        """A versão do índice para quem codifica com `model`. Sem vetores dele, a do primeiro conjunto — a busca então
        recusa a mistura (`IndexMismatch`), nomeando os dois lados."""
        entry = self.vectors_of(model)
        if entry is None:
            if not self.models:
                raise ValueError(f"artefato {self.build} sem nenhum conjunto de vetores")
            entry = self.models[0]
        return IndexVersion(chunking=self.chunking, embedding=entry.model, sparse=self.sparse, built_at=self.built_at)


def vectors_file(model: EmbeddingModel) -> str:
    """O caminho relativo do arquivo de vetores de um modelo: `vectors/<nome>-<12 da impressão digital>.f32`, em
    minúsculas e hífens — dois pesos com o mesmo nome não dividem arquivo."""
    return f"{VECTORS}/{_slug(model.name)}-{_slug(model.fingerprint)[:12]}.f32"


def _slug(text: str) -> str:
    return _SLUG.sub("-", text.lower()).strip("-")


def pack_vectors(vectors: Sequence[Sequence[float]], dimensions: int) -> bytes:
    """Os vetores, na ordem, em float32 little-endian. Dimensão errada é recusada."""
    values = array("f")
    for vector in vectors:
        if len(vector) != dimensions:
            raise ValueError(f"vetor com {len(vector)} dimensões; o modelo tem {dimensions}")
        values.extend(vector)
    if sys.byteorder != "little":  # pragma: no cover - as máquinas-alvo são little-endian
        values.byteswap()
    return values.tobytes()


def unpack_vectors(data: bytes, dimensions: int, count: int) -> list[tuple[float, ...]]:
    """O inverso de `pack_vectors`; o tamanho precisa bater com `count` trechos."""
    values = array("f")
    values.frombytes(data)
    if sys.byteorder != "little":  # pragma: no cover
        values.byteswap()
    if len(values) != dimensions * count:
        raise ValueError(f"arquivo de vetores com {len(values)} valores; esperado {count} vezes {dimensions}")
    return [tuple(values[row * dimensions : (row + 1) * dimensions]) for row in range(count)]


def manifest_to_json(manifest: Manifest) -> dict[str, Any]:
    return {
        "format": FORMAT,
        "index": manifest.index,
        "build": manifest.build,
        "chunking": manifest.chunking,
        "sparse": manifest.sparse,
        "built_at": manifest.built_at.isoformat(),
        "files": {CHUNKS: manifest.chunks_sha256, DOCUMENTS: manifest.documents_sha256},
        "models": [
            {
                "name": entry.model.name,
                "fingerprint": entry.model.fingerprint,
                "dimensions": entry.model.dimensions,
                "file": entry.file,
                "sha256": entry.sha256,
            }
            for entry in manifest.models
        ],
    }


def manifest_from_json(raw: Mapping[str, Any]) -> Manifest:
    if raw.get("format") != FORMAT:
        raise ValueError(f"formato de artefato {raw.get('format')!r} não suportado (este código lê o {FORMAT})")
    files = raw["files"]
    return Manifest(
        index=raw["index"],
        build=raw["build"],
        chunking=raw["chunking"],
        sparse=raw["sparse"],
        built_at=datetime.fromisoformat(raw["built_at"]),
        chunks_sha256=files[CHUNKS],
        documents_sha256=files[DOCUMENTS],
        models=tuple(
            ModelVectors(
                model=EmbeddingModel(
                    name=entry["name"], fingerprint=entry["fingerprint"], dimensions=int(entry["dimensions"])
                ),
                file=entry["file"],
                sha256=entry["sha256"],
            )
            for entry in raw["models"]
        ),
    )
