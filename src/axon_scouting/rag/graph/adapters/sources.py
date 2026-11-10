"""`ArtifactSources`: o que a construção offline do grafo lê (spec 106, research R13) — os trechos, o catálogo e o
manifesto do artefato da 102 e os YAML curados do `corpus/`. Só offline; o `yaml` fica aqui."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml


class ArtifactSources:
    def __init__(self, index: Path, organization: Path) -> None:
        self._index = index
        self._organization = organization

    def chunks(self) -> Iterable[Mapping[str, Any]]:
        with (self._index / "chunks.jsonl").open(encoding="utf-8") as lines:
            return [json.loads(line) for line in lines if line.strip()]

    def documents(self) -> Iterable[Mapping[str, Any]]:
        with (self._index / "documents.jsonl").open(encoding="utf-8") as lines:
            return [json.loads(line) for line in lines if line.strip()]

    def manifest(self) -> Mapping[str, Any]:
        loaded: Mapping[str, Any] = json.loads((self._index / "manifest.json").read_text(encoding="utf-8"))
        return loaded

    def organization(self) -> Mapping[str, Any]:
        loaded = yaml.safe_load(self._organization.read_text(encoding="utf-8"))
        if not isinstance(loaded, Mapping):
            raise ValueError(f"{self._organization}: precisa ser um mapeamento")
        return loaded

    def organization_sha256(self) -> str:
        return hashlib.sha256(self._organization.read_bytes()).hexdigest()
