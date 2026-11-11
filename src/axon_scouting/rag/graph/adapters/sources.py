"""`ArtifactSources`: o que a construção offline do grafo lê (spec 106, research R13) — os trechos, o catálogo e o
manifesto do artefato da 102, os YAML curados do `corpus/` e as fichas das atas. Só offline; o `yaml` fica aqui."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml


class ArtifactSources:
    def __init__(
        self, index: Path, organization: Path, domain: Path | None = None, minutes: Path | None = None
    ) -> None:
        self._index = index
        self._organization = organization
        self._domain = domain
        self._minutes = minutes

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

    def domain(self) -> Mapping[str, Any]:
        if self._domain is None or not self._domain.exists():
            return {}
        loaded = yaml.safe_load(self._domain.read_text(encoding="utf-8"))
        if not isinstance(loaded, Mapping):
            raise ValueError(f"{self._domain}: precisa ser um mapeamento")
        return loaded

    def domain_sha256(self) -> str | None:
        if self._domain is None or not self._domain.exists():
            return None
        return hashlib.sha256(self._domain.read_bytes()).hexdigest()

    def minutes(self) -> Iterable[Mapping[str, Any]]:
        if self._minutes is None or not self._minutes.is_dir():
            return []
        found: list[Mapping[str, Any]] = []
        for path in sorted(self._minutes.glob("*.yaml")):
            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
            if not isinstance(loaded, Mapping):
                raise ValueError(f"{path}: precisa ser um mapeamento")
            found.append(loaded)
        return found
