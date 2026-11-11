"""Adapters em memória do grafo de domínio, para as suítes (spec 106, research R13; constituição XIII)."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from axon_scouting.rag.graph.api import GraphUnavailable
from axon_scouting.rag.graph.ports import GraphRecord


class MemoryGraphStore:
    """`GraphStore` sobre registros em memória; com `chunking`, recusa o grafo de outro build como o do artefato."""

    def __init__(self, records: Sequence[GraphRecord] = (), *, chunking: str | None = None) -> None:
        self._records = list(records)
        self._chunking = chunking

    def records(self) -> Iterable[GraphRecord]:
        meta = next((r for r in self._records if r.get("kind") == "meta"), None)
        if meta is not None and self._chunking is not None and meta.get("chunking") != self._chunking:
            raise GraphUnavailable(
                f"o grafo é do chunking {meta.get('chunking')!r}, e o índice é {self._chunking!r}: rode "
                "`scripts/grafo.py build`"
            )
        return list(self._records)


class MemorySources:
    """`GraphSources` sobre mapeamentos em memória."""

    def __init__(
        self,
        *,
        chunks: Sequence[Mapping[str, Any]] = (),
        documents: Sequence[Mapping[str, Any]] = (),
        manifest: Mapping[str, Any] | None = None,
        organization: Mapping[str, Any] | None = None,
        organization_sha256: str = "0" * 64,
        domain: Mapping[str, Any] | None = None,
        minutes: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        self._domain = dict(domain or {})
        self._minutes = list(minutes)
        self._chunks = list(chunks)
        self._documents = list(documents)
        self._manifest = dict(manifest or {"chunking": "memoria"})
        self._organization = dict(organization or {})
        self._sha = organization_sha256

    def chunks(self) -> Iterable[Mapping[str, Any]]:
        return list(self._chunks)

    def minutes(self) -> Iterable[Mapping[str, Any]]:
        return list(self._minutes)

    def documents(self) -> Iterable[Mapping[str, Any]]:
        return list(self._documents)

    def manifest(self) -> Mapping[str, Any]:
        return dict(self._manifest)

    def organization(self) -> Mapping[str, Any]:
        return dict(self._organization)

    def organization_sha256(self) -> str:
        return self._sha

    def domain(self) -> Mapping[str, Any]:
        return dict(self._domain)

    def domain_sha256(self) -> str | None:
        return None
