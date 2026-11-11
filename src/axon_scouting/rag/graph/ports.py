"""Portas secundárias do grafo de domínio (spec 106, research R13; design.md, seção 7)."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Protocol, runtime_checkable

GraphRecord = Mapping[str, Any]


@runtime_checkable
class GraphStore(Protocol):
    """De onde vêm os registros do grafo na pergunta. Arquivo ausente → nenhum registro; grafo de outro build →
    `GraphUnavailable`."""

    def records(self) -> Iterable[GraphRecord]: ...


@runtime_checkable
class GraphSources(Protocol):
    """O que a construção offline lê: os trechos do artefato com o caminho na árvore, o catálogo de versões, o manifesto
    do índice e os YAML curados do `corpus/`."""

    def chunks(self) -> Iterable[Mapping[str, Any]]:
        """Os registros de `chunks.jsonl`: `chunk_id`, `doc_key`, `document_type`, `document_title`, `scope`, `status`,
        `path`…"""
        ...

    def documents(self) -> Iterable[Mapping[str, Any]]:
        """Os registros de `documents.jsonl`: `doc_key` e `fields` (o catálogo da 102)."""
        ...

    def manifest(self) -> Mapping[str, Any]: ...

    def organization(self) -> Mapping[str, Any]:
        """`corpus/organizacao.yaml`, já lido."""
        ...

    def organization_sha256(self) -> str: ...

    def domain(self) -> Mapping[str, Any]:
        """`corpus/dominio.yaml`, já lido (G2); ausente, `{}`."""
        ...

    def domain_sha256(self) -> str | None: ...
