"""Semântica de `IndexFilters` (spec 102 research R5), como predicado puro.

O adapter em memória a avalia diretamente; o adapter Qdrant a traduz para um `Filter`, e a suíte de conformidade prova
que os dois concordam. Só datas e o vínculo de sucessão decidem a vigência — vale para qualquer data de referência.
"""

from __future__ import annotations

from axon_scouting.rag.index.api import CatalogEntry, IndexedChunk, IndexFilters, ValidityStatus


def matches(filters: IndexFilters, chunk: IndexedChunk | CatalogEntry) -> bool:
    """O trecho — ou o documento do catálogo, que tem os mesmos campos de vigência (ADR-047) — passa nos filtros."""
    if chunk.status is ValidityStatus.RASCUNHO:
        return False
    if filters.branch is not None:
        branches = chunk.effective_branches if isinstance(chunk, IndexedChunk) else _document_branches(chunk)
        if branches and filters.branch not in branches:
            return False
    if isinstance(chunk, IndexedChunk) and chunk.shape != filters.shape and (filters.shape or chunk.shape == "outline"):
        return False  # o sumário (`outline`) fica fora da busca comum
    if filters.document_types and chunk.document_type not in filters.document_types:
        return False
    if filters.bodies and not set(chunk.bodies) & set(filters.bodies):
        return False
    if filters.scopes and chunk.scope not in filters.scopes:
        return False
    until = filters.effective_until
    if filters.previous_version:
        return (
            chunk.superseded_by is not None
            and chunk.expiry_date is not None
            and chunk.expiry_date < until
            and (chunk.successor_expiry_date is None or chunk.successor_expiry_date >= until)
        )
    if chunk.effective_date > until:
        return False
    if chunk.expiry_date is not None:
        if chunk.expiry_date < filters.effective_from:
            return False
        if not filters.include_superseded and chunk.expiry_date < until:
            return False
    return True


def _document_branches(entry: CatalogEntry) -> tuple[str, ...]:
    return (entry.branch,) if entry.branch else ()
