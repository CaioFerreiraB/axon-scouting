"""Semântica de `IndexFilters` (spec 102 research R5), como predicado puro.

O adapter em memória a avalia diretamente; o adapter Qdrant a traduz para um `Filter`, e a suíte de conformidade prova
que os dois concordam. Só datas e o vínculo de sucessão decidem a vigência — vale para qualquer data de referência.
"""

from __future__ import annotations

from axon_scouting.rag.index.api import IndexedChunk, IndexFilters, ValidityStatus


def matches(filters: IndexFilters, chunk: IndexedChunk) -> bool:
    if chunk.status is ValidityStatus.RASCUNHO:
        return False
    if filters.branch is not None and chunk.branch not in (None, filters.branch):
        return False
    if filters.document_types and chunk.document_type not in filters.document_types:
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
