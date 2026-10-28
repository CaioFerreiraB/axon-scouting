"""A árvore de unidades lida dos registros de documento (spec 102 FR-029; ADR-048), igual em todo adapter.

Os adapters entregam os registros do build publicado; aqui se monta o caminho de cada unidade e se aplicam os filtros:
os de vigência e tipo, pelo documento (`matches`, a semântica da busca), e o de ramo, pelos ramos da unidade — senão
pelo do documento.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable

from axon_scouting.rag.index.api import IndexFilters, UnitEntry, UnitNode, UnitRef
from axon_scouting.rag.index.domain.filters import matches
from axon_scouting.rag.index.domain.points import catalog_entry
from axon_scouting.rag.index.ports import StoredDocument


def outline_of(document: StoredDocument) -> list[UnitEntry]:
    """A árvore do documento em pré-ordem, cada unidade com o caminho acima dela."""
    entry = catalog_entry(document)
    by_id = {unit.unit_id: unit for unit in document.units}
    return [UnitEntry(unit=unit, path=_path(unit, by_id), document=entry) for unit in document.units]


def units_of(documents: Iterable[StoredDocument], kind: str, filters: IndexFilters) -> list[UnitEntry]:
    """As unidades de `kind` nos documentos que os filtros admitem, por `doc_key` e na ordem da árvore."""
    found: list[UnitEntry] = []
    unbranched = dataclasses.replace(filters, branch=None, shape=None)
    for document in sorted(documents, key=lambda d: d.doc_key):
        entry = catalog_entry(document)
        if not matches(unbranched, entry):
            continue
        for unit_entry in outline_of(document):
            if unit_entry.unit.kind != kind:
                continue
            branches = unit_entry.unit.branches or ((entry.branch,) if entry.branch else ())
            if filters.branch is not None and branches and filters.branch not in branches:
                continue
            found.append(unit_entry)
    return found


def _path(unit: UnitNode, by_id: dict[str, UnitNode]) -> tuple[UnitRef, ...]:
    path: list[UnitRef] = []
    current = by_id.get(unit.parent) if unit.parent else None
    while current is not None:
        path.append(UnitRef(unit_id=current.unit_id, kind=current.kind, title=current.title))
        current = by_id.get(current.parent) if current.parent else None
    return tuple(reversed(path))
