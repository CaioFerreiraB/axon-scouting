"""A enumeração pela estrutura dos documentos (spec 103 FR-015; research R17; ADR-048). Puro: sem I/O, sem relógio.

Com o tipo de unidade pedido, uma ficha por documento com as unidades desse tipo que os filtros admitem — título,
caminho e página de cada uma —, na ordem do documento; sem tipo, a tool busca os trechos com forma de lista. A ficha é
`<doc_key>#unidades/<tipo>`, e a métrica de fatos da 105 a refaz a partir da árvore gravada no artefato.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from axon_ai_contracts import Chunk, JSONValue

from axon_scouting.rag.index.api import CatalogEntry, IndexFilters, UnitEntry, UnitKind
from axon_scouting.rag.retrieval.domain.ordering import FUSION_SCORE, STRUCTURE_RANK, marks

UNITS_MARKER: Final = "#unidades/"
"""O id da ficha: `<doc_key>#unidades/<tipo>`. Trecho do índice sempre tem `@p<página>` depois do `#`."""


@dataclass(frozen=True, slots=True, kw_only=True)
class EnumerationRequest:
    """O pedido de enumeração que a 101 publicou em `retrieval_filters["enumeration"]`."""

    unit_kind: str | None
    terms: tuple[str, ...]


def units_card_id(doc_key: str, kind: str) -> str:
    return f"{doc_key}{UNITS_MARKER}{kind}"


def units_card_key(chunk_id: str) -> tuple[str, str] | None:
    """(`doc_key`, tipo) de uma ficha de unidades; `None` para outro id."""
    doc_key, marker, kind = chunk_id.partition(UNITS_MARKER)
    return (doc_key, kind) if marker and kind else None


def by_document(entries: Sequence[UnitEntry]) -> list[tuple[CatalogEntry, list[UnitEntry]]]:
    """As unidades agrupadas por documento, na ordem em que vieram (a do índice: `doc_key`, depois a da árvore)."""
    groups: dict[str, tuple[CatalogEntry, list[UnitEntry]]] = {}
    for entry in entries:
        groups.setdefault(entry.document.doc_key, (entry.document, []))[1].append(entry)
    return list(groups.values())


def units_text(document: CatalogEntry, units: Sequence[UnitEntry], kind: UnitKind | None) -> str:
    """A ficha: o documento, quantas unidades do tipo e uma linha por unidade, com o caminho e a página."""
    plural = kind.plurals[0] if kind and kind.plurals else (kind.name if kind else "unidades")
    lines = [document.document_title, f"Lista de {plural} do documento ({len(units)}):"]
    for entry in units:
        where = " · ".join(ref.title for ref in entry.path)
        pages = (
            entry.unit.page_start
            if entry.unit.page_start == entry.unit.page_end
            else (f"{entry.unit.page_start}–{entry.unit.page_end}")
        )
        lines.append(f"- {entry.unit.title}" + (f" — {where}" if where else "") + f" (p. {pages})")
    return "\n".join(lines)


def units_card(
    document: CatalogEntry,
    units: Sequence[UnitEntry],
    *,
    kind: UnitKind | None,
    rank: int,
    total: int,
    filters: IndexFilters,
) -> Chunk:
    """A ficha como `Chunk` da 001, com a marca que a põe à frente dos trechos no reranker."""
    plural = kind.plurals[0] if kind and kind.plurals else (kind.name if kind else "unidades")
    metadata: dict[str, JSONValue] = {
        "document_id": document.document_id,
        "document_type": document.document_type,
        "branch": document.branch,
        "jurisdiction": document.jurisdiction,
        "effective_date": document.effective_date.isoformat(),
        "expiry_date": None if document.expiry_date is None else document.expiry_date.isoformat(),
        "superseded_by": document.superseded_by,
        "unit_kind": units[0].unit.kind if units else None,
        FUSION_SCORE: 1.0,
        STRUCTURE_RANK: rank,
        **marks(document, filters),
    }
    return Chunk(
        chunk_id=units_card_id(document.doc_key, units[0].unit.kind if units else (kind.name if kind else "")),
        text=units_text(document, units, kind),
        score=1.0,
        document=document.document_title,
        version=document.version,
        status=document.status.value,
        section=f"lista de {plural}, {rank + 1} de {total}",
        page=None,
        metadata=metadata,
    )


def with_unit_cards(
    text_of: Callable[[str], str | None],
    units_of: Callable[[str], Sequence[UnitEntry]],
    kinds: Mapping[str, UnitKind],
) -> Callable[[str], str | None]:
    """O texto por id, de trecho ou de ficha de unidades: a ficha se refaz da árvore gravada (a métrica de fatos da
    105). Refeita sem os filtros da pergunta, ela traz todas as unidades do tipo no documento."""

    def text(chunk_id: str) -> str | None:
        key = units_card_key(chunk_id)
        if key is None:
            return text_of(chunk_id)
        doc_key, kind = key
        units = [u for u in units_of(doc_key) if u.unit.kind == kind]
        return units_text(units[0].document, units, kinds.get(kind)) if units else None

    return text
