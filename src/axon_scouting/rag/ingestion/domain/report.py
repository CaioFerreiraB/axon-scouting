"""Relatório de uma execução de ingestão (spec 102 contracts/ingestion-cli.md)."""

from __future__ import annotations

from dataclasses import dataclass

from axon_scouting.rag.index.api import IndexVersion, ValidityStatus


@dataclass(frozen=True, slots=True, kw_only=True)
class DocumentOutcome:
    doc_key: str
    status: ValidityStatus
    chunks: int
    reused: bool
    normative_units: int = 0
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class IngestionReport:
    """Com `problems`, nada foi escrito (tudo ou nada — research R13)."""

    problems: tuple[str, ...] = ()
    documents: tuple[DocumentOutcome, ...] = ()
    removed: tuple[str, ...] = ()
    version: IndexVersion | None = None
    build: str | None = None

    @property
    def ok(self) -> bool:
        return not self.problems
