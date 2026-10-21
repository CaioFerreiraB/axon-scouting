"""Vigência e sucessão por linhagem, derivadas do catálogo inteiro (spec 102 FR-007, FR-008; research R4).

O status é a posição da versão na linhagem — estrutural, sem relógio: no máximo uma `vigente` por linhagem, por
construção (SC-003). Se uma versão estava em vigor numa data quem decide são as datas, na busca (research R5).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta

from axon_scouting.rag.index.api import ValidityStatus
from axon_scouting.rag.ingestion.domain.catalog import CatalogEntry


@dataclass(frozen=True, slots=True, kw_only=True)
class Validity:
    status: ValidityStatus
    effective_date: date
    expiry_date: date | None = None
    superseded_by: str | None = None
    successor_expiry_date: date | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class LineageProblem:
    document_id: str
    message: str


@dataclass(frozen=True, slots=True, kw_only=True)
class LineageResult:
    validities: Mapping[str, Validity]
    """Por chave de documento (`<document_id>-<version>`)."""
    problems: tuple[LineageProblem, ...]

    @property
    def ok(self) -> bool:
        return not self.problems


def derive_lineages(entries: Sequence[CatalogEntry]) -> LineageResult:
    groups: dict[str, list[CatalogEntry]] = defaultdict(list)
    for entry in entries:
        groups[entry.document_id].append(entry)
    validities: dict[str, Validity] = {}
    problems: list[LineageProblem] = []
    for document_id, versions in groups.items():
        derived, messages = _derive(versions)
        problems.extend(LineageProblem(document_id=document_id, message=m) for m in messages)
        if not messages:
            validities.update(derived)
    return LineageResult(validities=validities if not problems else {}, problems=tuple(problems))


def _derive(versions: list[CatalogEntry]) -> tuple[dict[str, Validity], list[str]]:
    """Cada etapa só roda se a anterior passou: a mensagem aponta a causa, não as consequências."""
    by_version = {entry.version: entry for entry in versions}

    missing = [e for e in versions if e.supersedes is not None and e.supersedes not in by_version]
    if missing:
        return {}, [f"{e.label} declara supersedes {e.supersedes!r}, que não existe na linhagem" for e in missing]

    for entry in versions:
        seen = {entry.version}
        current = entry
        while current.supersedes is not None:
            if current.supersedes in seen:
                return {}, [f"ciclo de substituição envolvendo {entry.label}"]
            seen.add(current.supersedes)
            current = by_version[current.supersedes]

    successors: dict[str, list[CatalogEntry]] = defaultdict(list)
    for entry in versions:
        if entry.supersedes is not None and entry.status is not ValidityStatus.RASCUNHO:
            successors[entry.supersedes].append(entry)
    doubled = {version: found for version, found in successors.items() if len(found) > 1}
    if doubled:
        return {}, [
            f"versão {version} substituída por mais de uma: {', '.join(e.version for e in found)}"
            for version, found in doubled.items()
        ]

    messages = [
        f"{found[0].label} substitui {version}, mas não começa depois dela "
        f"({found[0].effective_date} ≤ {by_version[version].effective_date})"
        for version, found in successors.items()
        if found[0].effective_date <= by_version[version].effective_date
    ]
    messages += [
        f"{e.label} está revogada e também substituída por {successors[e.version][0].version}"
        for e in versions
        if e.status is ValidityStatus.REVOGADO and e.version in successors
    ]
    if messages:
        return {}, messages

    own: dict[str, Validity] = {}
    for entry in versions:
        successor = successors.get(entry.version, [None])[0]
        if entry.status in (ValidityStatus.RASCUNHO, ValidityStatus.REVOGADO):
            own[entry.version] = Validity(
                status=entry.status, effective_date=entry.effective_date, expiry_date=entry.expiry_date
            )
        elif successor is not None:
            expiry = successor.effective_date - timedelta(days=1)
            if entry.expiry_date is not None and entry.expiry_date != expiry:
                messages.append(
                    f"{entry.label} declara expiry_date {entry.expiry_date}, mas a véspera do sucessor "
                    f"{successor.version} é {expiry}"
                )
            own[entry.version] = Validity(
                status=ValidityStatus.SUBSTITUIDO,
                effective_date=entry.effective_date,
                expiry_date=expiry,
                superseded_by=successor.doc_key,
            )
        else:
            own[entry.version] = Validity(
                status=ValidityStatus.VIGENTE, effective_date=entry.effective_date, expiry_date=entry.expiry_date
            )
    if messages:
        return {}, messages

    in_force = sorted(v for v, validity in own.items() if validity.status is ValidityStatus.VIGENTE)
    if len(in_force) > 1:
        return {}, [
            f"mais de uma versão vigente ({', '.join(in_force)}): declare qual substitui qual com supersedes, "
            "ou marque as demais como rascunho ou revogado"
        ]

    derived: dict[str, Validity] = {}
    for entry in versions:
        validity = own[entry.version]
        if validity.status is ValidityStatus.SUBSTITUIDO:
            successor = successors[entry.version][0]
            validity = Validity(
                status=validity.status,
                effective_date=validity.effective_date,
                expiry_date=validity.expiry_date,
                superseded_by=validity.superseded_by,
                successor_expiry_date=own[successor.version].expiry_date,
            )
        derived[entry.doc_key] = validity
    return derived, []
