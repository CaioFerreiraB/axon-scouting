"""Catálogo do corpus (`corpus.yaml`): validação pura das entradas (spec 102 FR-011; research R3).

O adapter só lê o YAML; aqui se valida o objeto já carregado. Todos os problemas são acumulados — o curador corrige o
catálogo de uma vez —, e qualquer problema impede a execução inteira (research R13). A existência do arquivo é do
parser; aqui o caminho só é conferido como texto.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import PurePath
from typing import Final

from ai_scouting.rag.index.api import BRANCHES, DocumentType, ValidityStatus

REQUIRED: Final = ("file", "document_id", "version", "type", "title", "jurisdiction", "effective_date")
OPTIONAL: Final = ("branch", "publication_date", "expiry_date", "supersedes", "status")
DECLARABLE_STATUS: Final = (ValidityStatus.RASCUNHO, ValidityStatus.REVOGADO)
"""Os únicos status que o curador escreve; `vigente` e `substituido` são derivados (research R4)."""

_DOCUMENT_ID: Final = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
_VERSION: Final = re.compile(r"^[0-9a-z]+(\.[0-9a-z]+)*$")


@dataclass(frozen=True, slots=True, kw_only=True)
class CatalogEntry:
    """Uma versão de documento, como o curador a declarou."""

    entry: int
    file: str
    path: PurePath
    document_id: str
    version: str
    document_type: DocumentType
    title: str
    jurisdiction: str
    effective_date: date
    branch: str | None = None
    publication_date: date | None = None
    expiry_date: date | None = None
    supersedes: str | None = None
    status: ValidityStatus | None = None

    @property
    def doc_key(self) -> str:
        return f"{self.document_id}-{self.version}"

    @property
    def label(self) -> str:
        return f"{self.document_id}@{self.version}"


@dataclass(frozen=True, slots=True, kw_only=True)
class CatalogProblem:
    entry: int | None
    key: str | None
    fields: tuple[str, ...]
    message: str


@dataclass(frozen=True, slots=True, kw_only=True)
class CatalogResult:
    entries: tuple[CatalogEntry, ...]
    problems: tuple[CatalogProblem, ...]

    @property
    def ok(self) -> bool:
        return not self.problems


class _Invalid(Exception):
    pass


def validate_catalog(raw: object, base_dir: PurePath) -> CatalogResult:
    """Valida o objeto carregado do `corpus.yaml`; `base_dir` é o diretório do catálogo."""
    if not isinstance(raw, Mapping) or not isinstance(raw.get("documents"), list) or not raw["documents"]:
        return CatalogResult(
            entries=(),
            problems=(
                CatalogProblem(
                    entry=None, key=None, fields=(), message="catálogo sem `documents` (uma lista não vazia)"
                ),
            ),
        )
    entries: list[CatalogEntry] = []
    problems: list[CatalogProblem] = []
    for number, item in enumerate(raw["documents"], start=1):
        entry, found = _validate_entry(number, item, base_dir)
        problems.extend(found)
        if entry is not None:
            entries.append(entry)
    seen: dict[tuple[str, str], int] = {}
    for entry in entries:
        pair = (entry.document_id, entry.version)
        if pair in seen:
            problems.append(
                CatalogProblem(
                    entry=entry.entry,
                    key=entry.label,
                    fields=("document_id", "version"),
                    message=f"entrada {entry.entry}: {entry.label} repete a entrada {seen[pair]}",
                )
            )
        else:
            seen[pair] = entry.entry
    problems.sort(key=lambda p: p.entry or 0)
    return CatalogResult(entries=() if problems else tuple(entries), problems=tuple(problems))


def _validate_entry(number: int, item: object, base_dir: PurePath) -> tuple[CatalogEntry | None, list[CatalogProblem]]:
    if not isinstance(item, Mapping):
        return None, [CatalogProblem(entry=number, key=None, fields=(), message=f"entrada {number}: não é um mapa")]
    key = _key(item)
    prefix = f"entrada {number}" + (f" ({key})" if key else "")
    problems: list[CatalogProblem] = []

    def problem(fields: Sequence[str], message: str) -> None:
        problems.append(CatalogProblem(entry=number, key=key, fields=tuple(fields), message=f"{prefix}: {message}"))

    unknown = sorted(str(k) for k in item if k not in REQUIRED and k not in OPTIONAL)
    for name in unknown:
        problem((name,), f"campo desconhecido: {name}")
    missing = [name for name in REQUIRED if item.get(name) in (None, "")]
    if missing:
        problem(missing, f"campos obrigatórios ausentes: {', '.join(missing)}")

    values: dict[str, object] = {}
    for name, parse in _PARSERS.items():
        if item.get(name) in (None, "") or name in missing:
            continue
        try:
            values[name] = parse(item[name])
        except _Invalid as exc:
            problem((name,), f"campo inválido {name}={item[name]!r}: {exc}")

    if values.get("status") is ValidityStatus.REVOGADO and "expiry_date" not in values:
        problem(("expiry_date",), "status revogado exige expiry_date")
    if problems:
        return None, problems

    file = str(values["file"])
    return (
        CatalogEntry(
            entry=number,
            file=file,
            path=base_dir / file,
            document_id=str(values["document_id"]),
            version=str(values["version"]),
            document_type=_as(values["type"], DocumentType),
            title=str(values["title"]),
            jurisdiction=str(values["jurisdiction"]),
            effective_date=_as(values["effective_date"], date),
            branch=_opt_str(values.get("branch")),
            publication_date=_opt(values.get("publication_date"), date),
            expiry_date=_opt(values.get("expiry_date"), date),
            supersedes=_opt_str(values.get("supersedes")),
            status=_opt(values.get("status"), ValidityStatus),
        ),
        [],
    )


def _key(item: Mapping[object, object]) -> str | None:
    document_id, version = item.get("document_id"), item.get("version")
    if isinstance(document_id, str) and isinstance(version, str | int):
        return f"{document_id}@{version}"
    return None


def _file(value: object) -> str:
    if not isinstance(value, str):
        raise _Invalid("esperado um caminho relativo ao catálogo")
    path = PurePath(value)
    if path.is_absolute() or ".." in path.parts:
        raise _Invalid("o arquivo precisa ficar dentro do diretório do catálogo (caminho relativo, sem `..`)")
    return value


def _document_id(value: object) -> str:
    if not isinstance(value, str) or not _DOCUMENT_ID.match(value):
        raise _Invalid("esperado ^[a-z0-9]+(-[a-z0-9]+)*$")
    return value


def _version(value: object) -> str:
    if isinstance(value, bool):
        raise _Invalid("esperado texto")
    if isinstance(value, float):
        raise _Invalid("número decimal no YAML perde zeros (1.10 vira 1.1); escreva a versão entre aspas")
    text = str(value) if isinstance(value, int) else value
    if not isinstance(text, str) or not _VERSION.match(text):
        raise _Invalid(r"esperado ^[0-9a-z]+(\.[0-9a-z]+)*$ (sem hífen)")
    return text


def _type(value: object) -> DocumentType:
    try:
        return DocumentType(str(value))
    except ValueError:
        raise _Invalid(f"esperado um de {', '.join(t.value for t in DocumentType)}") from None


def _text(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _Invalid("esperado texto não vazio")
    return value.strip()


def _branch(value: object) -> str:
    if value not in BRANCHES:
        raise _Invalid(f"esperado um de {', '.join(BRANCHES)} (ou omitido, para documento geral)")
    return str(value)


def _date(value: object) -> date:
    if isinstance(value, datetime):
        raise _Invalid("esperada uma data, sem hora")
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError:
            pass
    raise _Invalid("esperada uma data AAAA-MM-DD")


def _status(value: object) -> ValidityStatus:
    if value not in [s.value for s in DECLARABLE_STATUS]:
        raise _Invalid("só `rascunho` ou `revogado`; `vigente` e `substituido` são derivados")
    return ValidityStatus(str(value))


_PARSERS: Final = {
    "file": _file,
    "document_id": _document_id,
    "version": _version,
    "type": _type,
    "title": _text,
    "jurisdiction": _text,
    "effective_date": _date,
    "branch": _branch,
    "publication_date": _date,
    "expiry_date": _date,
    "supersedes": _version,
    "status": _status,
}


def _as[T](value: object, kind: type[T]) -> T:
    assert isinstance(value, kind)
    return value


def _opt[T](value: object, kind: type[T]) -> T | None:
    return None if value is None else _as(value, kind)


def _opt_str(value: object) -> str | None:
    return None if value is None else str(value)
