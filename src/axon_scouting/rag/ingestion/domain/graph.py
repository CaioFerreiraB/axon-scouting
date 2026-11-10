"""O grafo das normas por órgão: a ligação `DISCIPLINA` entre as seções das normas e os órgãos (relatório "Diretoria
Regional", Fase 2; ADR-063).

O catálogo de órgãos e instituições vem de `corpus/organizacao.yaml` (curado). A ligação sai por regra, do título das
unidades de estrutura (título, capítulo, seção) das normas vigentes: "Seção III – Da Diretoria Regional" disciplina a
Diretoria Regional. Vale o nome mais longo que o título contém — "do conselho consultivo regional" é o regional, não o
nacional. O Estatuto (nacional) e o Regulamento da Região SP (regional) disciplinam o mesmo tipo de órgão, e é isso que
leva a hierarquia das normas para a recuperação: a pergunta sobre a Diretoria Regional recebe as duas seções. Puro: sem
I/O, sem relógio.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

NORMATIVE_TYPES: Final = ("estatuto", "regulamento", "por")
"""Os tipos cujas seções disciplinam órgãos: as normas de organização. Resoluções e políticas tratam de temas."""
WHOLE_DOCUMENT_TYPES: Final = ("regulamento",)
"""O regulamento cujo título nomeia um órgão ("Regulamento Interno do Conselho de Administração Nacional") o disciplina
por inteiro: os títulos das seções dele não repetem o nome do órgão."""
STRUCTURE_KINDS: Final = ("titulo", "capitulo", "secao")
SCOPE_ORDER: Final = {"mundial": 0, "interamericano": 1, "nacional": 2, "regional": 3, "local": 4}
"""A norma de cima primeiro: a regra do nível superior abre a resposta, e a do inferior completa (ADR-061)."""


class OrganizationError(ValueError):
    """`organizacao.yaml` fora do formato; a mensagem nomeia a entrada."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Organ:
    """Um tipo de órgão ou de instituição do catálogo, com os nomes que o reconhecem (normalizados)."""

    id: str
    name: str
    level: str
    terms: tuple[str, ...]
    institution: bool = False
    """Tipo de instituição (UEB, região, grupo), e não de órgão: só a seção o disciplina, nunca o documento inteiro."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ChunkPlace:
    """Onde um trecho está: o documento, o nível dele e as unidades da árvore acima dele."""

    chunk_id: str
    doc_key: str
    document_type: str
    document_title: str
    scope: str | None
    vigente: bool
    path: tuple[tuple[str, str, str], ...]
    """(unit_id, kind, title) da raiz até o trecho."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Section:
    """Uma unidade de norma que disciplina um órgão, com os trechos dela na ordem do documento."""

    organ: str
    doc_key: str
    unit_id: str
    title: str
    scope: str | None
    chunk_ids: tuple[str, ...]


def organs_from_mapping(raw: Mapping[str, Any]) -> list[Organ]:
    """Os tipos de órgão e de instituição de `organizacao.yaml`; o nome e os apelidos viram termos normalizados."""
    found: list[Organ] = []
    for key in ("tipos_de_orgao", "tipos_de_instituicao"):
        for entry in raw.get(key) or []:
            if not isinstance(entry, Mapping) or not isinstance(entry.get("id"), str):
                raise OrganizationError(f"{key}: entrada sem id: {entry!r}")
            names = [str(entry.get("nome") or ""), *(str(alias) for alias in entry.get("aliases") or [])]
            terms = tuple(dict.fromkeys(term for term in (normalize(name) for name in names) if len(term) >= 4))
            if not terms:
                raise OrganizationError(f"{entry['id']}: sem nome nem apelido")
            found.append(
                Organ(
                    id=entry["id"],
                    name=str(entry.get("nome") or entry["id"]),
                    level=str(entry.get("nivel") or ""),
                    terms=terms,
                    institution=key == "tipos_de_instituicao",
                )
            )
    ids = [organ.id for organ in found]
    if len(set(ids)) != len(ids):
        raise OrganizationError("id repetido entre órgãos e instituições")
    return found


def organ_of_title(title: str, organs: Sequence[Organ]) -> str | None:
    """O órgão que o título nomeia: o termo mais longo contido nele; nenhum, `None`."""
    text = f" {normalize(title)} "
    best: tuple[int, str] | None = None
    for organ in organs:
        for term in organ.terms:
            if f" {term} " in text and (best is None or len(term) > best[0]):
                best = (len(term), organ.id)
    return None if best is None else best[1]


def sections(places: Iterable[ChunkPlace], organs: Sequence[Organ]) -> list[Section]:
    """As seções das normas vigentes que disciplinam cada órgão: a unidade de estrutura mais funda cujo título nomeia o
    órgão (a seção, e não o capítulo "Do Nível Regional"), com todos os trechos abaixo dela; ou o regulamento inteiro,
    quando o título dele nomeia o órgão."""
    grouped: dict[tuple[str, str], list[str]] = {}
    meta: dict[tuple[str, str], tuple[str, str, str | None]] = {}
    bodies_only = [organ for organ in organs if not organ.institution]
    for place in places:
        if not place.vigente or place.document_type not in NORMATIVE_TYPES:
            continue
        if place.document_type in WHOLE_DOCUMENT_TYPES:
            whole = organ_of_title(place.document_title, bodies_only)
            if whole is not None:
                key = (whole, f"{place.doc_key}#documento")
                grouped.setdefault(key, []).append(place.chunk_id)
                meta[key] = (place.doc_key, place.document_title, place.scope)
                continue
        for unit_id, kind, title in reversed(place.path):
            if kind not in STRUCTURE_KINDS:
                continue
            organ = organ_of_title(title, organs)
            if organ is not None:
                key = (organ, unit_id)
                grouped.setdefault(key, []).append(place.chunk_id)
                meta[key] = (place.doc_key, title, place.scope)
                break
    found = [
        Section(
            organ=organ,
            doc_key=meta[(organ, unit)][0],
            unit_id=unit,
            title=meta[(organ, unit)][1],
            scope=meta[(organ, unit)][2],
            chunk_ids=tuple(dict.fromkeys(chunk_ids)),
        )
        for (organ, unit), chunk_ids in grouped.items()
    ]
    return sorted(found, key=lambda s: (s.organ, SCOPE_ORDER.get(s.scope or "", 9), s.doc_key, s.unit_id))


def normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    plain = "".join(char for char in decomposed if not unicodedata.combining(char))
    return " ".join("".join(char if char.isalnum() else " " for char in plain).split())
