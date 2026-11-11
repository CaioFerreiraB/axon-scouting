"""As entidades curadas de programa e adultos (spec 106 G2, research R5): `corpus/dominio.yaml` (ramos, seções, áreas,
conceitos, `rege`) e os cargos do `corpus/organizacao.yaml` que têm `aliases`. Puro: recebe os YAML já lidos."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from axon_scouting.rag.graph.domain.build.organs import VERIFY, OrganizationError, normalize

SECTIONS: Final = (
    ("ramos", "ramo"),
    ("secoes", "secao"),
    ("areas", "area"),
    ("conceitos", "conceito"),
    ("documentos", "doc"),
)
"""`documentos` (G4): o documento como alvo da linha do tempo ("o que se decidiu sobre o Regulamento Regional"), pelo
`document_id`; os apelidos do título e do número de todo o acervo são do G5."""
DOMAIN: Final = "dominio.yaml"
ORGANIZATION: Final = "organizacao.yaml"


@dataclass(frozen=True, slots=True, kw_only=True)
class DomainEntity:
    id: str
    type: str
    name: str
    aliases: tuple[str, ...]
    """Os nomes normalizados que ligam títulos (e, com `recognizable`, reconhecem a pergunta)."""
    recognizable: bool
    source: str
    ramo: str | None = None
    categories: tuple[str, ...] = ()
    covers: tuple[str, ...] = ()
    """As entidades de que esta é o termo genérico (`abrange`): cada uma `E_UM` esta."""


def domain_entities(domain: Mapping[str, Any], organization: Mapping[str, Any]) -> list[DomainEntity]:
    found: list[DomainEntity] = []
    for entry in organization.get("cargos") or []:
        # Só os cargos de programa e adultos (sem `orgao`): os de governança têm apelidos curtos ("vice",
        # "presidente") que valem em vários órgãos, e o nome deles já está nos termos do órgão (102 R43).
        if (
            isinstance(entry, Mapping)
            and entry.get("aliases")
            and not entry.get("orgao")
            and isinstance(entry.get("id"), str)
        ):
            found.append(_entity(entry, "cargo", ORGANIZATION, "cargos"))
    for section, kind in SECTIONS:
        for entry in domain.get(section) or []:
            if not isinstance(entry, Mapping) or not isinstance(entry.get("id"), str):
                raise OrganizationError(f"{DOMAIN}: {section}: entrada sem id: {entry!r}")
            found.append(_entity(entry, kind, DOMAIN, section))
    ids = [entity.id for entity in found]
    if len(set(ids)) != len(ids):
        raise OrganizationError(f"{DOMAIN}: id repetido")
    return found


def _entity(entry: Mapping[str, Any], kind: str, file: str, section: str) -> DomainEntity:
    aliases = tuple(dict.fromkeys(t for t in (normalize(str(a)) for a in entry.get("aliases") or ()) if len(t) >= 3))
    if not aliases:
        raise OrganizationError(f"{file}: {section}/{entry['id']}: sem apelidos")
    fonte = entry.get("fonte")
    return DomainEntity(
        id=f"{kind}:{entry['id']}",
        type=kind,
        name=str(entry.get("nome") or entry["id"]),
        aliases=aliases,
        recognizable=bool(entry.get("reconhece", True)),
        source=fonte if isinstance(fonte, str) and fonte != VERIFY else f"{file}:{section}/{entry['id']}",
        ramo=f"ramo:{entry['ramo']}" if entry.get("ramo") else None,
        categories=tuple(str(c) for c in entry.get("categorias") or ()),
        covers=tuple(str(c) for c in entry.get("abrange") or ()),
    )


def rege(domain: Mapping[str, Any]) -> list[tuple[str, str, str]]:
    """(área, alvo, fonte) das arestas `REGE` curadas."""
    found: list[tuple[str, str, str]] = []
    for entry in domain.get("rege") or []:
        if isinstance(entry, Mapping):
            for target in entry.get("para") or []:
                found.append((f"area:{entry['area']}", str(target), str(entry.get("fonte"))))
    return found


def alias_map(entities: Sequence[DomainEntity]) -> dict[str, tuple[str, ...]]:
    return {entity.id: entity.aliases for entity in entities}
