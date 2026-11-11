"""As instituições e os órgãos reais (spec 106 G3, US3; research R20): as seções `instituicoes` e `orgaos` do
`corpus/organizacao.yaml`, com o tipo (`INSTANCIA_DE`), as de cima (`PARTE_DE`) e a fonte.

Decisão 1 do mantenedor: a instância sem fonte no acervo fica fora, e a aresta que aponta para ela também. Tipo
desconhecido é erro do catálogo (`OrganizationError`). Puro: recebe o YAML já lido e os ids dos trechos do índice.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Final

from axon_scouting.rag.graph.domain.build.organs import VERIFY, OrganizationError, normalize

SECTIONS: Final = (
    ("instituicoes", "inst", "tipos_de_instituicao", "tipo_inst"),
    ("orgaos", "orgao", "tipos_de_orgao", "tipo_orgao"),
)


@dataclass(frozen=True, slots=True, kw_only=True)
class Instance:
    id: str
    """`inst:ge_inah`, `orgao:dr_sp`."""
    type: str
    """`inst` ou `orgao`."""
    name: str
    kind_of: str
    """O tipo: `tipo_inst:grupo_escoteiro`, `tipo_orgao:diretoria_regional`."""
    level: str | None
    terms: tuple[str, ...]
    """Os apelidos normalizados que reconhecem a instância numa pergunta (vazio: só o tipo reconhece)."""
    source: str
    parents: tuple[str, ...] = ()
    """As instituições de que esta é parte (`inst:…`)."""
    parent_source: str | None = None
    jurisdiction: str | None = None
    folder: str | None = None
    taxonomy: str | None = None
    """No órgão, a pasta da taxonomia do tipo (o `taxonomia` do tipo): a dos documentos que ele emite."""


@dataclass(slots=True)
class Instances:
    found: list[Instance] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)

    def institutions(self) -> list[Instance]:
        return [i for i in self.found if i.type == "inst"]

    def organs(self) -> list[Instance]:
        return [i for i in self.found if i.type == "orgao"]


def instances_from_mapping(raw: Mapping[str, Any], known_chunks: set[str]) -> Instances:
    types: dict[str, Mapping[str, Any]] = {}
    for _, _, type_section, type_prefix in SECTIONS:
        for entry in raw.get(type_section) or []:
            if isinstance(entry, Mapping) and isinstance(entry.get("id"), str):
                types[f"{type_prefix}:{entry['id']}"] = entry
    result = Instances()
    for section, prefix, _, type_prefix in SECTIONS:
        for entry in raw.get(section) or []:
            if not isinstance(entry, Mapping) or not isinstance(entry.get("id"), str):
                raise OrganizationError(f"{section}: entrada sem id: {entry!r}")
            ident = f"{prefix}:{entry['id']}"
            kind_of = f"{type_prefix}:{entry.get('instancia_de')}"
            if kind_of not in types:
                raise OrganizationError(
                    f"{section}/{entry['id']}: instancia_de desconhecido: {entry.get('instancia_de')!r}"
                )
            source = entry.get("fonte")
            if not isinstance(source, str) or source == VERIFY or source not in known_chunks:
                result.dropped.append(f"{ident}: fonte {source!r} fora do acervo")
                continue
            kind = types[kind_of]
            parents = entry.get("parte_de") or ()
            parents = (parents,) if isinstance(parents, str) else tuple(str(p) for p in parents)
            parent_source = entry.get("parte_de_fonte")
            names = [str(alias) for alias in entry.get("aliases") or ()]
            result.found.append(
                Instance(
                    id=ident,
                    type=prefix,
                    name=str(entry.get("nome") or entry["id"]),
                    kind_of=kind_of,
                    level=str(entry.get("nivel") or kind.get("nivel") or "") or None,
                    terms=tuple(dict.fromkeys(t for t in (normalize(n) for n in names) if len(t) >= 4)),
                    source=source,
                    parents=tuple(f"inst:{p}" for p in parents),
                    parent_source=parent_source
                    if isinstance(parent_source, str) and parent_source in known_chunks
                    else None,
                    jurisdiction=str(entry["jurisdicao"]) if entry.get("jurisdicao") else None,
                    folder=str(entry["pasta"]) if entry.get("pasta") else None,
                    taxonomy=str(kind["taxonomia"]) if prefix == "orgao" and kind.get("taxonomia") else None,
                )
            )
    ids = [i.id for i in result.found]
    if len(set(ids)) != len(ids):
        raise OrganizationError("instituicoes/orgaos: id repetido")
    kept = set(ids)
    for instance in list(result.found):
        missing = [p for p in instance.parents if p not in kept]
        if missing:
            result.dropped.append(f"PARTE_DE {instance.id} → {', '.join(missing)}: instituição fora do catálogo")
    return result


def type_parts(raw: Mapping[str, Any], known_chunks: set[str]) -> list[tuple[str, str, str]]:
    """(tipo de órgão, tipo de instituição, fonte): o `instituicao` de cada tipo de órgão vira `PARTE_DE` entre os
    tipos — a Assembleia de Grupo é parte do Grupo Escoteiro. A fonte é o `instituido_em`, quando está no acervo; senão,
    a entrada do catálogo."""
    institutions = {e["id"] for e in raw.get("tipos_de_instituicao") or [] if isinstance(e, Mapping) and "id" in e}
    found: list[tuple[str, str, str]] = []
    for entry in raw.get("tipos_de_orgao") or []:
        if not isinstance(entry, Mapping) or entry.get("instituicao") not in institutions:
            continue
        established = entry.get("instituido_em")
        source = (
            established
            if isinstance(established, str) and established in known_chunks
            else f"organizacao.yaml:tipos_de_orgao/{entry['id']}"
        )
        found.append((f"tipo_orgao:{entry['id']}", f"tipo_inst:{entry['instituicao']}", source))
    return found
