"""Taxonomia do acervo e forma dos documentos (`corpus/taxonomia.yaml` e `corpus/estrutura.yaml`; spec 102 FR-023,
FR-024; ADR-048; research R33, R34).

Validação pura dos objetos já carregados, como a do catálogo: todos os problemas são acumulados, e qualquer um impede a
execução. A pasta do arquivo é a entrada da taxonomia (a região ou a UEL como subpasta dela); o tipo semântico da
entrada dá o perfil, salvo a primeira exceção de `documents` que casa (por pasta ou `document_id`). Sem os dois
arquivos, a ingestão não roda (ADR-049): o tipo do documento é o da taxonomia.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from fnmatch import fnmatchcase
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Final

from axon_scouting.rag.index.api import BRANCHES, SCOPES, taxonomy_id
from axon_scouting.rag.ingestion.domain.structure import UnitKind
from axon_scouting.rag.ingestion.domain.text import fold

PROFILES: Final = frozenset(
    {
        "regras",
        "artigos",
        "secoes_numeradas",
        "capitulos",
        "progressao",
        "especialidades",
        "ata",
        "relatorio",
        "comunicado",
        "apresentacao",
        "perguntas",
    }
)
"""Os perfis que a árvore (`domain/tree.py`) sabe montar."""
NORMATIVE_PROFILES: Final[Mapping[str, UnitKind]] = {"regras": "rule", "artigos": "article"}
"""Perfil normativo → a unidade da `structure.py`."""
SOURCES: Final = frozenset({"marcadores", "sumário", "bookmarks", "rodapé", "títulos", "páginas"})
_PAGES: Final = re.compile(r"^(\d+)-(\d+)$")


@dataclass(frozen=True, slots=True, kw_only=True)
class TaxonomyEntry:
    category: str
    subcategory: str | None
    level: str | None
    semantic_type: str
    folder: str
    inferred: bool = False


@dataclass(frozen=True, slots=True, kw_only=True)
class UnitKindSpec:
    name: str
    label: str
    plurals: tuple[str, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class Profile:
    name: str
    sources: tuple[str, ...]
    units: tuple[str, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class UnitDeclaration:
    """Uma unidade declarada numa exceção: o tipo e de onde ela vem (faixa de páginas do sumário, padrão de título ou
    um grupo nomeado, com ramos)."""

    kind: str
    pages: tuple[int, int] | None = None
    source: str | None = None
    parent: str | None = None
    name: str | None = None
    branches: tuple[str, ...] = ()
    pattern: str | None = None
    under: tuple[str, ...] = ()
    trim: str | None = None
    """Padrão tirado do fim do título da unidade (o cabeçalho de tabela colado ao nome da competência; ADR-050)."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Rule:
    match: tuple[str, ...] = ()
    folder: tuple[str, ...] = ()
    profile: str
    branches: tuple[str, ...] = ()
    units: tuple[UnitDeclaration, ...] = ()
    sources: tuple[str, ...] | None = None
    """Fontes no lugar das do perfil: o guia de cargos lê só os títulos, porque o sumário dele cobriria os cargos."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Axis:
    name: str
    blocks: tuple[str, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class Resolution:
    """O perfil de um documento e a exceção que o deu (`None` = o tipo semântico, ou o `type` sem taxonomia)."""

    profile: Profile
    rule: Rule | None = None
    classification: TaxonomyEntry | None = None
    axes: tuple[Axis, ...] = ()
    aliases: Mapping[str, str] = field(default_factory=dict)
    unit_kinds: Mapping[str, UnitKindSpec] = field(default_factory=dict)

    def fingerprint(self) -> str:
        """Muda quando muda o que define a árvore deste documento: entra na chave de conteúdo."""
        raw = {
            "profile": asdict(self.profile),
            "rule": None if self.rule is None else asdict(self.rule),
            "axes": [asdict(a) for a in self.axes],
            "aliases": dict(sorted(self.aliases.items())),
            "kinds": {k: asdict(v) for k, v in sorted(self.unit_kinds.items())},
        }
        return hashlib.sha256(json.dumps(raw, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]


@dataclass(frozen=True, slots=True, kw_only=True)
class CorpusStructure:
    taxonomy: tuple[TaxonomyEntry, ...]
    unit_kinds: Mapping[str, UnitKindSpec]
    axes: tuple[Axis, ...]
    aliases: Mapping[str, str]
    profiles: Mapping[str, Profile]
    profile_by_type: Mapping[str, str]
    rules: tuple[Rule, ...]

    def classify(self, file: str) -> TaxonomyEntry | None:
        """A entrada da pasta do arquivo (sem o `ocr/`); a região ou a UEL é subpasta dela."""
        folder = PurePosixPath(file).parent
        if folder.parts[:1] == ("ocr",):
            folder = PurePosixPath(*folder.parts[1:])
        by_folder = {entry.folder: entry for entry in self.taxonomy}
        return by_folder.get(str(folder)) or by_folder.get(str(folder.parent))

    def resolve(self, document_id: str, entry: TaxonomyEntry) -> Resolution:
        rule = next((r for r in self.rules if _applies(r, document_id, entry.folder)), None)
        name = rule.profile if rule is not None else self.profile_by_type[entry.semantic_type]
        profile = self.profiles[name]
        if rule is not None and rule.sources is not None:
            profile = Profile(name=profile.name, sources=rule.sources, units=profile.units)
        return Resolution(
            profile=profile,
            rule=rule,
            classification=entry,
            axes=self.axes,
            aliases=self.aliases,
            unit_kinds=self.unit_kinds,
        )


def _applies(rule: Rule, document_id: str, folder: str) -> bool:
    return any(fnmatchcase(document_id, m) for m in rule.match) or any(fnmatchcase(folder, f) for f in rule.folder)


def type_of(resolution: Resolution) -> str:
    """O id do tipo do documento: o da entrada da taxonomia (ADR-049)."""
    assert resolution.classification is not None
    return taxonomy_id(resolution.classification.semantic_type)


def bodies_of(resolution: Resolution) -> tuple[str, ...]:
    """Os órgãos do documento: os ids da categoria e da subcategoria da taxonomia (ADR-049)."""
    entry = resolution.classification
    assert entry is not None
    return tuple(dict.fromkeys(taxonomy_id(v) for v in (entry.category, entry.subcategory) if v))


JURISDICTION_SCOPES: Final = MappingProxyType(
    {
        "nacional": "nacional",
        "mundial": "mundial",
        "interamericana": "interamericano",
        "interamericano": "interamericano",
    }
)
"""A abrangência pela jurisdição do catálogo, quando a taxonomia não diz o nível; a de uma região é `regional`."""


def scope_of(resolution: Resolution, jurisdiction: str) -> str:
    """O nível da taxonomia como id; sem ele, a jurisdição do catálogo (uma região é `regional`; ADR-049)."""
    entry = resolution.classification
    if entry is not None and entry.level and taxonomy_id(entry.level) in SCOPES:
        return taxonomy_id(entry.level)
    return JURISDICTION_SCOPES.get(jurisdiction, "regional")


def parse_structure(taxonomy: object, structure: object) -> tuple[CorpusStructure | None, list[str]]:
    """Os dois arquivos validados juntos; com problema, `None` e a lista inteira."""
    problems: list[str] = []
    entries = _taxonomy(taxonomy, problems)
    if not isinstance(structure, Mapping) or structure.get("version") != 1:
        return None, [*problems, "estrutura.yaml: esperado um mapa com `version: 1`"]
    kinds = _unit_kinds(structure.get("unit_kinds"), problems)
    axes, aliases = _vocabulary(structure.get("vocabulary"), problems)
    profiles = _profiles(structure.get("profiles"), kinds, problems)
    by_type = _by_type(structure.get("profile_by_type"), profiles, problems)
    for entry in entries:
        if entry.semantic_type not in by_type:
            problems.append(f"taxonomia: o tipo {entry.semantic_type!r} ({entry.folder}) não tem perfil")
    rules = _rules(structure.get("documents"), profiles, kinds, problems)
    if problems:
        return None, problems
    return (
        CorpusStructure(
            taxonomy=tuple(entries),
            unit_kinds=kinds,
            axes=axes,
            aliases=aliases,
            profiles=profiles,
            profile_by_type=by_type,
            rules=tuple(rules),
        ),
        [],
    )


def canonical_axis(text: str, structure: Resolution) -> str | None:
    """O eixo do vocabulário cujo nome (ou sinônimo) aparece no texto, sem caixa nem acento."""
    folded = fold(text)
    for alias, target in structure.aliases.items():
        if fold(alias) in folded:
            return target
    return next((axis.name for axis in structure.axes if fold(axis.name) in folded), None)


def canonical_block(text: str, structure: Resolution) -> tuple[str, str] | None:
    """(eixo, bloco) do bloco do vocabulário com que o título começa — o nome varia no fim ("Aprendizagem Contínua e
    Desenvolvimento Vocacional")."""
    folded = fold(text)
    for axis in structure.axes:
        for block in axis.blocks:
            if folded.startswith(fold(block)):
                return axis.name, block
    return None


# --- leitura -------------------------------------------------------------------------------------------------------


def _taxonomy(raw: object, problems: list[str]) -> list[TaxonomyEntry]:
    if not isinstance(raw, Mapping) or raw.get("version") != 1 or not isinstance(raw.get("entries"), list):
        problems.append("taxonomia.yaml: esperado um mapa com `version: 1` e `entries` (lista)")
        return []
    entries: list[TaxonomyEntry] = []
    seen: set[str] = set()
    for number, item in enumerate(raw["entries"], start=1):
        required = ("categoria", "tipo", "pasta")
        if not isinstance(item, Mapping) or not all(isinstance(item.get(k), str) for k in required):
            problems.append(f"taxonomia, entrada {number}: `categoria`, `tipo` e `pasta` são obrigatórios")
            continue
        folder = str(item["pasta"]).strip("/")
        path = PurePosixPath(folder)
        if path.is_absolute() or ".." in path.parts or path.parts[:1] == ("ocr",):
            problems.append(f"taxonomia, entrada {number}: pasta inválida {folder!r}")
            continue
        if folder in seen:
            problems.append(f"taxonomia, entrada {number}: pasta repetida {folder!r}")
            continue
        seen.add(folder)
        entries.append(
            TaxonomyEntry(
                category=str(item["categoria"]),
                subcategory=_opt_text(item.get("subcategoria")),
                level=_opt_text(item.get("nivel")),
                semantic_type=str(item["tipo"]),
                folder=folder,
                inferred=item.get("origem") == "inferido",
            )
        )
    return entries


def _unit_kinds(raw: object, problems: list[str]) -> dict[str, UnitKindSpec]:
    if not isinstance(raw, Mapping) or not raw:
        problems.append("estrutura.yaml: `unit_kinds` vazio")
        return {}
    kinds: dict[str, UnitKindSpec] = {}
    for name, spec in raw.items():
        spec = spec if isinstance(spec, Mapping) else {}
        plural = spec.get("plural", [])
        plurals = (plural,) if isinstance(plural, str) else tuple(str(p) for p in plural or ())
        label = spec.get("label") or str(name).capitalize()
        kinds[str(name)] = UnitKindSpec(name=str(name), label=str(label), plurals=plurals)
    return kinds


def _vocabulary(raw: object, problems: list[str]) -> tuple[tuple[Axis, ...], dict[str, str]]:
    if raw is None:
        return (), {}
    if not isinstance(raw, Mapping):
        problems.append("estrutura.yaml: `vocabulary` precisa ser um mapa")
        return (), {}
    axes_raw = raw.get("eixo") or {}
    axes = tuple(
        Axis(name=str(name), blocks=tuple(str(b) for b in blocks or ()))
        for name, blocks in (axes_raw.items() if isinstance(axes_raw, Mapping) else ())
    )
    aliases = {str(k): str(v) for k, v in (raw.get("aliases") or {}).items()}
    names = {axis.name for axis in axes}
    for alias, target in aliases.items():
        if target not in names:
            problems.append(f"estrutura.yaml: sinônimo {alias!r} aponta para eixo inexistente {target!r}")
    return axes, aliases


def _profiles(raw: object, kinds: Mapping[str, UnitKindSpec], problems: list[str]) -> dict[str, Profile]:
    if not isinstance(raw, Mapping) or not raw:
        problems.append("estrutura.yaml: `profiles` vazio")
        return {}
    profiles: dict[str, Profile] = {}
    for name, spec in raw.items():
        spec = spec if isinstance(spec, Mapping) else {}
        if name not in PROFILES:
            known = ", ".join(sorted(PROFILES))
            problems.append(f"estrutura.yaml: perfil {name!r} sem implementação (conhecidos: {known})")
            continue
        sources = tuple(str(s) for s in spec.get("sources") or ())
        units = tuple(str(u) for u in spec.get("units") or ())
        for source in sources:
            if source not in SOURCES:
                problems.append(f"estrutura.yaml: perfil {name}: fonte desconhecida {source!r}")
        for unit in units:
            if unit not in kinds:
                problems.append(f"estrutura.yaml: perfil {name}: tipo de unidade não declarado {unit!r}")
        profiles[str(name)] = Profile(name=str(name), sources=sources, units=units)
    return profiles


def _by_type(raw: object, profiles: Mapping[str, Profile], problems: list[str]) -> dict[str, str]:
    if not isinstance(raw, Mapping) or not raw:
        problems.append("estrutura.yaml: `profile_by_type` vazio")
        return {}
    for kind, profile in raw.items():
        if profile not in profiles:
            problems.append(f"estrutura.yaml: o tipo {kind!r} aponta para perfil inexistente {profile!r}")
    return {str(k): str(v) for k, v in raw.items()}


def _rules(
    raw: object, profiles: Mapping[str, Profile], kinds: Mapping[str, UnitKindSpec], problems: list[str]
) -> list[Rule]:
    rules: list[Rule] = []
    items: Sequence[object] = raw if isinstance(raw, list) else []
    for number, item in enumerate(items, start=1):
        where = f"estrutura.yaml, exceção {number}"
        if not isinstance(item, Mapping) or not (item.get("match") or item.get("folder")):
            problems.append(f"{where}: precisa de `match` ou `folder`")
            continue
        profile = str(item.get("profile"))
        if profile not in profiles:
            problems.append(f"{where}: perfil inexistente {profile!r}")
        branches = _branches(item.get("branches"), where, problems)
        units = tuple(u for u in (_unit(u, where, kinds, problems) for u in item.get("units") or ()) if u)
        sources = _strings(item.get("sources")) if item.get("sources") is not None else None
        for source in sources or ():
            if source not in SOURCES:
                problems.append(f"{where}: fonte desconhecida {source!r}")
        rules.append(
            Rule(
                match=_strings(item.get("match")),
                folder=_strings(item.get("folder")),
                profile=profile,
                branches=branches,
                units=units,
                sources=sources,
            )
        )
    return rules


def _unit(raw: object, where: str, kinds: Mapping[str, UnitKindSpec], problems: list[str]) -> UnitDeclaration | None:
    if not isinstance(raw, Mapping) or not isinstance(raw.get("kind"), str):
        problems.append(f"{where}: unidade sem `kind`")
        return None
    kind = str(raw["kind"])
    if kind not in kinds:
        problems.append(f"{where}: tipo de unidade não declarado {kind!r}")
    parent = _opt_text(raw.get("parent"))
    if parent is not None and parent not in kinds:
        problems.append(f"{where}: `parent` não declarado {parent!r}")
    pages: tuple[int, int] | None = None
    if raw.get("pages") is not None:
        match = _PAGES.match(str(raw["pages"]))
        if match is None or int(match.group(1)) > int(match.group(2)):
            problems.append(f"{where}: faixa de páginas inválida {raw['pages']!r} (esperado `início-fim`)")
        else:
            pages = (int(match.group(1)), int(match.group(2)))
    pattern = _opt_text(raw.get("pattern"))
    if pattern is not None:
        try:
            re.compile(pattern)
        except re.error as exc:
            problems.append(f"{where}: padrão inválido {pattern!r}: {exc}")
    trim = _opt_text(raw.get("trim"))
    if trim is not None:
        try:
            re.compile(trim)
        except re.error as exc:
            problems.append(f"{where}: padrão inválido {trim!r}: {exc}")
    source = _opt_text(raw.get("source"))
    if source is not None and source not in SOURCES:
        problems.append(f"{where}: fonte desconhecida {source!r}")
    return UnitDeclaration(
        kind=kind,
        pages=pages,
        source=source,
        parent=parent,
        name=_opt_text(raw.get("name")),
        branches=_branches(raw.get("branches"), where, problems),
        pattern=pattern,
        under=_strings(raw.get("under")),
        trim=trim,
    )


def _branches(raw: object, where: str, problems: list[str]) -> tuple[str, ...]:
    values = _strings(raw)
    for value in values:
        if value not in BRANCHES:
            problems.append(f"{where}: ramo desconhecido {value!r} (esperado um de {', '.join(BRANCHES)})")
    return values


def _strings(raw: object) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        return (raw,)
    if isinstance(raw, Sequence):
        return tuple(str(v) for v in raw)
    return (str(raw),)


def _opt_text(raw: object) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip()
    return None if not text or text == "-" else text
