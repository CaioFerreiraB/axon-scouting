"""A árvore de unidades de um documento (spec 102 FR-025 a FR-028; ADR-048; research R34).

Função pura de (trechos, perfil, fontes): cada trecho recebe o caminho das unidades acima dele, o rótulo com o tipo e o
título de cada nível, a forma (`prose`/`list`) e os ramos; o documento recebe a árvore em pré-ordem. O tamanho dos
trechos não muda — a janela continua sendo o que se busca; a unidade é um nó.

As fontes, por perfil: os marcadores normativos (o `article` e os agrupamentos do trecho), os títulos detectados
(`chapter`/`section`/`subsection`), o sumário e os bookmarks (o capítulo pela página), o rodapé corrido (o eixo e o
grupo) e as unidades declaradas no `estrutura.yaml` (faixa de páginas do sumário, grupo nomeado com ramos, padrão de
título).
As unidades `under` (competência, cargo; ADR-050) são os títulos logo abaixo de um título nomeado na declaração
("COMPETÊNCIAS ESSENCIAIS" → cada competência; "LINHA ESCOTISTA" → cada cargo). `unit_boundary` diz ao chunking quais
títulos abrem unidade, para a junção de seções pequenas não engolir uma especialidade, um bloco ou um cargo.
"""

from __future__ import annotations

import dataclasses
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from statistics import median
from typing import Final

from axon_scouting.rag.index.api import Shape, UnitNode, UnitRef
from axon_scouting.rag.ingestion.domain.chunking import ChunkDraft
from axon_scouting.rag.ingestion.domain.corpus_structure import (
    NORMATIVE_PROFILES,
    Resolution,
    UnitDeclaration,
    canonical_axis,
    canonical_block,
)
from axon_scouting.rag.ingestion.domain.ids import slug
from axon_scouting.rag.ingestion.domain.sources import Bookmark, TocEntry
from axon_scouting.rag.ingestion.domain.text import fold

TYPED: Final = frozenset(
    {
        "eixo",
        "bloco",
        "especialidade",
        "insignia",
        "grupo_de_insignias",
        "competencia",
        "reuniao",
        "cargo",
        "item_de_pauta",
        "pergunta",
        "nivel",
    }
)
"""Tipos que entram no rótulo como "Tipo: título"; os demais (capítulo, seção) só com o título, e os que o corpo já
diz (artigo, regra, documento, slide), sem rótulo."""
_UNLABELED: Final = frozenset({"artigo", "regra", "paragrafo", "documento", "slide"})
OUTLINE_DEPTH: Final = 2
MIN_OUTLINE_UNITS: Final = 4
OUTLINE_TOKENS: Final = 300
"""O sumário de cada documento (ADR-048): os dois primeiros níveis da árvore, com página, em janelas de até 300 tokens
estimados — uma lista de títulos rende mais tokens do e5 por caractere que a prosa; com menos de 4 unidades, nenhum."""
_OUTLINE_SKIP: Final = frozenset({"documento", "slide", "paragrafo", "artigo", "regra"})
"""Fora do sumário: o que tem por título um número (artigo, regra) ou é o documento inteiro — "Art. 1, Art. 2…" não diz
de que trata a norma; os capítulos e as seções dela, sim."""
MIN_LIST_ITEMS: Final = 4
MAX_ITEM_WORDS: Final = 12
"""Trecho `list`: ao menos 4 itens marcados, e o item mediano com até 12 palavras (FR-027)."""
_MARKER: Final = re.compile(r"(?:^|(?<=\s))(?:•|[IVXLC]{1,6}\s?[-–—]|[a-z]\)|\d{1,2}[.)])\s+")
_PROGRESSION: Final = "progress"
_NUMBERED_UNIT: Final = re.compile(r"^\d{1,2}\s*[).]\s")
PROFILE_UNITS: Final = {"ata": "item_de_pauta", "perguntas": "pergunta"}
"""O perfil cujas linhas de unidade (`chunking.UNIT_LINES`) viram títulos numerados → o tipo deles na árvore."""


class TreeProblem(ValueError):
    """A declaração não casa com o documento (faixa sem entrada no sumário): o documento é recusado (FR-024)."""


@dataclass(frozen=True, slots=True, kw_only=True)
class PlacedChunk:
    draft: ChunkDraft
    """O trecho com o rótulo novo."""
    path: tuple[UnitRef, ...]
    unit_kind: str | None
    shape: Shape
    branches: tuple[str, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class DocumentTree:
    chunks: tuple[PlacedChunk, ...]
    units: tuple[UnitNode, ...]


@dataclass(frozen=True, slots=True)
class _Level:
    kind: str
    title: str
    branches: tuple[str, ...] = ()
    anchor: int | None = None
    """A página da entrada do sumário ou do bookmark: duas entradas de mesmo título são unidades distintas."""


@dataclass
class _Node:
    unit_id: str
    kind: str
    title: str
    parent: str | None
    branches: tuple[str, ...]
    page_start: int
    page_end: int
    children: list[str] = field(default_factory=list)


def build_tree(
    drafts: Sequence[ChunkDraft],
    resolution: Resolution,
    *,
    doc_key: str,
    title: str,
    branch: str | None,
    toc: Sequence[TocEntry] = (),
    bookmarks: Sequence[Bookmark] = (),
    running: Sequence[tuple[str, ...]] = (),
) -> DocumentTree:
    profile = resolution.profile
    sources = set(profile.sources)
    rule = resolution.rule
    declared = rule.units if rule is not None else ()
    ranges = _declared_ranges(declared, toc if "sumário" in sources else ())
    tops = _tops(sources, toc, bookmarks, ranges)
    default_branches = (rule.branches if rule is not None and rule.branches else ()) or ((branch,) if branch else ())
    builder = _Builder(doc_key)
    placed: list[PlacedChunk] = []
    for draft in drafts:
        labels_on_page = running[draft.page_start - 1] if 0 < draft.page_start <= len(running) else ()
        levels = _levels(draft, resolution, title, tops, ranges, declared, labels_on_page)
        nodes = builder.place(levels, draft.page_start, draft.page_end, default_branches)
        path = tuple(UnitRef(unit_id=n.unit_id, kind=n.kind, title=n.title) for n in nodes)
        branches = next((n.branches for n in reversed(nodes) if n.branches), default_branches)
        prefix = draft.prefix if profile.name in NORMATIVE_PROFILES else _prefix(draft, title, nodes, resolution)
        placed.append(
            PlacedChunk(
                draft=dataclasses.replace(draft, prefix=prefix),
                path=path,
                unit_kind=nodes[-1].kind if nodes else None,
                shape=shape_of(draft.body),
                branches=branches,
            )
        )
    units = builder.units()
    placed.extend(_outline(units, title, resolution, default_branches))
    return DocumentTree(chunks=tuple(placed), units=units)


def _outline(
    units: Sequence[UnitNode], title: str, resolution: Resolution, branches: tuple[str, ...]
) -> list[PlacedChunk]:
    """Os trechos de sumário: um item por unidade dos dois primeiros níveis, o de baixo recuado, com a página."""
    depth: dict[str, int] = {}
    entries: list[tuple[UnitNode, int]] = []
    for unit in units:
        level = depth[unit.unit_id] = 1 if unit.parent is None else depth.get(unit.parent, 1) + 1
        if level <= OUTLINE_DEPTH and unit.kind not in _OUTLINE_SKIP:
            entries.append((unit, level))
    if len(entries) < MIN_OUTLINE_UNITS:
        return []
    windows: list[list[tuple[UnitNode, str]]] = [[]]
    size = 0
    for unit, level in entries:
        spec = resolution.unit_kinds.get(unit.kind)
        name = (
            f"{spec.label if spec else unit.kind.replace('_', ' ').capitalize()}: {unit.title}"
            if unit.kind in TYPED
            else unit.title
        )
        line = ("  " * (level - 1)) + f"- {name} (p. {unit.page_start})"
        cost = math.ceil(len(line) / 4)
        if windows[-1] and size + cost > OUTLINE_TOKENS:
            windows.append([])
            size = 0
        windows[-1].append((unit, line))
        size += cost
    if len(windows) > 1 and size < OUTLINE_TOKENS // 3:  # a janela final curta junta-se à anterior
        windows[-2].extend(windows.pop())
    return [
        PlacedChunk(
            draft=ChunkDraft(
                section_key="sumario",
                prefix=f"[{title} · Sumário]",
                body="\n".join(line for _, line in window),
                page_start=min(unit.page_start for unit, _ in window),
                page_end=max(unit.page_end for unit, _ in window),
            ),
            path=(),
            unit_kind=None,
            shape="outline",
            branches=branches,
        )
        for window in windows
    ]


def shape_of(body: str) -> Shape:
    """`list` quando o corpo são itens marcados (•, inciso, alínea, numeração) e curtos (FR-027)."""
    pieces = _MARKER.split(body)
    items = [piece for piece in pieces[1:] if piece.strip()]
    if len(items) < MIN_LIST_ITEMS:
        return "prose"
    return "list" if median(len(item.split()) for item in items) <= MAX_ITEM_WORDS else "prose"


# --- níveis de cada trecho ------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Range:
    declaration: UnitDeclaration
    entries: tuple[TocEntry, ...]

    def covers(self, page: int) -> bool:
        assert self.declaration.pages is not None
        start, end = self.declaration.pages
        return start <= page <= end

    def entry_at(self, page: int) -> TocEntry | None:
        current = None
        for entry in self.entries:
            if entry.page <= page:
                current = entry
        return current


def _declared_ranges(declared: Sequence[UnitDeclaration], toc: Sequence[TocEntry]) -> list[_Range]:
    ranges: list[_Range] = []
    for declaration in declared:
        if declaration.pages is None:
            continue
        start, end = declaration.pages
        entries = tuple(e for e in toc if start <= e.page <= end) if declaration.source == "sumário" else ()
        if declaration.source == "sumário" and not entries:
            raise TreeProblem(
                f"a faixa {start}-{end} de {declaration.kind} não tem entrada no sumário"
                + ("" if toc else " (sumário ausente ou desalinhado)")
            )
        ranges.append(_Range(declaration=declaration, entries=entries))
    return ranges


def _tops(
    sources: set[str], toc: Sequence[TocEntry], bookmarks: Sequence[Bookmark], ranges: Sequence[_Range]
) -> list[tuple[int, str]]:
    """O capítulo de cada página: os bookmarks do nível mais alto, senão as entradas do sumário fora das faixas
    declaradas (dentro delas, as entradas são as unidades)."""
    if "bookmarks" in sources and bookmarks:
        top = min(b.level for b in bookmarks)
        return sorted(((b.page, b.title) for b in bookmarks if b.level == top), key=lambda item: item[0])
    if "sumário" in sources and toc:
        inside = {(e.title, e.page) for r in ranges for e in r.entries}
        # na mesma página, vale a ordem do sumário ("Exemplos de reuniões" e "1a Reunião", ambos na p. 14)
        return sorted(((e.page, e.title) for e in toc if (e.title, e.page) not in inside), key=lambda item: item[0])
    return []


def _top_at(tops: Sequence[tuple[int, str]], page: int) -> tuple[int, str] | None:
    current = None
    for start, name in tops:
        if start <= page:
            current = (start, name)
    return current


def _levels(
    draft: ChunkDraft,
    resolution: Resolution,
    title: str,
    tops: Sequence[tuple[int, str]],
    ranges: Sequence[_Range],
    declared: Sequence[UnitDeclaration],
    running: tuple[str, ...],
) -> list[_Level]:
    profile = resolution.profile.name
    headings = list(draft.headings) or [h for h in (draft.chapter, draft.section, draft.subsection) if h]
    page = draft.page_start
    if profile in NORMATIVE_PROFILES:
        kinds = ["capitulo", "secao", "secao"]
        normative = [_Level(kind, h) for kind, h in zip(kinds, headings, strict=False)]
        if draft.article:
            normative.append(_Level("regra" if profile == "regras" else "artigo", draft.article))
        return normative
    if profile == "comunicado":
        return [_Level("documento", title), *(_Level("secao", h) for h in headings)]
    if profile == "apresentacao":
        return [_Level("slide", f"Slide {page}"), *(_Level("secao", h) for h in headings)]

    levels: list[_Level] = []
    covered: list[str] = []
    blocks: list[str] = []
    group = next((r for r in ranges if r.declaration.kind == "grupo_de_insignias" and r.covers(page)), None)
    unit = next((r for r in ranges if r.declaration.source == "sumário" and r.covers(page)), None)
    entry = unit.entry_at(page) if unit is not None else None
    patterns = [d for d in declared if d.pattern]
    unders = [d for d in declared if d.under]
    if not any(r.covers(page) for r in ranges):  # numa faixa declarada, o capítulo é a própria unidade
        top = _top_at(tops, page)
        if top is not None:
            anchor, name = top
            levels.append(_Level(_pattern_kind(name, patterns) or "capitulo", name, anchor=anchor))
            covered.append(name)
    if profile == "especialidades" and group is not None:
        name = group.declaration.name or "Grupo"
        levels.append(_Level("grupo_de_insignias", name, group.declaration.branches))
    if profile in ("progressao", "especialidades"):
        axis_block = _axis_and_block(headings, running, levels, resolution, profile)
        if axis_block is not None:
            axis, block = axis_block
            under_axis = entry is None or unit is None or unit.declaration.parent == "eixo"
            if axis is not None and under_axis and group is None:
                levels.append(_Level("eixo", axis))
                covered.append(axis)
            if block is not None and entry is None:
                levels.append(_Level("bloco", block))
                blocks.append(block)
    if entry is not None and unit is not None:
        levels.append(_Level(unit.declaration.kind, entry.title, anchor=entry.page))
        covered.append(entry.title)
    rest = [h for h in headings if not _covered(h, covered) and not _covered(h, blocks, prefix=True)]
    for index, heading in enumerate(rest):
        parent = rest[index - 1] if index else (levels[-1].title if levels else None)
        kind = _pattern_kind(heading, patterns) or _under_kind(parent, unders)
        if kind is not None and any(level.kind == kind for level in levels):
            kind = None  # "REUNIÃO DE TROPA" dentro da "1a Reunião de Tropa" é seção dela, não outra reunião
        if kind is not None:
            heading = _trimmed(heading, kind, declared)
            if not heading:
                kind, heading = None, rest[index]  # o título inteiro é o que o `trim` tira ("Nível", "Preliminar")
        if kind is None and index == 0 and profile in PROFILE_UNITS and _NUMBERED_UNIT.match(heading):
            kind = PROFILE_UNITS[profile]
        if kind is None:
            kind = "capitulo" if not levels and index == 0 else "secao"
        levels.append(_Level(kind, heading))
    return levels


def _axis_and_block(
    headings: Sequence[str],
    running: tuple[str, ...],
    levels: Sequence[_Level],
    resolution: Resolution,
    profile: str,
) -> tuple[str | None, str | None] | None:
    """O eixo pelo rodapé (ou pelo bloco) e o bloco pelo título que casa um bloco do vocabulário — este só no capítulo
    da Progressão Pessoal (o rodapé tem um eixo, ou o capítulo/rodapé fala de progressão), para "Valores" de outro
    capítulo não virar bloco."""
    axis = next((a for label in running if (a := canonical_axis(label, resolution))), None)
    if profile == "especialidades":
        return (axis, None) if axis else None
    context = " ".join([*running, *(level.title for level in levels if level.kind == "capitulo")])
    in_progression = axis is not None or _PROGRESSION in fold(context)
    block = None
    if in_progression:
        block = next((found for h in headings if (found := canonical_block(h, resolution))), None)
    if block is not None:  # o eixo do vocabulário vence o do rodapé, que na página de transição já é o seguinte
        return block
    return (axis, None) if axis else None


def _trimmed(title: str, kind: str, declared: Sequence[UnitDeclaration]) -> str:
    trim = next((d.trim for d in declared if d.kind == kind and d.trim), None)
    if trim is None:
        return title
    return re.sub(trim, "", title, flags=re.IGNORECASE).strip()


def _under_kind(parent: str | None, unders: Sequence[UnitDeclaration]) -> str | None:
    """O tipo declarado para os títulos logo abaixo de um dos títulos de `under` (comparação sem caixa nem acento,
    pelo começo: "COMPETÊNCIAS ESSENCIAIS" casa "Competências Essenciais")."""
    if parent is None:
        return None
    folded = fold(parent)
    return next((d.kind for d in unders if any(folded.startswith(fold(name)) for name in d.under)), None)


def unit_boundary(resolution: Resolution, toc: Sequence[TocEntry] = ()) -> Callable[[Sequence[str], int], bool]:
    """Diz se o título `path[index]` abre uma unidade com nome: entrada do sumário numa faixa declarada, bloco do
    vocabulário (na progressão), padrão de título declarado ou filho de um título de `under`."""
    rule = resolution.rule
    declared = rule.units if rule is not None else ()
    patterns = [d for d in declared if d.pattern]
    unders = [d for d in declared if d.under]
    entries = {
        fold(entry.title)
        for d in declared
        if d.pages is not None and d.source == "sumário"
        for entry in toc
        if d.pages[0] <= entry.page <= d.pages[1]
    }
    blocks = resolution.profile.name in ("progressao", "especialidades")

    def boundary(path: Sequence[str], index: int) -> bool:
        title = path[index]
        folded = fold(title)
        if folded in entries or _pattern_kind(title, patterns):
            return True
        if blocks and canonical_block(title, resolution) is not None:
            return True
        return index > 0 and _under_kind(path[index - 1], unders) is not None

    return boundary


def _pattern_kind(title: str, patterns: Sequence[UnitDeclaration]) -> str | None:
    return next((d.kind for d in patterns if re.search(d.pattern or "", title, re.IGNORECASE)), None)


def _covered(heading: str, covered: Sequence[str], *, prefix: bool = False) -> bool:
    """O título já é um nível da árvore: igual a ele ou, para os blocos (cujo nome varia no fim), começando por ele."""
    folded = fold(heading)
    return any(folded == fold(c) or (prefix and folded.startswith(fold(c))) for c in covered)


# --- nós e rótulo ---------------------------------------------------------------------------------------------------


class _Builder:
    def __init__(self, doc_key: str) -> None:
        self._doc_key = doc_key
        self._nodes: dict[str, _Node] = {}
        self._by_key: dict[tuple[str | None, str, str, int | None], str] = {}
        self._roots: list[str] = []

    def place(self, levels: Sequence[_Level], start: int, end: int, default_branches: tuple[str, ...]) -> list[_Node]:
        nodes: list[_Node] = []
        parent: _Node | None = None
        for level in levels:
            key = (parent.unit_id if parent else None, level.kind, fold(level.title), level.anchor)
            unit_id = self._by_key.get(key)
            if unit_id is None:
                unit_id = self._new_id(parent, level)
                branches = level.branches or (parent.branches if parent else default_branches)
                node = _Node(
                    unit_id=unit_id,
                    kind=level.kind,
                    title=level.title,
                    parent=parent.unit_id if parent else None,
                    branches=branches,
                    page_start=start,
                    page_end=end,
                )
                self._nodes[unit_id] = node
                self._by_key[key] = unit_id
                (parent.children if parent else self._roots).append(unit_id)
            node = self._nodes[unit_id]
            node.page_start, node.page_end = min(node.page_start, start), max(node.page_end, end)
            nodes.append(node)
            parent = node
        return nodes

    def _new_id(self, parent: _Node | None, level: _Level) -> str:
        segment = f"{level.kind}-{slug(level.title)[:48]}".rstrip("-")
        base = f"{parent.unit_id}/{segment}" if parent else f"{self._doc_key}#u/{segment}"
        unit_id, n = base, 1
        while unit_id in self._nodes:
            n += 1
            unit_id = f"{base}~{n}"
        return unit_id

    def units(self) -> tuple[UnitNode, ...]:
        ordered: list[UnitNode] = []

        def visit(unit_id: str) -> None:
            node = self._nodes[unit_id]
            ordered.append(
                UnitNode(
                    unit_id=node.unit_id,
                    kind=node.kind,
                    title=node.title,
                    page_start=node.page_start,
                    page_end=node.page_end,
                    parent=node.parent,
                    branches=node.branches,
                )
            )
            for child in node.children:
                visit(child)

        for root in self._roots:
            visit(root)
        return tuple(ordered)


def _prefix(draft: ChunkDraft, title: str, nodes: Sequence[_Node], resolution: Resolution) -> str:
    """O rótulo do trecho: o título do documento, cada nível da árvore e o que o rótulo antigo tinha além deles."""
    labels: list[str] = []
    for node in nodes:
        if node.kind in _UNLABELED:
            continue
        if node.kind in TYPED:
            spec = resolution.unit_kinds.get(node.kind)
            labels.append(f"{spec.label if spec else node.kind.replace('_', ' ').capitalize()}: {node.title}")
        else:
            labels.append(node.title)
    headings = {fold(h) for h in (draft.headings or (draft.chapter, draft.section, draft.subsection)) if h}
    titles = [node.title for node in nodes]
    old = draft.prefix.strip("[]").split(" · ")[1:]
    extra = [label for label in old if fold(label) not in headings and not _covered(label, titles)]
    return "[" + " · ".join([title, *labels, *extra]) + "]"
