"""Estrutura normativa: uma unidade (artigo ou regra) = um chunk (spec 102 FR-002, FR-003; research R7).

A tabela de marcadores é dado do módulo e cresce com o corpus real (SC-005). Marcadores casam sem distinguir caixa nem
acento, sempre no começo da linha. Parágrafos (`§`), incisos e alíneas não cortam: ficam no chunk do artigo. Uma
unidade só começa no início de parágrafo — "nos termos do\\nArt. 5º do Estatuto" é citação, não artigo. Texto fora de
unidade (preâmbulo, considerandos, anexos) vai para janelas narrativas.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Final, Literal

from axon_scouting.rag.index.api import DocumentType
from axon_scouting.rag.ingestion.domain.chunking import (
    ChunkDraft,
    ChunkingPolicy,
    is_stub,
    narrative_chunks,
    paragraph_text,
)
from axon_scouting.rag.ingestion.domain.ids import slug
from axon_scouting.rag.ingestion.domain.text import Line

_NUMBER: Final = r"([ivxlcdm]+|\d+)"
_REST: Final = r"(?![a-z0-9])\s*(?:[–—:.-]\s*(.*))?$"
"""Depois do número do agrupamento: nada, ou um separador e o nome."""


@dataclass(frozen=True, slots=True)
class _Grouping:
    level: int  # 0 anexo, 1 título, 2 capítulo, 3 seção, 4 subseção
    word: str
    pattern: re.Pattern[str]


GROUPINGS: Final = (
    _Grouping(0, "Anexo", re.compile(rf"^anexo\s+{_NUMBER}{_REST}")),
    _Grouping(1, "Título", re.compile(rf"^titulo\s+{_NUMBER}{_REST}")),
    _Grouping(2, "Capítulo", re.compile(rf"^capitulo\s+{_NUMBER}{_REST}")),
    _Grouping(3, "Seção", re.compile(rf"^secao\s+{_NUMBER}{_REST}")),
    _Grouping(4, "Subseção", re.compile(rf"^subsecao\s+{_NUMBER}{_REST}")),
)
_ANNEXES: Final = re.compile(r"^anexos?$")
"""Seção de anexos sem número ("ANEXOS", no POR)."""
ARTICLE: Final = re.compile(r"^art(?:igo)?\.?\s*(\d+)(?:-([a-z])(?![a-z]))?(?![0-9])")
RULE: Final = re.compile(r"^regra\s+(\d+)(?![0-9])")
MAX_HEADING: Final = 80
UnitKind = Literal["article", "rule"]
UNIT_BY_TYPE: Final[dict[DocumentType, UnitKind]] = {
    DocumentType.POR: "rule",
    DocumentType.ESTATUTO: "article",
    DocumentType.RESOLUCAO: "article",
}
"""A unidade de cada tipo normativo: a resolução cita regras do POR dentro dos artigos (corpus real, SC-005)."""
_SPLIT_MARKERS: Final = frozenset({"regra", "art", "art.", "artigo"})
"""Marcador que o PDF põe sozinho na linha, com o número na seguinte ("REGRA" / "001 DEFINIÇÃO…", no POR)."""
_SENTENCE_END: Final = (".", ":", ";", "!", "?")
_OPEN_ENDINGS: Final = frozenset(
    {"a", "à", "ao", "com", "da", "das", "de", "do", "dos", "e", "em", "na", "nas", "no", "nos", "o", "para"}
)
"""Nome de agrupamento que termina assim continua na linha seguinte ("Segurança nas" / "Atividades Escoteiras")."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Heading:
    level: int
    label: str  # "Capítulo III"
    name: str | None  # "Dos Associados"

    @property
    def metadata(self) -> str:
        return f"{self.label} – {self.name}" if self.name else self.label


@dataclass(frozen=True, slots=True, kw_only=True)
class Unit:
    key: str  # "art42", "regra33"
    label: str  # "Art. 42", "Regra 33"


@dataclass(frozen=True, slots=True, kw_only=True)
class NormativeResult:
    """Os chunks e o número de unidades; zero unidades = o documento vai inteiro para janelas (fallback)."""

    drafts: list[ChunkDraft]
    units: int


@dataclass
class _State:
    title: str
    policy: ChunkingPolicy
    path: dict[int, Heading] = field(default_factory=dict)
    drafts: list[ChunkDraft] = field(default_factory=list)
    units: int = 0
    loose_runs: int = 0
    unit: Unit | None = None
    unit_path: dict[int, Heading] = field(default_factory=dict)
    buffer: list[Line] = field(default_factory=list)
    carried: list[Line] = field(default_factory=list)
    """Texto solto curto e sem frase entre duas unidades (o nome de uma seção que não se reconheceu): entra na unidade
    seguinte, depois do título dela (research R28)."""

    def labels(self, path: dict[int, Heading]) -> list[str]:
        """O caminho no prefixo, com o nome de cada agrupamento ("Capítulo 8 – Ramo Filhotes"; research R28)."""
        return [path[level].metadata for level in sorted(path)]

    def flush(self, *, last: bool = False) -> None:
        if not any(line.text for line in self.buffer):
            self.buffer = []
            self.unit = None
            return
        if self.unit is None and not last and is_stub(self.buffer):
            self.carried = [*self.carried, *self.buffer]
        elif self.unit is not None:
            self._flush_unit(self.unit)
        else:
            self.buffer[:0], self.carried = self.carried, []
            self._flush_loose()
        self.buffer = []
        self.unit = None

    def _flush_unit(self, unit: Unit) -> None:
        path = self.unit_path
        lines = [line for line in self.buffer if line.text]
        head, rest = lines[0], self.buffer[self.buffer.index(lines[0]) + 1 :]
        if self.carried:  # parágrafo próprio, depois do título da unidade
            rest, self.carried = [*self.carried, Line(text="", page=self.carried[-1].page), *rest], []
        body = (
            f"{head.text}\n{paragraph_text(rest)}".rstrip()  # título da regra em maiúsculas é parágrafo próprio
            if _is_caps(head.text)
            else paragraph_text([head, *rest])
        )
        self.drafts.append(
            ChunkDraft(
                section_key=unit.key,
                prefix="[" + " · ".join([self.title, *self.labels(path)]) + "]",
                body=body,
                page_start=lines[0].page,
                page_end=lines[-1].page,
                chapter=_meta(path, 2),
                section=_meta(path, 3),
                subsection=_meta(path, 4),
                article=unit.label,
                normative=True,
            )
        )

    def _flush_loose(self) -> None:
        if 0 in self.path:
            key = slug(self.path[0].label)
        elif self.units == 0:
            key = "preambulo"
        else:
            self.loose_runs += 1
            key = f"s{self.loose_runs}"
        self.drafts.extend(
            narrative_chunks(self.buffer, self.title, self.policy, section_key=key, context=self.labels(self.path))
        )


def normative_chunks(lines: Sequence[Line], title: str, policy: ChunkingPolicy, *, unit: UnitKind) -> NormativeResult:
    state = _State(title=title, policy=policy)
    pending_name: Heading | None = None
    named: Heading | None = None
    previous = ""
    for line in _merge_split_markers(lines):
        text = line.text
        folded = _fold(text)
        if named is not None:
            open_name = (named.name or "").split()[-1:] in ([word] for word in _OPEN_ENDINGS)
            continues = text[:1].islower() or open_name or (_is_caps(named.name or "") and _is_caps(text))
            if continues and not _is_marker(folded) and len(text) <= MAX_HEADING and not text.endswith(_SENTENCE_END):
                named = Heading(level=named.level, label=named.label, name=f"{named.name} {text}")
                _set_heading(state, named)  # nome em duas linhas: "Orientação Patrimonial" / "e Financeira"
                continue
            named = None
        if pending_name is not None:
            if text and not _is_marker(folded) and len(text) <= MAX_HEADING and not text.endswith(_SENTENCE_END):
                named = Heading(level=pending_name.level, label=pending_name.label, name=text)
                _set_heading(state, named)
                pending_name = None
                previous = ""  # depois de um cabeçalho, a próxima linha começa parágrafo
                continue
            pending_name = None
        grouping = _grouping(text, folded)
        named = None
        if grouping is not None:
            state.flush()
            _set_heading(state, grouping)
            pending_name = grouping if grouping.name is None else None
            named = grouping if grouping.name is not None else None  # o nome na mesma linha também continua
            previous = ""
            continue
        found = _unit(folded, line.text, unit)
        starts_paragraph = not previous or previous.endswith(_SENTENCE_END) or _is_caps(previous)
        if found is not None and (starts_paragraph or _is_caps(text)):
            state.flush()
            state.units += 1
            state.unit = found
            state.unit_path = dict(state.path)
        state.buffer.append(line)
        if text:
            previous = text
    state.flush()
    if state.carried:  # sobra no fim: vira texto solto, como antes
        state.buffer, state.carried = state.carried, []
        state.flush(last=True)
    if state.units == 0:
        return NormativeResult(drafts=[], units=0)
    return NormativeResult(drafts=state.drafts, units=state.units)


def _set_heading(state: _State, heading: Heading) -> None:
    for level in [lv for lv in state.path if lv >= heading.level]:
        del state.path[level]
    state.path[heading.level] = heading


def _grouping(text: str, folded: str) -> Heading | None:
    if len(text) > MAX_HEADING:
        return None
    if _ANNEXES.match(folded.strip()):
        return Heading(level=0, label=text.strip().capitalize(), name=None)
    for grouping in GROUPINGS:
        match = grouping.pattern.match(folded)
        if match:
            number = match.group(1)
            label = f"{grouping.word} {number.upper() if not number.isdigit() else number}"
            name = _original_rest(text, match) if match.group(2) else None
            return Heading(level=grouping.level, label=label, name=name or None)
    return None


def _original_rest(text: str, match: re.Match[str]) -> str:
    """O nome no texto original (o dobrado perde acentos e caixa); `_fold` preserva as posições."""
    return text[match.start(2) :].strip()


def _unit(folded: str, text: str, kind: UnitKind) -> Unit | None:
    if kind == "article" and (match := ARTICLE.match(folded)):
        number, suffix = int(match.group(1)), match.group(2)
        return Unit(
            key=f"art{number}" + (f"-{suffix}" if suffix else ""),
            label=f"Art. {number}" + (f"-{suffix.upper()}" if suffix else ""),
        )
    if kind == "rule" and (match := RULE.match(folded)):
        digits = match.group(1)  # como o documento escreve: "Regra 033"
        return Unit(key=f"regra{int(digits)}", label=f"Regra {digits}")
    return None


def _merge_split_markers(lines: Sequence[Line]) -> list[Line]:
    merged: list[Line] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        following = lines[index + 1] if index + 1 < len(lines) else None
        if following is not None and _fold(line.text).strip() in _SPLIT_MARKERS and following.text[:1].isdigit():
            merged.append(Line(text=f"{line.text} {following.text}", page=line.page))
            index += 2
            continue
        merged.append(line)
        index += 1
    return merged


def _is_marker(folded: str) -> bool:
    return any(g.pattern.match(folded) for g in GROUPINGS) or bool(ARTICLE.match(folded) or RULE.match(folded))


def _is_caps(text: str) -> bool:
    letters = [ch for ch in text if ch.isalpha()]
    return len(letters) >= 3 and all(ch.isupper() for ch in letters)


def _meta(path: dict[int, Heading], level: int) -> str | None:
    heading = path.get(level)
    return None if heading is None else heading.metadata


def _fold(text: str) -> str:
    """Minúsculas e sem acento, um caractere por caractere do original (as posições se preservam)."""
    return "".join(_fold_char(ch) for ch in text)


def _fold_char(ch: str) -> str:
    base = unicodedata.normalize("NFKD", ch.lower())
    stripped = "".join(c for c in base if not unicodedata.combining(c))
    return stripped[:1] if stripped else ch
