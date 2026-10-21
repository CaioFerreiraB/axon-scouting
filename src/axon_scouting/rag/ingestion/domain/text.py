"""Limpeza do texto extraído das páginas, antes da estrutura e das janelas (spec 102 research R7).

O original preservado é o texto bruto das páginas (FR-009); a limpeza só alimenta o chunking. Cada linha guarda a
página de origem; linha vazia marca fronteira de parágrafo.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

_PAGE_NUMBER: Final = re.compile(
    r"^(?:-\s*)?(?:p[áa]g(?:ina)?\.?\s*)?\d{1,4}(?:\s*(?:/|de)\s*\d{1,4})?(?:\s*-)?$", re.IGNORECASE
)
_HYPHENATED: Final = re.compile(r"[^\W\d_]-$")
_PHOTO_CREDIT: Final = re.compile(r"(?<=\w)@\s+[A-ZÀ-Ý][\w'. -]*$")
"""Crédito de foto colado ao fim da linha pela diagramação ("…Escotismo@ Guillerme Ferreira Heck")."""
_TOC_ENTRY: Final = re.compile(r"(?:\.\s?){4,}\s*\d+$|(?:…\s*){2,}\d+$")
"""Linha de índice: pontilhado até o número da página."""
MIN_PAGES_FOR_RUNNING: Final = 3
MIN_TOC_ENTRIES: Final = 3
TOC_SHARE: Final = 0.3
"""Página com ao menos 3 linhas de índice, e elas são 30% das linhas: é página de índice e sai inteira."""


@dataclass(frozen=True, slots=True)
class Line:
    text: str
    page: int


def clean_pages(pages: Sequence[str]) -> list[Line]:
    per_page = [[] if _is_toc(page) else _page_lines(page) for page in pages]
    running = _running_lines(per_page)
    lines: list[Line] = []
    for number, page_lines in enumerate(per_page, start=1):
        content = _without_running(page_lines, running)
        lines.extend(Line(text=text, page=number) for text in content)
    return _collapse_blanks(_join_hyphenated(lines))


def _page_lines(page: str) -> list[str]:
    lines = [_PHOTO_CREDIT.sub("", " ".join(raw.split())).rstrip() for raw in page.splitlines()]
    return [line for line in lines if not _PAGE_NUMBER.match(line) and not _TOC_ENTRY.search(line)]


def _is_toc(page: str) -> bool:
    lines = [" ".join(raw.split()) for raw in page.splitlines() if raw.strip()]
    entries = sum(1 for line in lines if _TOC_ENTRY.search(line))
    return entries >= MIN_TOC_ENTRIES and entries >= TOC_SHARE * len(lines)


def _key(line: str) -> str:
    """Sem o número nas pontas (cabeçalho que alterna o lado da página) e com os demais dígitos neutralizados."""
    tokens = line.split()
    while len(tokens) > 1 and tokens[0].isdigit():
        tokens = tokens[1:]
    while len(tokens) > 1 and tokens[-1].isdigit():
        tokens = tokens[:-1]
    return re.sub(r"\d+", "#", " ".join(tokens))


def _edges(lines: list[str]) -> list[str]:
    content = [line for line in lines if line]
    return [] if not content else list(dict.fromkeys((content[0], content[-1])))


def _running_lines(per_page: list[list[str]]) -> set[str]:
    """Linhas (com dígitos neutralizados) no topo ou no rodapé de no mínimo 3 páginas, e: em mais da metade delas, ou
    sempre trazendo o número da própria página (o cabeçalho corrido que muda por capítulo)."""
    if len(per_page) < MIN_PAGES_FOR_RUNNING:
        return set()
    counts: Counter[str] = Counter()
    numbered: dict[str, bool] = {}
    for number, lines in enumerate(per_page, start=1):
        for line in _edges(lines):
            key = _key(line)
            counts[key] += 1
            tokens = line.split()
            carries = len(tokens) > 1 and str(number) in (tokens[0], tokens[-1])
            numbered[key] = numbered.get(key, True) and carries
    return {key for key, n in counts.items() if n >= MIN_PAGES_FOR_RUNNING and (n > len(per_page) / 2 or numbered[key])}


def _without_running(lines: list[str], running: set[str]) -> list[str]:
    if not running:
        return lines
    edges = set(_edges(lines))
    return [line for line in lines if not (line in edges and _key(line) in running)]


def _join_hyphenated(lines: list[Line]) -> list[Line]:
    joined: list[Line] = []
    for line in lines:
        previous = joined[-1] if joined else None
        if previous and line.text[:1].islower() and _HYPHENATED.search(previous.text):
            joined[-1] = Line(text=previous.text[:-1] + line.text, page=previous.page)
        else:
            joined.append(line)
    return joined


def _collapse_blanks(lines: list[Line]) -> list[Line]:
    result: list[Line] = []
    for line in lines:
        if not line.text and (not result or not result[-1].text):
            continue
        result.append(line)
    while result and not result[-1].text:
        result.pop()
    return result
