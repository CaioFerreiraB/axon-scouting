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
MIN_PAGES_FOR_RUNNING: Final = 3


@dataclass(frozen=True, slots=True)
class Line:
    text: str
    page: int


def clean_pages(pages: Sequence[str]) -> list[Line]:
    per_page = [_page_lines(page) for page in pages]
    running = _running_lines(per_page)
    lines: list[Line] = []
    for number, page_lines in enumerate(per_page, start=1):
        content = _without_running(page_lines, running)
        lines.extend(Line(text=text, page=number) for text in content)
    return _collapse_blanks(_join_hyphenated(lines))


def _page_lines(page: str) -> list[str]:
    lines = [" ".join(raw.split()) for raw in page.splitlines()]
    return [line for line in lines if not _PAGE_NUMBER.match(line)]


def _key(line: str) -> str:
    return re.sub(r"\d+", "#", line)


def _edges(lines: list[str]) -> list[str]:
    content = [line for line in lines if line]
    return [] if not content else list(dict.fromkeys((content[0], content[-1])))


def _running_lines(per_page: list[list[str]]) -> set[str]:
    """Linhas (com dígitos neutralizados) no topo ou no rodapé de mais da metade das páginas, com no mínimo 3."""
    if len(per_page) < MIN_PAGES_FOR_RUNNING:
        return set()
    counts = Counter(_key(line) for lines in per_page for line in _edges(lines))
    return {key for key, n in counts.items() if n >= MIN_PAGES_FOR_RUNNING and n > len(per_page) / 2}


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
