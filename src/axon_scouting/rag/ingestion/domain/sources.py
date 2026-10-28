"""Fontes da árvore que o texto do trecho não guarda: o sumário e os bookmarks do PDF (spec 102 FR-026; ADR-048).

O sumário dá a página impressa; a do PDF pode diferir (capa e expediente sem número). O deslocamento é o que mais
alinha os títulos do sumário a uma linha das páginas apontadas, entre −`MAX_OFFSET` e +`MAX_OFFSET`; sem alinhamento
nenhum, o sumário não é usado — melhor sem árvore que com a árvore no lugar errado.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from axon_scouting.rag.ingestion.domain.text import fold, toc_pages

MAX_OFFSET: Final = 6
MIN_ALIGNED: Final = 0.3
"""Fração das entradas que precisam alinhar no deslocamento escolhido."""
_ORDINAL: Final = re.compile(r"^\d{1,3}\s*[ªaº°]$")
"""O ordinal que o sumário põe numa linha sozinho ("1a" / "Reunião de Alcateia ...... 14")."""
_ENTRY: Final = re.compile(
    r"^(?P<title>.*?\S)\s*(?:(?:\.\s?){3,}|(?:…\s*){2,}|(?:-\s?){4,}|\s{2,})\s*(?P<page>\d{1,4})$"
)


@dataclass(frozen=True, slots=True, kw_only=True)
class TocEntry:
    title: str
    page: int
    """Página do PDF (a primeira é a 1), já com o deslocamento."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Bookmark:
    level: int
    """0 = o nível mais alto."""
    title: str
    page: int


def toc_entries(pages: Sequence[str]) -> tuple[TocEntry, ...]:
    """As entradas do sumário com página do PDF, na ordem do sumário; vazio sem sumário ou sem alinhamento."""
    raw: list[tuple[str, str, int]] = []  # (título com o ordinal, título para alinhar, página impressa)
    for number in toc_pages(pages):
        ordinal = ""
        for line in pages[number - 1].splitlines():
            text = " ".join(line.split())
            match = _ENTRY.match(text)
            if match is None:
                ordinal = text if _ORDINAL.match(text) else ("" if text else ordinal)
                continue
            title = match.group("title").strip(" .-–·…")
            if any(ch.isalpha() for ch in title):
                shown = f"{ordinal} {title}" if ordinal else title
                if not raw or raw[-1][0] != shown:  # a mesma unidade repetida em seguida (outra página dela)
                    raw.append((shown, title, int(match.group("page"))))
            ordinal = ""
    if not raw:
        return ()
    offset = _offset([(title, page) for _, title, page in raw], pages)
    if offset is None:
        return ()
    return tuple(
        TocEntry(title=shown, page=page + offset) for shown, _, page in raw if 1 <= page + offset <= len(pages)
    )


def bookmarks_from(raw: Sequence[tuple[int, str, int]], page_count: int) -> tuple[Bookmark, ...]:
    """Os bookmarks do parser (nível, título, página), sem os vazios e os de página fora do documento."""
    return tuple(
        Bookmark(level=level, title=" ".join(title.split()), page=page)
        for level, title, page in raw
        if title.strip() and 1 <= page <= page_count
    )


def _offset(entries: Sequence[tuple[str, int]], pages: Sequence[str]) -> int | None:
    folded_pages = [{fold(line) for line in page.splitlines() if line.strip()} for page in pages]
    hits: Counter[int] = Counter()
    for title, printed in entries:
        wanted = fold(title)
        for delta in range(-MAX_OFFSET, MAX_OFFSET + 1):
            page = printed + delta
            if 1 <= page <= len(pages) and any(
                line == wanted or line.startswith(wanted) or line.endswith(wanted) for line in folded_pages[page - 1]
            ):
                hits[delta] += 1
    if not hits:
        return None
    delta, count = max(hits.items(), key=lambda item: (item[1], -abs(item[0])))
    return delta if count >= max(2, MIN_ALIGNED * len(entries)) else None
