"""Limpeza do texto extraído das páginas, antes da estrutura e das janelas (spec 102 research R7, R27).

O original preservado é o texto bruto das páginas (FR-009); a limpeza só alimenta o chunking. Cada linha guarda a
página de origem; linha vazia marca fronteira de parágrafo. Cabeçalho e rodapé corridos são procurados nas primeiras e
últimas linhas de cada página — eles vêm em blocos de várias linhas —, e o hífen de quebra que o PDFium marca (U+FFFE)
segue a regra do `-` no fim da linha.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

_PAGE_NUMBER: Final = re.compile(
    r"^(?:-\s*)?(?:p[áa]g(?:ina)?\.?\s*)?\d{1,4}(?:\s*(?:/|de)\s*\d{1,4})?(?:\s*-)?$", re.IGNORECASE
)
_HYPHENATED: Final = re.compile(r"[^\W\d_]-$")
_PHOTO_CREDIT: Final = re.compile(r"(?<=\w)@\s+[A-ZÀ-Ý][\w'. -]*$")
"""Crédito de foto colado ao fim da linha pela diagramação ("…Escotismo@ Guillerme Ferreira Heck")."""
_TOC_ENTRY: Final = re.compile(r"(?:\.\s?){4,}\s*\d+$|(?:…\s*){2,}\d+$|(?:-\s?){4,}\s*\d+$")
"""Linha de índice: pontilhado (ou tracejado, ADR-048) até o número da página."""
_SOFT_HYPHEN_AT_END: Final = re.compile(r"[\u00ad\ufffe](?=\r?\n)")
_SOFT_BREAK: Final = re.compile(r"\ufffe\s*(?=[A-ZÀ-Ý0-9\"“(])")
"""U+FFFE antes de maiúscula: o PDFium juntou duas linhas ("…DA UEB" + U+FFFE + "Realizada no dia…")."""
_SOFT_HYPHENS: Final = re.compile(r"[\u00ad\ufffe]\s*")
"""O resto: hífen de quebra no meio da linha que o PDFium juntou ("com" + U+FFFE + "põem") — some, como o `-` de
quebra."""
_GLUED_PAGE_NUMBER: Final = re.compile(r"^\d{1,4}(?=[A-ZÀ-Ý]{2})")
"""Número da página colado ao título em maiúsculas pela diagramação ("30IDENTIDADE E SÍMBOLOS")."""
_PROTECTED: Final = re.compile(r"^(?:regra|art\.?|artigo)(?:\s+\d.*)?$", re.IGNORECASE)
"""O marcador de unidade normativa sozinho na linha ("REGRA") abre muitas páginas do POR: nunca é linha corrida."""
MIN_PAGES_FOR_RUNNING: Final = 3
EDGE_LINES: Final = 4
"""Quantas linhas de cada borda da página podem ser cabeçalho ou rodapé corrido."""
SPAN_SHARE: Final = 0.8
"""Linha que aparece nas bordas de 80% das páginas entre a primeira e a última ocorrência é corrida (o título do
capítulo repetido nas páginas dele)."""
MIN_TOC_ENTRIES: Final = 3
TOC_SHARE: Final = 0.3
"""Página com ao menos 3 linhas de índice, e elas são 30% das linhas: é página de índice e sai inteira."""


@dataclass(frozen=True, slots=True)
class Line:
    text: str
    page: int


def clean_pages(pages: Sequence[str]) -> list[Line]:
    per_page = [[] if _is_toc(page) else _page_lines(page) for page in pages]
    running = _running_lines(per_page, pages)
    lines: list[Line] = []
    for number, page_lines in enumerate(per_page, start=1):
        content = _without_running(page_lines, running)
        lines.extend(Line(text=text, page=number) for text in content)
    return _collapse_blanks(_join_hyphenated(lines))


def _page_lines(page: str) -> list[str]:
    page = _SOFT_HYPHENS.sub("", _SOFT_BREAK.sub("\n", _SOFT_HYPHEN_AT_END.sub("-", page)))
    lines = [_unglued(_PHOTO_CREDIT.sub("", " ".join(raw.split())).rstrip()) for raw in page.splitlines()]
    return [line for line in lines if not _PAGE_NUMBER.match(line) and not _TOC_ENTRY.search(line)]


def _unglued(line: str) -> str:
    """Sem o número de página colado ao começo de um título em maiúsculas."""
    match = _GLUED_PAGE_NUMBER.match(line)
    if match is None:
        return line
    rest = line[match.end() :]
    letters = [ch for ch in rest if ch.isalpha()]
    return rest if letters and all(ch.isupper() for ch in letters) else line


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
    """Até `EDGE_LINES` primeiras e últimas linhas com texto — no máximo um terço das linhas da página, para que uma
    página curta não seja toda borda —, sem repetir; marcador de unidade nunca é borda."""
    content = [line for line in lines if line]
    size = min(EDGE_LINES, max(1, len(content) // 3))
    edges = content[:size] + content[-size:]
    return [line for line in dict.fromkeys(edges) if not _PROTECTED.match(line)]


def _running_lines(per_page: list[list[str]], raw_pages: Sequence[str] = ()) -> set[str]:
    """Linhas (com dígitos neutralizados) nas bordas de no mínimo 3 páginas, e: em mais da metade delas, em 80% das
    páginas entre a primeira e a última ocorrência (o título do capítulo nas páginas dele), ou sempre trazendo o número
    da própria página (o cabeçalho corrido que muda por capítulo) — inclusive colado a ele ("100MEIO AMBIENTE", o eixo
    no rodapé dos guias de progressão; ADR-050)."""
    if len(per_page) < MIN_PAGES_FOR_RUNNING:
        return set()
    flat = [" ".join(page.split()) for page in raw_pages]
    pages: dict[str, list[int]] = {}
    numbered: dict[str, bool] = {}
    for number, lines in enumerate(per_page, start=1):
        for key, line in dict((_key(line), line) for line in _edges(lines)).items():
            pages.setdefault(key, []).append(number)
            tokens = line.split()
            carries = len(tokens) > 1 and str(number) in (tokens[0], tokens[-1])
            if not carries and number <= len(flat):
                carries = f"{number}{line}" in flat[number - 1] or f"{line}{number}" in flat[number - 1]
            numbered[key] = numbered.get(key, True) and carries
    return {
        key
        for key, found in pages.items()
        if len(found) >= MIN_PAGES_FOR_RUNNING
        and (len(found) > len(per_page) / 2 or len(found) >= SPAN_SHARE * (found[-1] - found[0] + 1) or numbered[key])
    }


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


# --- estrutura que a limpeza tira do texto (ADR-048) ---------------------------------------------------------------


def fold(text: str) -> str:
    """Sem caixa, sem acento e com os espaços colapsados — para comparar títulos e rótulos."""
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return " ".join("".join(ch for ch in decomposed if not unicodedata.combining(ch)).split())


def toc_pages(pages: Sequence[str]) -> list[int]:
    """As páginas de índice (a primeira é a 1), que `clean_pages` descarta."""
    return [number for number, page in enumerate(pages, start=1) if _is_toc(page)]


def running_labels(pages: Sequence[str]) -> list[tuple[str, ...]]:
    """Para cada página, as linhas corridas que `clean_pages` tira das bordas, sem o número da página nas pontas
    ("24 Habilidades para a Vida" → "Habilidades para a Vida"): o eixo, o grupo ou o capítulo do rodapé."""
    per_page = [[] if _is_toc(page) else _page_lines(page) for page in pages]
    running = _running_lines(per_page, pages)
    labels: list[tuple[str, ...]] = []
    for page_lines in per_page:
        edges = _edges(page_lines) if running else []
        found = [_without_page_number(line) for line in edges if _key(line) in running]
        labels.append(tuple(dict.fromkeys(label for label in found if any(ch.isalpha() for ch in label))))
    return labels


def _without_page_number(line: str) -> str:
    tokens = line.split()
    while len(tokens) > 1 and tokens[0].isdigit():
        tokens = tokens[1:]
    while len(tokens) > 1 and tokens[-1].isdigit():
        tokens = tokens[:-1]
    return " ".join(tokens)
