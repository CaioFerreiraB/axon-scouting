"""Títulos pela tipografia (spec 102 FR-031; ADR-050). Puro: recebe as linhas com fonte e tamanho que o parser leu.

O texto sozinho não distingue um título de um nome em maiúsculas: no relatório anual de 2023, "CAIO FERREIRA
BERNARDO" (corpo, Avenir 10 pt) virava título, e "Prêmio Walter Dohme" (Gotcha Gothic 14 pt), não. Quando o
documento tem estilos de título — uma fonte, um tamanho ou um negrito que só aparecem em linhas curtas —, a
tipografia decide: é título a linha num desses estilos, e o nível vem do tamanho (o maior é o 1); no estilo do
tamanho do corpo (negrito, outra família), só a linha que se sustenta sozinha — o destaque no meio do parágrafo não é
título. Documento sem estilo de título (OCR, texto corrido numa fonte só) fica com os títulos pelo texto, como antes.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import median
from typing import Final

from axon_scouting.rag.ingestion.domain.layout import TextLine

LARGER: Final = 1.12
"""Um estilo 12% maior que o do corpo é de título."""
MAX_SHARE: Final = 0.3
"""Estilo com mais de 30% dos caracteres do documento é corpo, não título (uma segunda fonte de texto)."""
MAX_HEADING_CHARS: Final = 100
MAX_MEDIAN_CHARS: Final = 70
"""As linhas de um estilo de título são curtas: a mediana até 70 caracteres."""
MIN_HEADING_LINES: Final = 2
SAME_LEVEL: Final = 1.0
"""Tamanhos a até 1 pt um do outro são o mesmo nível."""
MAX_LEVEL: Final = 4
MIN_PAGES: Final = 2
"""Um estilo de título se repete em 2 páginas ou mais (a citação em 30 pt de uma página só não é título); documento
de até 4 páginas dispensa a repetição."""
SHORT_DOCUMENT: Final = 4
STRUCTURAL_PAGES: Final = 3
"""Os níveis saem dos estilos que estruturam o documento (em 3 páginas ou mais); a capa em 80 pt e a citação em 30 pt
de duas páginas recebem o nível do estilo estrutural de tamanho mais próximo."""
MAX_RUN: Final = 3
"""Mais de 3 linhas seguidas no mesmo estilo de título são um parágrafo em destaque, não um título."""
_WEIGHT_SUFFIX: Final = re.compile(r"(?:MT|PS|Std|Pro|LT)$")


@dataclass(frozen=True, slots=True, kw_only=True)
class TypographicHeading:
    page: int
    """A primeira página é a 1."""
    text: str
    """O texto da linha, com os espaços colapsados (a chave com que a limpeza a reencontra)."""
    level: int


Style = tuple[str, bool, float]


def style_of(line: TextLine) -> Style | None:
    if line.size <= 0:
        return None
    return (line.font, line.bold, round(line.size * 2) / 2)


def key(text: str) -> str:
    return " ".join(text.split())


def family(font: str) -> str:
    """A raiz do nome da fonte: "Avenir-Book" e "Avenir-Medium" são a mesma família; "GotchaGothic", outra."""
    return _WEIGHT_SUFFIX.sub("", re.split(r"[-,]", font, maxsplit=1)[0])


def headings(pages: Sequence[Sequence[TextLine]]) -> tuple[TypographicHeading, ...]:
    """Os títulos tipográficos do documento; vazio quando ele não tem estilo de título."""
    chars: Counter[Style] = Counter()
    lengths: defaultdict[Style, list[int]] = defaultdict(list)
    seen_on: defaultdict[Style, set[int]] = defaultdict(set)
    for number, lines in enumerate(pages, start=1):
        for line in lines:
            style = style_of(line)
            text = key(line.text)
            if style is None or not any(ch.isalpha() for ch in text):
                continue
            chars[style] += len(text)
            lengths[style].append(len(text))
            seen_on[style].add(number)
    if not chars:
        return ()
    body = chars.most_common(1)[0][0]
    total = sum(chars.values())
    candidates = {
        style
        for style in chars
        if style != body
        and _distinct(style, body)
        and chars[style] <= MAX_SHARE * total
        and len(lengths[style]) >= MIN_HEADING_LINES
        and median(lengths[style]) <= MAX_MEDIAN_CHARS
        and (len(seen_on[style]) >= MIN_PAGES or len(pages) <= SHORT_DOCUMENT)
    }
    if not candidates:
        return ()
    structural = {style for style in candidates if len(seen_on[style]) >= STRUCTURAL_PAGES} or candidates
    levels = _levels(candidates, structural)
    found: list[TypographicHeading] = []
    for number, lines in enumerate(pages, start=1):
        following = {id(line): _next_text(lines, index) for index, line in enumerate(lines)}
        for run in _runs(lines):
            style = style_of(run[0])
            if style not in candidates or len(run) > MAX_RUN:
                continue
            larger = style[2] >= LARGER * body[2]
            for line in run:
                text = key(line.text)
                if len(text) > MAX_HEADING_CHARS or not any(ch.isalpha() for ch in text) or _sentence(text):
                    continue
                if not larger and _inline(text, following[id(line)]):
                    continue
                found.append(TypographicHeading(page=number, text=text, level=levels[style]))
    return tuple(found)


def _next_text(lines: Sequence[TextLine], index: int) -> str:
    return next((key(line.text) for line in lines[index + 1 :] if key(line.text)), "")


def _inline(text: str, following: str) -> bool:
    """Destaque no meio do parágrafo (negrito ou itálico do tamanho do corpo), não título: a linha começa em
    minúscula, termina em vírgula ou travessão, ou a frase continua na linha seguinte, em minúscula ("– “Descobrir
    novos territórios com um" / "grupo de amigos” – está…")."""
    first = next((ch for ch in text if ch.isalpha()), "")
    return first.islower() or text.endswith((",", "–", "-", "—")) or following[:1].islower()


def _sentence(text: str) -> bool:
    """Termina em ponto final com três palavras ou mais: o fim de uma frase no estilo do título, não um título."""
    return text.endswith(".") and not text.endswith("...") and len(text.split()) >= 3


def _runs(lines: Sequence[TextLine]) -> list[list[TextLine]]:
    """As linhas seguidas (na ordem de leitura) no mesmo estilo; linha sem letra não interrompe nem conta."""
    runs: list[list[TextLine]] = []
    for line in lines:
        if not any(ch.isalpha() for ch in line.text):
            continue
        if runs and style_of(runs[-1][-1]) == style_of(line):
            runs[-1].append(line)
        else:
            runs.append([line])
    return runs


def _distinct(style: Style, body: Style) -> bool:
    """Maior que o corpo, ou do mesmo tamanho mas em negrito ou noutra família."""
    font, bold, size = style
    body_font, body_bold, body_size = body
    if size >= LARGER * body_size:
        return True
    if size < body_size - SAME_LEVEL:
        return False  # menor que o corpo: legenda, rodapé, crédito
    return (bold and not body_bold) or family(font) != family(body_font)


def _levels(styles: set[Style], structural: set[Style]) -> dict[Style, int]:
    """Nível pelo tamanho dos estilos estruturais: o maior é 1; tamanhos a até 1 pt dividem o nível; do quarto em
    diante, 4. Os outros estilos ficam com o nível do estrutural de tamanho mais próximo."""
    sizes = sorted({style[2] for style in structural}, reverse=True)
    level_of: dict[float, int] = {}
    level, previous = 0, None
    for size in sizes:
        if previous is None or previous - size > SAME_LEVEL:
            level += 1
        level_of[size] = min(level, MAX_LEVEL)
        previous = size
    return {style: level_of[min(sizes, key=lambda size: abs(size - style[2]))] for style in styles}
