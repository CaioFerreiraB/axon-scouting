"""Ordem de leitura de uma página a partir da geometria das linhas (spec 102 US6, FR-017; research R26).

O parser entrega as linhas na ordem em que o PDF as desenha, que nem sempre é a de leitura: no POR, o título de cada
regra é desenhado depois do corpo da página inteira; nos manuais, o título da seção vem depois do texto que ele abre, e
cabeçalhos de margem vêm repetidos, impressos duas vezes no mesmo lugar. Aqui, puro: tira as linhas impressas em
duplicata e, numa página de uma coluna, ordena de cima para baixo — linhas na mesma faixa vertical ficam na ordem do
parser. Página de várias colunas, ou com linha sem posição, fica na ordem do parser: ordenar por altura misturaria as
colunas.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

WIDE: Final = 0.6
"""Linha com mais de 60% da largura da página atravessa colunas: não conta para decidir quantas há."""
MIN_COLUMN_LINES: Final = 5
"""Uma coluna tem ao menos 5 linhas; menos é cabeçalho de margem, legenda ou número de página."""
SAME_BAND: Final = 0.5
"""Duas linhas estão na mesma faixa quando se sobrepõem em pelo menos metade da altura da menor."""
_DUPLICATE: Final = 0.5
"""Tolerância, em pontos, para a mesma linha impressa duas vezes no mesmo lugar."""


@dataclass(frozen=True, slots=True, kw_only=True)
class TextLine:
    """Uma linha de texto e a caixa dela, em pontos, com a origem no canto inferior esquerdo (a do PDF)."""

    text: str
    left: float
    bottom: float
    right: float
    top: float


def reading_order(lines: Sequence[TextLine | str], page_width: float) -> list[str]:
    """O texto das linhas na ordem de leitura. Uma `str` é linha sem posição (o parser não achou caixa)."""
    unique = _without_duplicates(lines)
    placed = [line for line in unique if isinstance(line, TextLine)]
    if len(placed) != len(unique) or not _single_column(placed, page_width):
        return [line.text if isinstance(line, TextLine) else line for line in unique]
    return [line.text for line in _top_to_bottom(placed)]


def _without_duplicates(lines: Sequence[TextLine | str]) -> list[TextLine | str]:
    seen: list[TextLine] = []
    kept: list[TextLine | str] = []
    for line in lines:
        if isinstance(line, TextLine):
            if any(_same_place(line, other) for other in seen):
                continue
            seen.append(line)
        kept.append(line)
    return kept


def _same_place(a: TextLine, b: TextLine) -> bool:
    return a.text == b.text and all(
        abs(x - y) <= _DUPLICATE
        for x, y in ((a.left, b.left), (a.bottom, b.bottom), (a.right, b.right), (a.top, b.top))
    )


def _single_column(lines: Sequence[TextLine], page_width: float) -> bool:
    """Agrupa as linhas estreitas pela sobreposição horizontal; mais de um grupo com 5 linhas ou mais é colunas."""
    narrow = sorted(
        (line for line in lines if line.text.strip() and line.right - line.left < WIDE * page_width),
        key=lambda line: line.left,
    )
    groups: list[list[float]] = []  # [esquerda, direita, linhas]
    for line in narrow:
        if groups and line.left <= groups[-1][1]:
            groups[-1][1] = max(groups[-1][1], line.right)
            groups[-1][2] += 1
        else:
            groups.append([line.left, line.right, 1])
    return sum(1 for group in groups if group[2] >= MIN_COLUMN_LINES) <= 1


def _top_to_bottom(lines: Sequence[TextLine]) -> list[TextLine]:
    """Faixas de cima para baixo; dentro da faixa, a ordem do parser (o "I -" antes do texto do inciso)."""
    by_height = sorted(range(len(lines)), key=lambda index: -lines[index].top)
    bands: list[tuple[float, float, list[int]]] = []
    for index in by_height:
        line = lines[index]
        if bands:
            low, high, members = bands[-1]
            overlap = min(high, line.top) - max(low, line.bottom)
            if overlap >= SAME_BAND * min(line.top - line.bottom, high - low):
                bands[-1] = (min(low, line.bottom), high, [*members, index])
                continue
        bands.append((line.bottom, line.top, [index]))
    return [lines[index] for _, _, members in bands for index in sorted(members)]
