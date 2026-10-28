"""Ordem de leitura de uma página a partir da geometria das linhas (spec 102 US6, FR-017; research R26).

O parser entrega as linhas na ordem em que o PDF as desenha, que nem sempre é a de leitura: no POR, o título de cada
regra é desenhado depois do corpo da página inteira; nos manuais, o título da seção vem depois do texto que ele abre, e
cabeçalhos de margem vêm repetidos, impressos duas vezes no mesmo lugar. Aqui, puro: tira as linhas impressas em
duplicata e, numa página de uma coluna, ordena de cima para baixo — linhas na mesma faixa vertical ficam na ordem do
parser.

Página de colunas de texto (ADR-050): cada faixa entre linhas largas é lida coluna a coluna, da esquerda para a direita,
cada coluna de cima para baixo, com uma linha vazia (fronteira de parágrafo) entre as colunas — no relatório anual de
2023, o PDF desenha a coluna da direita antes da esquerda, e a leitura do parser juntava o premiado de um prêmio ao
texto do outro. Colunas cujas linhas compartilham a altura (a mesma linha de base dos dois lados) são uma tabela e
ficam na ordem do parser, como a página com linha sem posição.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Final

WIDE: Final = 0.6
"""Linha com mais de 60% da largura da página atravessa colunas: não conta para decidir quantas há."""
MIN_COLUMN_LINES: Final = 5
"""Uma coluna tem ao menos 5 linhas; menos é cabeçalho de margem, legenda ou número de página."""
SAME_BAND: Final = 0.5
"""Duas linhas estão na mesma faixa quando se sobrepõem em pelo menos metade da altura da menor."""
_DUPLICATE: Final = 0.5
"""Tolerância, em pontos, para a mesma linha impressa duas vezes no mesmo lugar."""
ROW_ALIGNED: Final = 1.5
"""Diferença máxima, em pontos, entre os topos de duas linhas de colunas vizinhas na mesma linha de tabela."""
TABLE_SHARE: Final = 0.6
"""Colunas em que 60% das linhas da menor têm par alinhado na outra são uma tabela."""
COLUMN_BREAK: Final = ""
"""A linha vazia entre duas colunas: fronteira de parágrafo para a limpeza e o chunking."""


@dataclass(frozen=True, slots=True, kw_only=True)
class TextLine:
    """Uma linha de texto e a caixa dela, em pontos, com a origem no canto inferior esquerdo (a do PDF)."""

    text: str
    left: float
    bottom: float
    right: float
    top: float
    size: float = 0.0
    """Tamanho efetivo da fonte, em pontos (o da maioria dos caracteres); 0 = desconhecido."""
    font: str = ""
    """Família da fonte, sem o prefixo de subconjunto."""
    bold: bool = False


def reading_order(lines: Sequence[TextLine | str], page_width: float) -> list[str]:
    """O texto das linhas na ordem de leitura. Uma `str` é linha sem posição (o parser não achou caixa); a linha vazia
    (`COLUMN_BREAK`) separa duas colunas."""
    return [line.text if isinstance(line, TextLine) else line for line in ordered_lines(lines, page_width)]


def ordered_lines(lines: Sequence[TextLine | str], page_width: float) -> list[TextLine | str]:
    """As linhas na ordem de leitura, com `COLUMN_BREAK` entre colunas."""
    unique = _without_duplicates(lines)
    placed = [line for line in unique if isinstance(line, TextLine)]
    if len(placed) != len(unique):
        return list(unique)
    columns = _columns(placed, page_width)
    if len(columns) <= 1:
        return list(_top_to_bottom(placed))
    if _row_aligned(placed, columns):
        return list(unique)
    return _by_columns(placed, columns, page_width)


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
    return len(_columns(lines, page_width)) <= 1


def _columns(lines: Sequence[TextLine], page_width: float) -> list[tuple[float, float]]:
    """Agrupa as linhas estreitas pela sobreposição horizontal; os grupos com 5 linhas ou mais são as colunas
    (esquerda, direita), da esquerda para a direita."""
    narrow = sorted(
        (line for line in lines if line.text.strip() and not _wide(line, page_width)),
        key=lambda line: line.left,
    )
    groups: list[list[float]] = []  # [esquerda, direita, linhas]
    for line in narrow:
        if groups and line.left <= groups[-1][1]:
            groups[-1][1] = max(groups[-1][1], line.right)
            groups[-1][2] += 1
        else:
            groups.append([line.left, line.right, 1])
    return [(group[0], group[1]) for group in groups if group[2] >= MIN_COLUMN_LINES]


def _wide(line: TextLine, page_width: float) -> bool:
    return line.right - line.left >= WIDE * page_width


def _column_of(line: TextLine, columns: Sequence[tuple[float, float]]) -> int:
    """A coluna com mais sobreposição horizontal; sem nenhuma, a de borda esquerda mais próxima."""
    overlaps = [min(line.right, right) - max(line.left, left) for left, right in columns]
    best = max(range(len(columns)), key=lambda index: overlaps[index])
    if overlaps[best] > 0:
        return best
    return min(range(len(columns)), key=lambda index: abs(columns[index][0] - line.left))


def _row_aligned(lines: Sequence[TextLine], columns: Sequence[tuple[float, float]]) -> bool:
    """Tabela: em cada par de colunas vizinhas, 60% das linhas da menor têm, na outra, uma linha de mesmo topo."""
    members: list[list[TextLine]] = [[] for _ in columns]
    for line in lines:
        if line.text.strip():
            members[_column_of(line, columns)].append(line)
    for a, b in pairwise(members):
        small, other = (a, b) if len(a) <= len(b) else (b, a)
        if not small:
            continue
        paired = sum(1 for line in small if any(abs(line.top - o.top) <= ROW_ALIGNED for o in other))
        if paired >= TABLE_SHARE * len(small):
            return True
    return False


def _by_columns(
    lines: Sequence[TextLine], columns: Sequence[tuple[float, float]], page_width: float
) -> list[TextLine | str]:
    """Faixas separadas pelas linhas largas, de cima para baixo; em cada faixa, uma coluna por vez."""
    result: list[TextLine | str] = []
    band: list[TextLine] = []

    def flush() -> None:
        groups: list[list[TextLine]] = [[] for _ in columns]
        for line in band:
            groups[_column_of(line, columns)].append(line)
        filled = [group for group in groups if group]
        for index, group in enumerate(filled):
            if index and result and result[-1] != COLUMN_BREAK:
                result.append(COLUMN_BREAK)
            result.extend(_top_to_bottom(group))
        band.clear()

    for line in sorted(lines, key=lambda item: -item.top):
        if _wide(line, page_width):
            flush()
            if result and result[-1] != COLUMN_BREAK:
                result.append(COLUMN_BREAK)
            result.append(line)
            continue
        band.append(line)
    flush()
    return result


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
