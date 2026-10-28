"""Janelas narrativas: cabeçalhos abrem seção, parágrafos são agrupados até o máximo (spec 102 FR-004; research R8).

Tamanho em tokens estimados (`⌈caracteres/4⌉`, a estimativa do core), para não acoplar o chunking ao tokenizador do
embedder. Os números são provisórios e entram na identidade da estratégia — e, portanto, na versão do índice
(princípio X).

Os títulos têm hierarquia (`estrutural-v3`, research R32): um título em maiúsculas que se repete no documento
("SUGESTÃO DE TEMAS" em cada especialidade) é subtítulo do título anterior que não se repete, e o rótulo fica
"GASTRONOMIA · SUGESTÃO DE TEMAS"; um item numerado não é título quando a numeração recomeça no documento (os
requisitos de cada especialidade), quando o número é um ano ou quando a linha seguinte continua a frase.

Com a tipografia do documento (ADR-050, `domain/typography.py`), o título é a linha num estilo de título, com o nível
do tamanho; a linha em maiúsculas ou numerada na fonte do corpo (o nome de um premiado, o item "10. Deliberar…" de uma
lista) deixa de ser título — só "CAPÍTULO N" continua valendo pelo texto. Linhas seguidas de um
mesmo título tipográfico se juntam ("Assembleia Regional" / "Escoteira").
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Final

from axon_scouting.rag.ingestion.domain.ids import slug
from axon_scouting.rag.ingestion.domain.text import Line

_CHAPTER: Final = re.compile(r"^CAP[IÍ]TULO\s+(\d+|[IVXLCDM]+)\b", re.IGNORECASE)
_NUMBERED: Final = re.compile(r"^(\d+(?:\.\d+){0,3})\.?\s+(\S.*)$")
_TERMINAL: Final = (".", ";", ":", ",", "!", "?")
_PARAGRAPH_END: Final = (".", "!", "?", ":", ";")
_ITEM: Final = re.compile(r"^(?:[a-z]\)|§|[IVXLC]+\s*[–—-])")
"""Alínea (`a)`), parágrafo de norma (`§`) e inciso (`I –`) começam parágrafo mesmo sem maiúscula."""
MAX_NUMBERED_HEADING: Final = 80
MAX_CAPS_HEADING: Final = 60
MIN_BODY_LETTERS: Final = 40
RECURRING: Final = 3
"""Vezes no documento a partir das quais um título em maiúsculas é subtítulo e um número de item é lista."""
_DIGITS: Final = re.compile(r"\d+")
"""Corpo com menos letras que isso e sem nenhuma frase terminada ("O Marco Simbólico" entre dois cabeçalhos) não vira
trecho: segue para o corpo da seção seguinte (research R28). Frase curta de verdade ("Use o lenço.") fica."""


AGENDA_ITEM: Final = re.compile(r"^(?P<num>\d{1,2})\s*[).]\s*(?P<title>[A-ZÀ-ÖØ-Ý][^:]{2,70}?)\s*:\s*(?P<rest>.*)$")
"""Item de pauta de ata ("08) PROJETOS OPAS/UEB: O Diretor…", "1. Abertura: …"): o título vai até os dois-pontos."""
QUESTION: Final = re.compile(r"^(?P<num>\d{1,2})\s*[).]\s*(?P<title>[A-ZÀ-ÖØ-Ý][^?]{2,160}\?)\s*(?P<rest>.*)$")
"""Pergunta numerada de FAQ, terminada em "?" (em uma linha ou duas)."""
UNIT_LINES: Final = {"ata": AGENDA_ITEM, "perguntas": QUESTION}
"""O perfil (`estrutura.yaml`) → a linha que abre uma unidade no texto narrativo."""
MAX_SKIP: Final = 3
"""Numeração que cresce, com saltos de até 3 (o OCR perde linhas): fora disso, a linha não é unidade."""


_BOUNDARY: Final = re.compile(r"[.!?…]+[\"”»)']*\s+(?=[A-ZÀ-ÖØ-Ý0-9\"“«(])")
ABBREVIATIONS: Final = frozenset(
    {
        "art",
        "arts",
        "nº",
        "n",
        "no",
        "sr",
        "sra",
        "srs",
        "dr",
        "dra",
        "p",
        "pp",
        "pág",
        "pag",
        "ex",
        "etc",
        "inc",
        "cap",
    }
)
"""Abreviações que terminam em ponto sem terminar a frase ("Art. 5", "nº. 3", "p. 12")."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ChunkingPolicy:
    """Faixa das janelas em tokens estimados. Configurável pela composição (`pack.py`, harness da 105), não pelo
    curador: mudar é outra versão do índice (research R8)."""

    window_min: int = 250
    window_max: int = 450
    strategy: str = "estrutural-v5"
    """v4 (ADR-048): a árvore de unidades e o rótulo tipado mudam o texto do trecho fora das normas. v5 (ADR-050):
    títulos pela tipografia, unidade normativa dividida no teto do embedder e filtro de ruído do OCR."""

    def __post_init__(self) -> None:
        if not 0 < self.window_min < self.window_max:
            raise ValueError(f"janela inválida: {self.window_min}-{self.window_max} (0 < mínimo < máximo)")

    def identity(self) -> str:
        return f"{self.strategy};janela={self.window_min}-{self.window_max}"


@dataclass(frozen=True, slots=True, kw_only=True)
class ChunkDraft:
    """Um chunk antes do id e dos metadados de documento."""

    section_key: str
    prefix: str
    body: str
    page_start: int
    page_end: int
    chapter: str | None = None
    section: str | None = None
    subsection: str | None = None
    article: str | None = None
    normative: bool = False
    headings: tuple[str, ...] = ()
    """O caminho inteiro de títulos do trecho narrativo (a tipografia dá até 4 níveis; `chapter`, `section` e
    `subsection` guardam só os três primeiros)."""
    line_pages: tuple[int, ...] = ()
    """A página de cada linha de `body` (a unidade normativa): o teto (ADR-050) dá a cada parte as páginas dela."""

    @property
    def text(self) -> str:
        return f"{self.prefix}\n{self.body}"


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / 4)


@dataclass(frozen=True, slots=True)
class _Heading:
    level: int  # 1 capítulo, 2 seção, 3 subseção
    key: str | None  # None: seção sem número (recebe `s<n>`)
    text: str


@dataclass(frozen=True, slots=True)
class _Paragraph:
    text: str
    page_start: int
    page_end: int
    starts: tuple[tuple[int, int], ...] = ()
    """(posição no texto, página) do começo de cada linha — dá a página de cada frase."""

    def page_at(self, offset: int) -> int:
        page = self.page_start
        for start, line_page in self.starts:
            if start > offset:
                break
            page = line_page
        return page


@dataclass(frozen=True, slots=True)
class _Piece:
    text: str
    page_start: int
    page_end: int
    opens_paragraph: bool


def narrative_chunks(
    lines: Sequence[Line],
    title: str,
    policy: ChunkingPolicy,
    *,
    section_key: str | None = None,
    context: Sequence[str] = (),
    units: re.Pattern[str] | None = None,
    typography: Mapping[tuple[int, str], int] | None = None,
    boundary: Callable[[Sequence[str], int], bool] | None = None,
) -> list[ChunkDraft]:
    """Janelas de um texto narrativo. `section_key` fixa a chave da seção inicial (ex.: `preambulo`); `context` entra no
    prefixo depois do título (ex.: o anexo de uma norma). `units` é a linha que abre uma unidade (`UNIT_LINES`: o item
    de pauta, a pergunta do FAQ): com numeração crescente, ela é título de nível 1, e os demais títulos ficam abaixo.
    `typography` é (página, linha) → nível dos títulos tipográficos; com ela, a linha em maiúsculas da fonte do corpo
    não é título. `boundary(caminho, i)` diz se o título `i` do caminho abre uma unidade com nome (a árvore sabe): ele
    nunca sai do caminho na junção das seções pequenas."""
    chunks: list[ChunkDraft] = []
    path: dict[int, str] = {}
    recurring: set[int] = set()  # níveis do caminho ocupados por subtítulos recorrentes
    caps_seen, numbers_seen = _heading_counts(lines)
    typographic = bool(typography)
    last_typographic: tuple[int, int] | None = None  # (posição, nível) do último título tipográfico
    unnamed = 0
    key = section_key
    body: list[Line] = []
    carried: list[Line] = []

    sections: list[_Section] = []

    def flush(*, last: bool = False) -> None:
        nonlocal unnamed, key, carried
        if not any(line.text for line in body):
            return
        if is_stub(body) and not last:
            carried = [*carried, *body]  # sem corpo que valha um trecho: vai com a seção seguinte
            return
        body[:0], carried = carried, []
        if key is None:
            unnamed += 1
            key = f"s{unnamed}"
        sections.append(_Section(path=tuple(sorted(path.items())), key=key, body=list(body)))

    items, joined = _unit_lines(lines, units)
    for position, line in enumerate(lines):
        if position in joined:
            continue
        if position in items:
            title, rest = items[position]
            flush()
            body = []
            path.clear()
            recurring.clear()
            path[1] = title
            unnamed += 1
            key = f"s{unnamed}"
            if rest:
                body.append(Line(text=rest, page=line.page))
            continue
        heading = _heading(line.text)
        styled = False  # título pela tipografia: dispensa as heurísticas de texto abaixo
        if typographic:
            assert typography is not None
            level = typography.get((line.page, " ".join(line.text.split())))
            if level is not None:
                if last_typographic == (position - 1, level) and path and not any(item.text for item in body):
                    path[max(path)] = f"{path[max(path)]} {line.text}"  # o título que quebrou em duas linhas
                    last_typographic = (position, level)
                    continue
                number = _NUMBERED.match(line.text)
                heading = _Heading(level=level, key=number.group(1) if number else None, text=line.text)
                last_typographic = (position, level)
                styled = True
            elif heading is not None and not _CHAPTER.match(line.text):
                heading = None  # maiúsculas ou item numerado na fonte do corpo: um nome ou um item, não um título
        if heading is not None and items:  # numa ata com itens, o título comum (inclusive o timbre) fica sob o item
            heading = replace(heading, level=max(heading.level, 2))
        if heading is not None and heading.key is None and section_key is not None:
            heading = None  # numa seção de chave fixa (preâmbulo, anexo), linha em maiúsculas é texto
        if not styled and heading is not None and _is_list_item(heading, numbers_seen, _next_text(lines, position)):
            heading = None
        repeats = (
            heading is not None
            and not typographic
            and heading.key is None
            and caps_seen[_signature(heading.text)] >= RECURRING
        )
        if heading is not None and repeats:
            parent = max((level for level in path if level not in recurring), default=0)
            heading = replace(heading, level=min(parent + 1, 3))
        if (
            not styled
            and path
            and not any(item.text for item in body)
            and _continues_heading(path[max(path)], line.text, heading)
        ):
            current = path[max(path)]
            path[max(path)] = f"{current} – {line.text}" if _CHAPTER.fullmatch(current) else f"{current} {line.text}"
            continue
        if heading is None:
            body.append(line)
            continue
        flush()
        body = []
        for level in [lv for lv in path if lv >= heading.level]:
            del path[level]
            recurring.discard(level)
        path[heading.level] = heading.text
        if repeats:
            recurring.add(heading.level)
        if heading.key is None:
            unnamed += 1
            key = f"s{unnamed}"
        else:
            key = heading.key
    flush()
    if carried:  # sobra curta no fim do texto: entra na última seção, ou vira a única
        if sections:
            sections[-1].body.extend(carried)
        else:
            body, carried = carried, []
            flush(last=True)
    for section in _merge_small(sections, policy, boundary):
        levels = dict(section.path)
        prefix = "[" + " · ".join([title, *context, *(text for _, text in section.path)]) + "]"
        chunks.extend(
            ChunkDraft(
                section_key=section.key,
                prefix=prefix,
                body=window_body,
                page_start=start,
                page_end=end,
                chapter=levels.get(1),
                section=levels.get(2),
                subsection=levels.get(3),
                headings=tuple(text for _, text in section.path),
            )
            for window_body, start, end in _windows(_paragraphs(section.body), policy)
        )
    return chunks


@dataclass(slots=True)
class _Section:
    path: tuple[tuple[int, str], ...]
    """(nível, título) do caminho, do mais alto ao mais baixo."""
    key: str
    body: list[Line]
    inline: dict[int, str] = field(default_factory=dict)
    """Os títulos já escritos no corpo, por nível (os que saíram do caminho na junção)."""
    orphan: bool = False
    """Pequena e sem ter onde entrar: a seção seguinte pode entrar nela."""


def _merge_small(
    sections: Sequence[_Section],
    policy: ChunkingPolicy,
    boundary: Callable[[Sequence[str], int], bool] | None = None,
) -> list[_Section]:
    """Seção pequena (menos que o mínimo da janela) junta-se à anterior quando é filha ou irmã dela e as duas cabem
    numa janela (ADR-050): o caminho passa a ser o comum, e os títulos que saíram dele entram no corpo, cada um como
    parágrafo — "Prêmio Walter Dohme · Ramo Lobinho" e os irmãos viram um trecho do prêmio com os ramos dentro, em
    vez de cinco trechos de uma linha. A pequena que não coube na anterior recebe a seguinte. Seções de capítulos
    diferentes não se juntam."""
    merged: list[_Section] = []
    for section in sections:
        previous = merged[-1] if merged else None
        if previous is not None and (_small(section, policy) or previous.orphan):
            common = _common(previous.path, section.path)
            fits = _tokens(previous.body) + _tokens(section.body) + len(section.path) * 4 <= policy.window_max
            descendant = bool(common) and common == previous.path  # a neta depois que a filha já entrou na mãe
            close = descendant or len(common) >= max(1, len(section.path) - 1, len(previous.path) - 1)
            if fits and close and not _crosses_unit(previous.path, section.path, len(common), boundary):
                lost = previous.path[len(common) :]
                previous.body[:0] = _heading_lines(lost, previous.body)
                previous.inline.update(dict(lost))
                new = [
                    (level, text) for level, text in section.path[len(common) :] if previous.inline.get(level) != text
                ]
                for level, text in new:
                    for deeper in [lv for lv in previous.inline if lv > level]:
                        del previous.inline[deeper]
                    previous.inline[level] = text
                separator = Line(text="", page=section.body[0].page)
                previous.body.extend([separator, *_heading_lines(new, section.body)])
                previous.body.extend(section.body)
                previous.path = common
                continue
        # Pequena que não coube na anterior: a seguinte (irmã ou filha) entra nela, se couber.
        merged.append(
            _Section(path=section.path, key=section.key, body=list(section.body), orphan=_small(section, policy))
        )
    return merged


def _crosses_unit(
    previous: Sequence[tuple[int, str]],
    section: Sequence[tuple[int, str]],
    common: int,
    boundary: Callable[[Sequence[str], int], bool] | None,
) -> bool:
    """Algum título que sairia do caminho abre unidade com nome."""
    if boundary is None:
        return False
    return any(
        boundary([text for _, text in path], index)
        for path in (previous, section)
        for index in range(common, len(path))
    )


def _small(section: _Section, policy: ChunkingPolicy) -> bool:
    return _tokens(section.body) < policy.window_min


def _tokens(lines: Sequence[Line]) -> int:
    return estimate_tokens(" ".join(line.text for line in lines if line.text))


def _common(a: Sequence[tuple[int, str]], b: Sequence[tuple[int, str]]) -> tuple[tuple[int, str], ...]:
    common: list[tuple[int, str]] = []
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        common.append(x)
    return tuple(common)


def _heading_lines(headings: Sequence[tuple[int, str]], body: Sequence[Line]) -> list[Line]:
    """Os títulos que saíram do caminho, como linhas próprias (seguidas de linha vazia, para não colar no texto)."""
    page = next((line.page for line in body if line.text), 1)
    lines: list[Line] = []
    for _, text in headings:
        lines.extend([Line(text=text, page=page), Line(text="", page=page)])
    return lines


def _unit_lines(lines: Sequence[Line], pattern: re.Pattern[str] | None) -> tuple[dict[int, tuple[str, str]], set[int]]:
    """As linhas que abrem unidade, por posição: (título, resto da linha); e as linhas juntadas à anterior (a pergunta
    que quebrou em duas). Só valem com numeração crescente — começando em até 3, com saltos de até `MAX_SKIP`."""
    if pattern is None:
        return {}, set()
    items: dict[int, tuple[str, str]] = {}
    joined: set[int] = set()
    last = 0
    for position, line in enumerate(lines):
        if position in joined or not line.text:
            continue
        match = pattern.match(line.text)
        following = position + 1
        if match is None and following < len(lines) and lines[following].text:
            match = pattern.match(f"{line.text} {lines[following].text}")
            if match is not None:
                joined.add(following)
        if match is None:
            continue
        number = int(match.group("num"))
        if not ((last == 0 and number <= MAX_SKIP) or last < number <= last + MAX_SKIP):
            joined.discard(following)
            continue
        last = number
        text = match.string
        items[position] = (text[: match.end("title")].strip(), match.group("rest").strip())
    if len(items) < 2:  # uma linha numerada sozinha não organiza o documento
        return {}, set()
    return items, joined


def _signature(text: str) -> str:
    return _DIGITS.sub("#", text.casefold())


def _heading_counts(lines: Sequence[Line]) -> tuple[Counter[str], Counter[str]]:
    """Quantas vezes cada título em maiúsculas (dígitos neutralizados) e cada número de item são candidatos."""
    caps: Counter[str] = Counter()
    numbers: Counter[str] = Counter()
    for line in lines:
        heading = _heading(line.text)
        if heading is None:
            continue
        if heading.key is None:
            caps[_signature(heading.text)] += 1
        elif not _CHAPTER.match(heading.text):
            numbers[heading.key] += 1
    return caps, numbers


def _next_text(lines: Sequence[Line], position: int) -> str:
    return next((line.text for line in lines[position + 1 :] if line.text), "")


def _is_list_item(heading: _Heading, numbers: Counter[str], following: str) -> bool:
    """Item numerado que não é título: numeração que recomeça no documento, um ano ("2019. CRIS salienta…") ou a frase
    que continua na linha seguinte ("2. Listar os equipamentos" / "e explicar como…")."""
    if heading.key is None or _CHAPTER.match(heading.text):
        return False
    year = "." not in heading.key and len(heading.key) == 4
    return numbers[heading.key] >= RECURRING or year or following[:1].islower()


def is_stub(lines: Sequence[Line]) -> bool:
    """Poucas letras e nenhuma frase terminada: um título solto, não um trecho (research R28)."""
    letters = sum(ch.isalpha() for line in lines for ch in line.text)
    return letters < MIN_BODY_LETTERS and not _has_sentence("\n".join(line.text for line in lines))


def _continues_heading(current: str, text: str, heading: _Heading | None) -> bool:
    """Linha logo depois de um título, antes de qualquer corpo, que é parte dele: o nome do capítulo depois de
    "CAPÍTULO 3", ou a segunda linha de um título em maiúsculas (que pode terminar em vírgula)."""
    if not text or len(text) > MAX_NUMBERED_HEADING or (heading is not None and heading.key is not None):
        return False
    if _CHAPTER.fullmatch(current):
        return text[:1].isupper() and not text.endswith(_PARAGRAPH_END)
    letters = [ch for ch in current if ch.isalpha()]
    caps = bool(letters) and all(ch.isupper() for ch in letters)
    line_letters = [ch for ch in text if ch.isalpha()]
    return caps and len(line_letters) >= 3 and all(ch.isupper() for ch in line_letters) and not text.endswith(".")


def _heading(text: str) -> _Heading | None:
    if not text or text.endswith(_TERMINAL):
        return None
    if match := _CHAPTER.match(text):
        return _Heading(level=1, key=f"cap{slug(match.group(1))}", text=text)
    match = _NUMBERED.match(text)
    if match and len(text) <= MAX_NUMBERED_HEADING and match.group(2)[0].isupper():
        number = match.group(1)
        return _Heading(level=min(number.count(".") + 1, 3), key=number, text=text)
    letters = [ch for ch in text if ch.isalpha()]
    if len(text) <= MAX_CAPS_HEADING and len(letters) >= 3 and all(ch.isupper() for ch in letters):
        return _Heading(level=1, key=None, text=text)
    return None


def _starts_paragraph(text: str) -> bool:
    first = text[:1]
    return first.isupper() or first.isdigit() or first in '•-–—*"“(«'


def _paragraphs(lines: Sequence[Line]) -> list[_Paragraph]:
    paragraphs: list[_Paragraph] = []
    current: list[Line] = []
    for line in lines:
        if not line.text:
            if current:
                paragraphs.append(_join(current))
                current = []
            continue
        if current and current[-1].text.endswith(_PARAGRAPH_END) and _starts_paragraph(line.text):
            paragraphs.append(_join(current))
            current = []
        current.append(line)
    if current:
        paragraphs.append(_join(current))
    return paragraphs


def paragraph_text(lines: Sequence[Line]) -> str:
    """As linhas unidas em parágrafos, um por linha do resultado."""
    return "\n".join(p.text for p in _paragraphs(lines))


def paragraph_pages(lines: Sequence[Line]) -> tuple[int, ...]:
    """A página em que começa cada parágrafo de `paragraph_text(lines)`, na mesma ordem."""
    return tuple(p.page_start for p in _paragraphs(lines))


def _join(lines: list[Line]) -> _Paragraph:
    starts, offset = [], 0
    for line in lines:
        starts.append((offset, line.page))
        offset += len(line.text) + 1
    return _Paragraph(
        text=" ".join(line.text for line in lines),
        page_start=lines[0].page,
        page_end=lines[-1].page,
        starts=tuple(starts),
    )


def split_sentences(text: str) -> list[str]:
    """Frases: pontuação final seguida de espaço e maiúscula, número ou aspas — salvo depois de abreviação."""
    sentences: list[str] = []
    start = 0
    for match in _BOUNDARY.finditer(text):
        before = text[start : match.start()].split()
        word = before[-1].casefold() if before else ""
        if text[match.start()] == "." and (word in ABBREVIATIONS or (len(word) == 1 and word.isalpha())):
            continue
        sentences.append(text[start : match.end()].strip())
        start = match.end()
    rest = text[start:].strip()
    if rest:
        sentences.append(rest)
    return sentences


def _sentence_pieces(paragraph: _Paragraph) -> list[_Piece]:
    pieces: list[_Piece] = []
    offset = 0
    for index, sentence in enumerate(split_sentences(paragraph.text)):
        begin = paragraph.text.find(sentence, offset)
        offset = begin + len(sentence)
        pieces.append(
            _Piece(
                text=sentence,
                page_start=paragraph.page_at(begin),
                page_end=paragraph.page_at(offset - 1),
                opens_paragraph=index == 0,
            )
        )
    return pieces


def _word_slices(piece: _Piece, window_max: int) -> list[_Piece]:
    """A frase como está, ou, maior que a janela, em fatias de palavras inteiras que cabem nela."""
    if estimate_tokens(piece.text) <= window_max:
        return [piece]
    slices: list[_Piece] = []
    words: list[str] = []
    for word in piece.text.split():
        if words and estimate_tokens(" ".join([*words, word])) > window_max:
            slices.append(replace(piece, text=" ".join(words), opens_paragraph=piece.opens_paragraph and not slices))
            words = []
        words.append(word)
    slices.append(replace(piece, text=" ".join(words), opens_paragraph=piece.opens_paragraph and not slices))
    return slices


def _render(pieces: Sequence[_Piece]) -> str:
    parts: list[str] = []
    for index, piece in enumerate(pieces):
        if index:
            parts.append("\n" if piece.opens_paragraph else " ")
        parts.append(piece.text)
    return "".join(parts)


def _windows(paragraphs: list[_Paragraph], policy: ChunkingPolicy) -> list[tuple[str, int, int]]:
    """Empacotamento guloso até `window_max` tokens estimados, preferindo fronteira de parágrafo.

    Parágrafo que não cabe: se a janela já passou do mínimo, ela fecha ali; senão (ou se ele sozinho passa do máximo),
    o parágrafo entra frase a frase. Frase não é cortada, salvo a maior que o máximo — uma lista de nomes sem
    pontuação, num relatório —, que se divide entre palavras (research R28).
    """
    windows: list[tuple[str, int, int]] = []
    current: list[_Piece] = []

    def fits(pieces: Sequence[_Piece]) -> bool:
        return estimate_tokens(_render(pieces)) <= policy.window_max

    def flush() -> None:
        nonlocal current
        if current:
            windows.append((_render(current), current[0].page_start, current[-1].page_end))
            current = []

    for paragraph in paragraphs:
        whole = _Piece(paragraph.text, paragraph.page_start, paragraph.page_end, opens_paragraph=True)
        if fits([*current, whole]):
            current.append(whole)
            continue
        if current and estimate_tokens(_render(current)) >= policy.window_min:
            flush()
            if fits([whole]):
                current.append(whole)
                continue
        for sentence in _sentence_pieces(paragraph):
            for piece in _word_slices(sentence, policy.window_max):
                if current and not fits([*current, piece]):
                    flush()
                current.append(piece)
    flush()
    return _without_stub_windows(windows)


def _without_stub_windows(windows: list[tuple[str, int, int]]) -> list[tuple[str, int, int]]:
    """Janela sem frase e com poucas letras (a sobra do fim de uma seção longa) junta-se à anterior (research R28)."""
    merged: list[tuple[str, int, int]] = []
    for text, start, end in windows:
        if merged and sum(ch.isalpha() for ch in text) < MIN_BODY_LETTERS and not _has_sentence(text):
            previous, first, _ = merged[-1]
            merged[-1] = (f"{previous}\n{text}", first, end)
        else:
            merged.append((text, start, end))
    return merged


def _has_sentence(text: str) -> bool:
    return any(line.rstrip().endswith(_PARAGRAPH_END) for line in text.split("\n"))
