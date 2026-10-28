"""O teto do trecho no embedder (spec 102 FR-032; ADR-050). Puro: a contagem de tokens chega como função.

O e5 lê 512 tokens e descarta o resto em silêncio; a janela é medida em tokens estimados (`⌈caracteres/4⌉`), e o texto
real chega a 1,6 vez a estimativa (números, nomes, ruído de OCR). No artefato da `estrutural-v4`, 378 trechos passavam
do teto — 82 artigos ou regras, que viravam um trecho só do tamanho que tivessem, e 296 janelas. Aqui, depois da
árvore (o rótulo final já existe), todo trecho que passa do teto é dividido:

- unidade normativa: nas linhas do corpo (o caput, cada parágrafo, inciso e alínea), e cada parte depois da primeira
  repete o caput curto e leva "Art. N (parte i de n)" no rótulo — a parte sozinha não diz de que artigo é;
- janela narrativa: nas linhas e, se preciso, nas frases;
- o que nem assim cabe (uma frase enorme, uma tabela sem pontuação): entre palavras.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Sequence
from typing import Final

from axon_scouting.rag.ingestion.domain.chunking import ChunkDraft, split_sentences

CEILING: Final = 512
"""Tokens que o e5 lê, contando o prefixo `passage: ` e os tokens especiais."""
REMNANT: Final = 40
"""Parte final com até 40 tokens (as medidas de uma figura, o fim de uma tabela) junta-se à anterior quando cabe."""
CAPUT_TOKENS: Final = 80
"""O caput repetido nas continuações tem até 80 tokens; maior, só o rótulo diz de que artigo é a parte."""

Count = Callable[[str], int]


def fit(draft: ChunkDraft, count: Count, ceiling: int = CEILING) -> list[ChunkDraft]:
    """O trecho como está, ou as partes que cabem no teto, na ordem do texto."""
    if count(draft.text) <= ceiling:
        return [draft]
    raw = draft.body.split("\n")
    known = len(draft.line_pages) == len(raw)
    pages_of = [draft.line_pages[i] if known else draft.page_start for i, line in enumerate(raw) if line.strip()]
    lines = [line for line in raw if line.strip()]
    caput = lines[0] if draft.normative and lines and count(lines[0]) <= CAPUT_TOKENS else None
    pieces = _pieces(lines, pages_of, count)
    bodies = _pack(pieces, draft, count, ceiling, caput)
    if len(bodies) == 1:
        return [dataclasses.replace(draft, body=bodies[0][0])]
    total = len(bodies)
    parts: list[ChunkDraft] = []
    for index, (body, first, last) in enumerate(bodies, start=1):
        prefix = _part_prefix(draft, index, total)
        start, end = (first, max(first, last)) if known else (draft.page_start, draft.page_end)
        parts.append(
            dataclasses.replace(draft, prefix=prefix, body=body, line_pages=(), page_start=start, page_end=end)
        )
    return parts


def _part_prefix(draft: ChunkDraft, index: int, total: int) -> str:
    if not draft.normative or not draft.article:
        return draft.prefix
    return f"{draft.prefix.removesuffix(']')} · {draft.article} (parte {index} de {total})]"


def _pieces(lines: Sequence[str], pages: Sequence[int], count: Count) -> list[tuple[str, bool, int]]:
    """(texto, abre linha, página) — as linhas, e, das que sozinhas são grandes demais, as frases e as fatias de
    palavras (com a página da linha)."""
    pieces: list[tuple[str, bool, int]] = []
    for line, page in zip(lines, pages, strict=True):
        if count(line) <= CEILING // 2:
            pieces.append((line, True, page))
            continue
        for number, sentence in enumerate(split_sentences(line)):
            for slice_number, words in enumerate(_word_slices(sentence, count)):
                pieces.append((words, number == 0 and slice_number == 0, page))
    return pieces


def _word_slices(sentence: str, count: Count) -> list[str]:
    if count(sentence) <= CEILING // 2:
        return [sentence]
    slices: list[str] = []
    words: list[str] = []
    for word in sentence.split():
        if words and count(" ".join([*words, word])) > CEILING // 2:
            slices.append(" ".join(words))
            words = []
        words.append(word)
    if words:
        slices.append(" ".join(words))
    return slices


def _pack(
    pieces: Sequence[tuple[str, bool, int]], draft: ChunkDraft, count: Count, ceiling: int, caput: str | None
) -> list[tuple[str, int, int]]:
    """Empacotamento das peças em partes equilibradas: o número de partes é o mínimo que o teto exige, e cada parte
    fecha ao passar da média — sem isso, a última parte sobrava com uma linha. O rótulo de parte (o mais longo, "parte
    99 de 99") já entra na conta; cada corpo leva a primeira e a última página das peças dele."""
    longest = _part_prefix(draft, 99, 99) if draft.normative and draft.article else draft.prefix
    budget = max(1, ceiling - count(f"{longest}\n") - (count(caput) if caput else 0))
    total = sum(count(piece) for piece, _, _ in pieces)
    target = math.ceil(total / math.ceil(total / budget)) if total > budget else total
    bodies: list[tuple[str, int, int]] = []
    current, first, last = "", draft.page_start, draft.page_start

    def text(body: str) -> str:
        return f"{longest}\n{body}"

    for piece, opens, page in pieces:
        candidate = piece if not current else f"{current}{chr(10) if opens else ' '}{piece}"
        if current and (count(text(candidate)) > ceiling or count(current) >= target):
            bodies.append((current, first, last))
            current = f"{caput}\n{piece}" if caput and piece != caput else piece
            first = last = page
            continue
        if not current:
            first = page
        current, last = candidate, page
    if current:
        bodies.append((current, first, last))
    while len(bodies) > 1 and count(bodies[-1][0]) <= REMNANT:  # a sobra pequena vai para a parte anterior, se couber
        (before, start, _), (tail, _, end) = bodies[-2], bodies[-1]
        tail = tail.removeprefix(f"{caput}\n") if caput else tail
        joined = f"{before}\n{tail}"
        if count(text(joined)) > ceiling:
            break
        bodies[-2:] = [(joined, start, end)]
    return bodies
