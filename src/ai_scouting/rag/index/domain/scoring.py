"""Pontuação da emulação em memória e ordem determinística (spec 102 research R11, R12).

As fórmulas são as do Qdrant — IDF `ln(1 + (N − n + 0.5)/(n + 0.5))` e RRF `Σ 1/(posição + 2)`, posição a partir de 0
(verificado na sonda de 2026-10-07) —, para o índice em memória se comportar como o real nos testes da 103.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence

from ai_scouting.rag.index.api import IndexedChunk
from ai_scouting.rag.index.ports import SparseVector


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return 0.0 if norm == 0 else sum(x * y for x, y in zip(a, b, strict=True)) / norm


def idf(total: int, containing: int) -> float:
    return math.log(1 + (total - containing + 0.5) / (containing + 0.5))


def sparse_score(query: SparseVector, document: SparseVector, weights: Mapping[int, float]) -> float:
    """Produto escalar com o IDF aplicado aos termos da pergunta (`modifier=IDF`)."""
    values = dict(zip(document.indices, document.values, strict=True))
    return sum(
        q * weights.get(i, 0.0) * values[i] for i, q in zip(query.indices, query.values, strict=True) if i in values
    )


def rrf(rankings: Iterable[Sequence[str]]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for position, key in enumerate(ranking):
            scores[key] = scores.get(key, 0.0) + 1 / (position + 2)
    return scores


def ordered(chunks: Iterable[IndexedChunk]) -> list[IndexedChunk]:
    """Score decrescente e, no empate, `chunk_id` — a mesma ordem em todo adapter."""
    return sorted(chunks, key=lambda c: (-(c.score or 0.0), c.chunk_id))
