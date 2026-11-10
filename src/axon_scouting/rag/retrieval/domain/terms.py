"""Pontuação por termos, sem vetor (spec 103; ADR-062, ADR-063): o BM25 de uma pergunta sobre poucos candidatos já
escolhidos por metadado — os resumos das atas, os trechos das seções que o grafo indicou.

Plural reduzido ao singular, frequência saturada e peso pela raridade entre os candidatos (IDF). Puro.
"""

from __future__ import annotations

import math
import unicodedata
from collections.abc import Sequence
from typing import Final

STOP: Final = frozenset(
    {
        "a",
        "o",
        "as",
        "os",
        "um",
        "uma",
        "de",
        "da",
        "do",
        "das",
        "dos",
        "em",
        "na",
        "no",
        "nas",
        "nos",
        "para",
        "por",
        "pela",
        "pelo",
        "com",
        "sem",
        "sobre",
        "e",
        "ou",
        "que",
        "se",
        "foi",
        "foram",
        "ser",
        "sao",
        "como",
        "mais",
        "quem",
        "qual",
        "quais",
        "quando",
        "onde",
        "quantos",
        "quantas",
        "entre",
        "desde",
        "ate",
        "apos",
        "anos",
        "ano",
        "pode",
        "podem",
    }
)
SATURATION: Final = 1.2


def normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    plain = "".join(char for char in decomposed if not unicodedata.combining(char))
    return " ".join("".join(char if char.isalnum() else " " for char in plain).split())


def stem(word: str) -> str:
    """Plural para o singular, o suficiente para "diretores"/"diretor", "eleições"/"eleição", "presidentes"."""
    if len(word) > 5 and word.endswith("oes"):
        return word[:-3] + "ao"
    if len(word) > 5 and word.endswith(("res", "zes", "les")):
        return word[:-2]
    if len(word) > 4 and word.endswith("s"):
        return word[:-1]
    return word


def query_terms(query: str) -> list[str]:
    return [
        stem(word) for word in normalize(query).split() if len(word) >= 3 and word not in STOP and not word.isdigit()
    ]


def bag(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for word in normalize(text).split():
        key = stem(word)
        counts[key] = counts.get(key, 0) + 1
    return counts


def idf(bags: Sequence[dict[str, int]], terms: Sequence[str]) -> dict[str, float]:
    total = len(bags)
    return {term: math.log(1 + (total + 1) / (1 + sum(term in b for b in bags))) for term in set(terms)}


def score(counts: dict[str, int], terms: Sequence[str], weights: dict[str, float]) -> float:
    return sum(weights[term] * counts[term] / (counts[term] + SATURATION) for term in terms if term in counts)
