"""`Bm25Encoder`: representação esparsa BM25 com stemmer português (spec 102 research R11).

O peso de frequência é calculado aqui; o IDF fica com o Qdrant (`modifier=IDF`) — o índice em memória o emula. Os
parâmetros são os padrões da literatura, provisórios e gravados na versão do índice para a 105 medir (princípio X).
"""

from __future__ import annotations

import re
import unicodedata
import zlib
from collections import Counter
from typing import Final

import snowballstemmer

from ai_scouting.rag.index.ports import SparseVector

K1: Final = 1.2
B: Final = 0.75
AVGDL: Final = 256

_TOKEN: Final = re.compile(r"[a-z0-9]+")

STOPWORDS: Final = frozenset(
    (
        "a",
        "ao",
        "aos",
        "aquela",
        "aquelas",
        "aquele",
        "aqueles",
        "aquilo",
        "as",
        "ate",
        "com",
        "como",
        "da",
        "das",
        "de",
        "dela",
        "delas",
        "dele",
        "deles",
        "depois",
        "do",
        "dos",
        "e",
        "ela",
        "elas",
        "ele",
        "eles",
        "em",
        "entre",
        "era",
        "eram",
        "essa",
        "essas",
        "esse",
        "esses",
        "esta",
        "estas",
        "este",
        "estes",
        "eu",
        "foi",
        "foram",
        "ha",
        "isso",
        "isto",
        "ja",
        "lhe",
        "lhes",
        "mais",
        "mas",
        "me",
        "mesmo",
        "meu",
        "meus",
        "minha",
        "minhas",
        "muito",
        "na",
        "nao",
        "nas",
        "nem",
        "no",
        "nos",
        "nossa",
        "nossas",
        "nosso",
        "nossos",
        "num",
        "numa",
        "o",
        "os",
        "ou",
        "para",
        "pela",
        "pelas",
        "pelo",
        "pelos",
        "por",
        "qual",
        "quando",
        "que",
        "quem",
        "se",
        "sem",
        "ser",
        "seu",
        "seus",
        "so",
        "sua",
        "suas",
        "tambem",
        "te",
        "tem",
        "tu",
        "tua",
        "tuas",
        "um",
        "uma",
        "umas",
        "uns",
        "voce",
        "voces",
        "vos",
    )
)
"""Palavras de função do português, já dobradas (sem acento)."""


class Bm25Encoder:
    def __init__(self) -> None:
        self._stemmer = snowballstemmer.stemmer("portuguese")

    def identity(self) -> str:
        return f"bm25-pt-v1;k1={K1};b={B};avgdl={AVGDL}"

    def encode_passage(self, text: str) -> SparseVector:
        terms = self._terms(text)
        counts = Counter(terms)
        norm = K1 * (1 - B + B * len(terms) / AVGDL)
        return _vector({term: tf * (K1 + 1) / (tf + norm) for term, tf in counts.items()})

    def encode_query(self, text: str) -> SparseVector:
        return _vector(dict.fromkeys(self._terms(text), 1.0))

    def _terms(self, text: str) -> list[int]:
        terms = []
        for token in _TOKEN.findall(_fold(text)):
            if token in STOPWORDS:
                continue
            word = str(int(token)) if token.isdigit() else self._stemmer.stemWord(token)
            terms.append(zlib.crc32(word.encode()))
        return terms


def _fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _vector(weights: dict[int, float]) -> SparseVector:
    indices = tuple(sorted(weights))
    return SparseVector(indices=indices, values=tuple(weights[i] for i in indices))
