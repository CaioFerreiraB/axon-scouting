"""Normalização de texto e casamento de termos (spec 101, research R6). Puro: sem I/O, sem relógio.

Pergunta e léxico passam pela mesma normalização, então o casamento é por palavra inteira, sem diferença de
maiúsculas, acentos ou pontuação. Plural e feminino são termos do léxico — não há stemmer.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Final

_SEPARATORS: Final = re.compile(r"[^a-z0-9]+")


def normalize(text: str) -> str:
    """Minúsculas, sem diacríticos; toda sequência que não é letra ou dígito vira um espaço."""
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    plain = "".join(char for char in decomposed if not unicodedata.combining(char))
    return " ".join(_SEPARATORS.sub(" ", plain).split())


def contains_term(normalized_text: str, term: str) -> bool:
    """O termo (já normalizado, de uma ou mais palavras) aparece inteiro no texto normalizado."""
    return f" {term} " in f" {normalized_text} "
