"""Dobra de texto para as comparações dos validators: minúsculas, sem acento, espaços colapsados (spec 104, R8).

Própria da camada: os validators não importam `context/` (submódulos do pack independentes — design.md).
"""

from __future__ import annotations

import unicodedata


def fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    plain = "".join(char for char in decomposed if not unicodedata.combining(char))
    return " ".join(plain.split())
