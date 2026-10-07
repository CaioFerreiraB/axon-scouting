"""Identificadores de chunk, chave de documento e chave de conteúdo (spec 102 FR-005; research R9, R13).

`chunk_id = "<document_id>-<version>#<seção>@p<página inicial>"`, com `~k` (k ≥ 2) quando (seção, página) se repete —
determinístico dado (PDF, catálogo, estratégia de chunking), então reingerir dá os mesmos ids (SC-002).
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter
from collections.abc import Sequence
from typing import Final

_TOKEN: Final = re.compile(r"[a-z0-9]+(?:\.[a-z0-9]+)*")


def doc_key(document_id: str, version: str) -> str:
    return f"{document_id}-{version}"


def slug(text: str) -> str:
    """ASCII minúsculo, palavras unidas por hífen; pontos só entre alfanuméricos (`4.2.1`)."""
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    folded = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return "-".join(_TOKEN.findall(folded))


def chunk_ids(key: str, sections: Sequence[tuple[str, int]]) -> list[str]:
    """Um id por (seção, página inicial), na ordem; a k-ésima repetição ganha `~k`."""
    seen: Counter[tuple[str, int]] = Counter()
    ids = []
    for section, page in sections:
        seen[(section, page)] += 1
        n = seen[(section, page)]
        ids.append(f"{key}#{section}@p{page}" + (f"~{n}" if n > 1 else ""))
    return ids


def content_key(source_sha256: str, document_type: str, title: str, parser: str) -> str:
    """Igual entre builds = chunks reaproveitáveis: mesma fonte, mesmo tipo, mesmo título (prefixo), mesmo parser."""
    return hashlib.sha256("|".join((source_sha256, document_type, title, parser)).encode()).hexdigest()
