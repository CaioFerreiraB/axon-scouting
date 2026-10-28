"""A lista de documentos do acervo (spec 103 FR-013, FR-014; research R16; ADR-047). Puro: sem I/O, sem relógio.

O pedido de lista da 101 traz a espécie — o id do tipo na taxonomia (ADR-049), ou `documento` para qualquer tipo — e
as palavras que a qualificam. Um documento entra quando é do tipo e alcança a maior fração dos qualificadores que algum
documento do acervo alcança, no título ou na classificação ("atas da assembleia nacional": as que têm as duas
palavras, não as que têm uma); a ordem é do mais recente. Título e pergunta se comparam
normalizados e no singular, por regra simples de plural do português: o que se compara são títulos curtos, e o que a
regra erra só tira o documento da lista — a busca por trechos continua rodando ao lado.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Final

from axon_ai_contracts import Chunk, JSONValue

from axon_scouting.rag.index.api import STATUS_LABELS, CatalogEntry, IndexFilters, ValidityStatus
from axon_scouting.rag.retrieval.domain.ordering import CATALOG_RANK, FUSION_SCORE, marks

ANY_KIND: Final = "documento"
CARD_SUFFIX: Final = "#acervo"
"""Sufixo do id da ficha: `<doc_key>#acervo`. Trecho do índice sempre tem `@p<página>` depois do `#`."""
BRANCH_NAMES: Final = {
    "filhotes": "Filhotes",
    "lobinho": "Lobinho",
    "escoteiro": "Escoteiro",
    "senior": "Sênior",
    "pioneiro": "Pioneiro",
}
"""O nome de cada ramo por extenso (na ficha e na subbusca da faceta, "… Ramo Sênior")."""

_SEPARATORS: Final = re.compile(r"[^a-z0-9]+")
_PLURALS: Final = (("oes", "ao"), ("aes", "ao"), ("ais", "al"), ("eis", "el"), ("ois", "ol"), ("ns", "m"))


@dataclass(frozen=True, slots=True, kw_only=True)
class CatalogRequest:
    """O pedido de lista que a 101 publicou em `retrieval_filters["catalog"]`."""

    kind: str
    terms: tuple[str, ...]


def normalize(text: str) -> str:
    """Minúsculas, sem diacríticos; toda sequência que não é letra ou dígito vira um espaço (a forma da 101)."""
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    plain = "".join(char for char in decomposed if not unicodedata.combining(char))
    return " ".join(_SEPARATORS.sub(" ", plain).split())


def singular(word: str) -> str:
    """O singular por regra: "politicas" → "politica", "resolucoes" → "resolucao", "manuais" → "manual"."""
    if len(word) <= 3 or word.isdigit():
        return word
    for plural, single in _PLURALS:
        if word.endswith(plural):
            return word[: -len(plural)] + single
    if word.endswith(("res", "zes")):
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def title_words(title: str) -> frozenset[str]:
    return frozenset(singular(word) for word in normalize(title).split())


def taxonomy_words(entry: CatalogEntry) -> frozenset[str]:
    """As palavras da classificação do documento (órgão, tipo, nível, jurisdição), no singular: "atas da assembleia
    regional" acha as atas da categoria Assembleias de nível Regional, mesmo sem a palavra no título (ADR-049)."""
    labels = (entry.category, entry.subcategory, entry.semantic_type, entry.level, entry.scope, entry.jurisdiction)
    return frozenset(singular(word) for label in labels if label for word in normalize(label).split())


def select(entries: Iterable[CatalogEntry], request: CatalogRequest) -> list[tuple[CatalogEntry, float]]:
    """Os documentos do tipo pedido (o id da taxonomia; `documento` = qualquer um), com a fração dos qualificadores no
    título ou na classificação, do mais recente ao mais antigo (ADR-047, ADR-049)."""
    kind = request.kind
    wanted = tuple(dict.fromkeys(singular(normalize(term)) for term in request.terms if normalize(term)))
    picked: list[tuple[CatalogEntry, float]] = []
    for entry in entries:
        if kind != ANY_KIND and entry.document_type != kind:
            continue
        words = title_words(entry.document_title) | taxonomy_words(entry)
        score = sum(term in words for term in wanted) / len(wanted) if wanted else 1.0
        if wanted and score == 0:
            continue
        picked.append((entry, score))
    best = max((score for _, score in picked), default=0.0)
    return sorted(
        ((entry, score) for entry, score in picked if score == best),
        key=lambda p: (-p[0].effective_date.toordinal(), normalize(p[0].document_title), p[0].doc_key),
    )


def card_id(doc_key: str) -> str:
    return f"{doc_key}{CARD_SUFFIX}"


def card_doc_key(chunk_id: str) -> str | None:
    """O `doc_key` de uma ficha; `None` para um trecho do índice."""
    return chunk_id.removesuffix(CARD_SUFFIX) if chunk_id.endswith(CARD_SUFFIX) else None


def card_text(entry: CatalogEntry) -> str:
    """A ficha: título, versão, vigência, abrangência, ramo e o link do original — só o que o catálogo registra."""
    lines = [
        entry.document_title,
        f"Documento do acervo, versão {entry.version}: {_validity(entry)}.",
        f"Abrangência {entry.jurisdiction}; {_branch(entry.branch)}.",
    ]
    if entry.source_url:
        lines.append(f"Original: {entry.source_url}")
    return "\n".join(lines)


def with_cards(
    text_of: Callable[[str], str | None], entry_of: Callable[[str], CatalogEntry | None]
) -> Callable[[str], str | None]:
    """O texto por id, de trecho ou de ficha: a ficha se refaz da entrada do documento (a métrica de fatos da 105)."""

    def text(chunk_id: str) -> str | None:
        doc_key = card_doc_key(chunk_id)
        if doc_key is None:
            return text_of(chunk_id)
        entry = entry_of(doc_key)
        return None if entry is None else card_text(entry)

    return text


def card(
    entry: CatalogEntry,
    *,
    rank: int,
    total: int,
    score: float,
    filters: IndexFilters,
    preferred: Sequence[str] = (),
) -> Chunk:
    """A ficha como `Chunk` da 001, com as marcas que põem a lista à frente dos trechos no reranker (R16)."""
    metadata: dict[str, JSONValue] = {
        "document_id": entry.document_id,
        "document_type": entry.document_type,
        "branch": entry.branch,
        "jurisdiction": entry.jurisdiction,
        "effective_date": entry.effective_date.isoformat(),
        "expiry_date": _iso(entry.expiry_date),
        "superseded_by": entry.superseded_by,
        "source_url": entry.source_url,
        FUSION_SCORE: score,
        CATALOG_RANK: rank,
        **marks(entry, filters, preferred),
    }
    return Chunk(
        chunk_id=card_id(entry.doc_key),
        text=card_text(entry),
        score=score,
        document=entry.document_title,
        version=entry.version,
        status=entry.status.value,
        section=f"ficha do acervo, {rank + 1} de {total}",
        page=None,
        metadata=metadata,
    )


def _validity(entry: CatalogEntry) -> str:
    since = _day(entry.effective_date)
    if entry.status is ValidityStatus.VIGENTE and entry.expiry_date is None:
        return f"em vigor desde {since}"
    label = STATUS_LABELS[entry.status]
    if entry.expiry_date is not None:
        return f"{label}; em vigor de {since} a {_day(entry.expiry_date)}"
    return f"{label}; desde {since}"


def _branch(branch: str | None) -> str:
    return "todos os ramos" if branch is None else f"Ramo {BRANCH_NAMES[branch]}"


def _day(value: date) -> str:
    return value.strftime("%d/%m/%Y")


def _iso(value: date | None) -> str | None:
    return None if value is None else value.isoformat()
