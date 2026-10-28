"""Ruído de OCR (spec 102 FR-034; ADR-050). Puro: as linhas limpas e o vocabulário chegam prontos.

A transcrição automática de uma ata escaneada traz o carimbo do cartório lido como texto ("(XX) 41 - 3225-3905
CURITIBA", "REG. TÍTULOS E DOCUYMENTOS") e restos de letras sob o carimbo ("liZadoO FICIO", "pagéstoNO «;"). O
carimbo em maiúsculas vira título e o sumário do documento; o resto polui o trecho. Aqui, só nos documentos de
`ocr/`:

- o **carimbo**: linha curta cujas palavras reaparecem (Jaccard ≥ 0,6) em 3 páginas ou mais do mesmo documento e que
  tem cara de carimbo ou timbre — telefone, CEP ou CNPJ, ou metade das palavras do léxico de cartório e endereço
  (`STAMP_WORDS`); "RAMO LOBINHO" repetido num relatório escaneado é conteúdo e fica;
- a **linha ilegível**: com 2 palavras ou mais, menos da metade plausível;
- o **rabo ilegível**: no fim de uma linha, 2 palavras ou mais seguidas não plausíveis saem.

Palavra plausível: está no vocabulário (as palavras de 2 documentos nascidos digitais ou mais, dobradas), ou é nome
próprio ou sigla bem formados (maiúscula inicial ou toda em maiúsculas, com vogal), ou é número. O que o filtro tira
fica no original (a transcrição); só não chega ao trecho.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence
from typing import Final

from axon_scouting.rag.ingestion.domain.text import Line

MIN_WORD: Final = 3
PLAUSIBLE_SHARE: Final = 0.5
STAMP_PAGES: Final = 3
STAMP_SIMILARITY: Final = 0.6
STAMP_MAX_CHARS: Final = 90
GARBAGE_TAIL: Final = 2
STAMP_WORDS: Final = frozenset(
    {
        "registro",
        "registrado",
        "registros",
        "titulos",
        "documentos",
        "cartorio",
        "oficio",
        "distribuidor",
        "escrevente",
        "autorizada",
        "autorizado",
        "tabeliao",
        "tabelionato",
        "srtd",
        "rtd",
        "civil",
        "pessoas",
        "juridicas",
        "protocolo",
        "averbacao",
        "microfilme",
        "selo",
        "autenticacao",
        "cnpj",
        "cep",
        "fone",
        "tel",
        "fax",
        "caixa",
        "postal",
        "rua",
        "avenida",
        "sala",
        "bairro",
        "curitiba",
        "parana",
        "andar",
    }
)
"""Palavras de carimbo de cartório e de timbre (dobradas)."""
_CONTACT: Final = re.compile(r"\(\d{2}\)|\d{4,5}[-.]\d{4}|\d{5}-\d{3}|\d{2}\.\d{3}\.\d{3}/")
"""Telefone, CEP ou CNPJ."""
_WORD: Final = re.compile(r"[^\W\d_]+(?:[-'][^\W\d_]+)*|\d[\d.,/ºª°-]*", re.UNICODE)
_VOWELS: Final = frozenset("aeiouáéíóúâêôãõàü")
_NUMBER: Final = re.compile(r"^\d[\d.,/ºª°-]*$")


def fold(word: str) -> str:
    decomposed = unicodedata.normalize("NFKD", word.casefold())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def words(text: str) -> list[str]:
    return _WORD.findall(text)


def vocabulary(texts: Iterable[Iterable[str]], min_documents: int = 2) -> frozenset[str]:
    """As palavras (dobradas, com 3 letras ou mais) que aparecem em `min_documents` documentos ou mais; cada item de
    `texts` são as linhas de um documento."""
    seen: dict[str, int] = {}
    for lines in texts:
        found = {fold(word) for line in lines for word in words(line) if len(word) >= MIN_WORD and word.isalpha()}
        for word in found:
            seen[word] = seen.get(word, 0) + 1
    return frozenset(word for word, count in seen.items() if count >= min_documents)


def plausible(word: str, known: frozenset[str]) -> bool:
    if _NUMBER.match(word):
        return True
    letters = [ch for ch in word if ch.isalpha()]
    if not letters:
        return False
    if len(letters) < MIN_WORD:
        return True  # artigo, preposição, sigla curta: não decide
    if fold(word) in known:
        return True
    if not any(ch in _VOWELS for ch in fold(word)):
        return False
    rest = word[1:]
    capitalized = word[0].isupper() and (rest.islower() or not any(ch.isalpha() for ch in rest))
    return capitalized or word.isupper()


def clean(lines: Sequence[Line], known: frozenset[str]) -> list[Line]:
    """As linhas sem carimbo, sem linha ilegível e sem rabo ilegível; linha vazia (fronteira) fica."""
    stamps = _stamps(lines, known)
    result: list[Line] = []
    for position, line in enumerate(lines):
        if not line.text:
            result.append(line)
            continue
        if position in stamps:
            continue
        found = words(line.text)
        judged = [word for word in found if len([ch for ch in word if ch.isalpha()]) >= MIN_WORD]
        if len(judged) >= 2 and sum(plausible(w, known) for w in judged) < PLAUSIBLE_SHARE * len(judged):
            continue
        text = _without_tail(line.text, known)
        if text:
            result.append(Line(text=text, page=line.page))
    return result


def _signature(text: str, known: frozenset[str]) -> frozenset[str]:
    return frozenset(fold(w) for w in words(text) if len(w) >= MIN_WORD or _NUMBER.match(w))


def _stamps(lines: Sequence[Line], known: frozenset[str]) -> set[int]:
    """As posições das linhas curtas que reaparecem, quase iguais, em `STAMP_PAGES` páginas ou mais."""
    short = [
        (position, line.page, _signature(line.text, known))
        for position, line in enumerate(lines)
        if line.text and len(line.text) <= STAMP_MAX_CHARS
    ]
    short = [item for item in short if len(item[2]) >= 2]
    found: set[int] = set()
    for position, page, signature in short:
        pages = {page}
        for _, other_page, other in short:
            if other_page not in pages and _jaccard(signature, other) >= STAMP_SIMILARITY:
                pages.add(other_page)
        if len(pages) >= STAMP_PAGES and _looks_like_stamp(lines[position].text, signature):
            found.add(position)
    return found


def _looks_like_stamp(text: str, signature: frozenset[str]) -> bool:
    return bool(_CONTACT.search(text)) or len(signature & STAMP_WORDS) * 2 >= len(signature)


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    return len(a & b) / len(a | b) if a or b else 0.0


def _without_tail(text: str, known: frozenset[str]) -> str:
    """Tira do fim as palavras não plausíveis seguidas (2 ou mais) e os símbolos soltos depois delas."""
    tokens = text.split()
    bad = 0
    for token in reversed(tokens):
        core = "".join(words(token))
        if core and plausible(core, known):
            break
        bad += 1
    if bad < GARBAGE_TAIL:
        return text
    return " ".join(tokens[: len(tokens) - bad])
