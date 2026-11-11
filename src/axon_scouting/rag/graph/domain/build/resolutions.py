"""As resoluções no grafo (spec 106 G5, US5; design.md, seção 2.3; research R22).

Por regra, sem modelo:
- **ementa**: as linhas depois do cabeçalho ("RESOLUÇÃO CAN 03/2021", "INSTRUÇÃO NORMATIVA 2024.02") até o
  "Considerando" (ou a fórmula de quem resolve, ou o primeiro artigo), na primeira página. O título no catálogo é só o
  número; o assunto da resolução é o que a ementa nomeia (FR-020);
- **número**: a série (`can`, `den`, `can-den`, `regional`, `in`, `ir`), o ano e o número de cada resolução, pelo
  título do catálogo — os apelidos de `doc:` ("resolucao can 15 2025") e o alvo de "revoga a Resolução CAN 09/2019";
- **`ALTERA` de regra do POR**: "Altera a Regra 178 do P.O.R." liga a resolução à regra do POR vigente **pelo título**
  que a resolução reproduz ("REGRA 178 – DISTINTIVO ADICIONAL DE IDIOMAS" → a regra 183, "DISTINTIVOS DE IDIOMAS"),
  porque a numeração do POR muda (105 R26). Pelo número só quando a resolução é posterior ao POR vigente; senão, fica
  de fora, com o motivo;
- **`ALTERA` e `REVOGA` entre resoluções**: pelo número citado depois do verbo ("revoga a Resolução CAN 009/2019",
  "Revogar a Resolução Regional 2024.11"); sem a série, a da própria resolução;
- **`CITA`**: "Regra 054" e "art. 27 do Estatuto" fora de uma alteração, da unidade que cita para a regra ou o artigo
  vigente — pelo número, só quando a resolução não é anterior à norma citada;
- **`SUBSTITUI`**: a versão mais nova de um documento → a anterior, pela linhagem do catálogo.
A fonte de cada ligação é o trecho em que ela está escrita. Puro.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from itertools import pairwise
from typing import Any, Final

from axon_scouting.rag.graph.domain.build.organs import normalize

RESOLUTION: Final = "resolucao"
LINEAGE_SOURCE: Final = "corpus.yaml:"
TITLE_MATCH: Final = 0.5
"""O Jaccard mínimo entre o título que a resolução reproduz e o da regra do POR (os certos dão 0,6 ou mais; os
errados, 0,4 ou menos, na medida de 2026-11)."""

_CONTEXT: Final = re.compile(r"^\[[^\]\n]*\]\n")
_HEADING: Final = re.compile(
    r"^\W{0,3}(RESOLU[ÇC][ÃA]O|INSTRU[ÇC][ÃA]O\s+(?:NORMATIVA|REGIONAL))\b"
    r"[^\n\d]{0,40}?(?:\d{1,4}(?:\s*[./]\s*\d{1,4})?(?:\s*,?\s*DE\s+\d{1,2}\s+DE\s+[A-ZÇ]+\s+DE\s+\d{4})?|$)",
    re.IGNORECASE,
)
_NUMBER_LINE: Final = re.compile(r"^\d{1,4}\s*[./]\s*\d{1,4}\s*$")
_EMENTA_END: Final = re.compile(
    r"^(considerando|cap[ií]tulo\s|o\s+conselho|a\s+diretoria|o\s+diretor|resolve|art\.?\s*\d|artigo\s|\d+\s*[ºo°]\s*[-–])",
    re.IGNORECASE,
)
_EMENTA_MAX: Final = 400
_ALTERS: Final = re.compile(r"\b(alter[ae]\w*|retific\w*|nova\s+reda[çc][ãa]o)\b", re.IGNORECASE)
_REVOKES: Final = re.compile(r"\brevog\w*", re.IGNORECASE)
_WINDOW_END: Final = re.compile(r"\.(?=\s+[A-ZÀ-Ü])|;|\n\s*(?=art|§|par[áa]grafo|\d+\s*[ºo°])", re.IGNORECASE)
_WINDOW: Final = 220
_RULES: Final = re.compile(r"\bregras?\s+(?:n[ºo°.]*\s*)?(\d{1,3}(?:\s*(?:,|e)\s*\d{1,3})*)", re.IGNORECASE)
_RULE_TITLE: Final = re.compile(r"REGRA\s+(\d{1,3})\s*[–—-]?\s*([^\n]+)")
_ARTICLE: Final = re.compile(r"\bart(?:igo)?s?\.?\s*(\d{1,3})\s*[ºo°]?[^.;\n]{0,40}?\bdo\s+estatuto\b", re.IGNORECASE)
_REFERENCE: Final = re.compile(
    r"\b(?:resolu[çc](?:[ãa]o|[õo]es)|RR)\s+(?:(CAN-DEN|CAN|DEN|regional(?:\s+SP)?)\s+)?(?:n[ºo°ᵒ.]*\s*)?"
    r"((?:\d{1,4}\s*[./ ]\s*\d{1,4}(?:\s*(?:,|;|e)\s*(?=\d))?)+)"
    r"(?:\s+(?:do|da)\s+(CAN|DEN))?",
    re.IGNORECASE,
)
_PAIR: Final = re.compile(r"(\d{1,4})\s*[./ ]\s*(\d{1,4})")
_DASHES: Final = frozenset({"–", "—", "-"})
_ITEM: Final = re.compile(r"^([IVX]{1,4}[.)]?|[A-Z][.)])$")
_PASSIVE: Final = re.compile(r"^\s*,?\s*(conforme|pel[oa]s?|atrav[ée]s\s+d[ao]s?)\b", re.IGNORECASE)
_STOP: Final = frozenset({"de", "da", "do", "das", "dos", "e", "a", "o", "as", "os", "para", "em", "no", "na", "com"})
_TITLE_NUMBER: Final = (
    (re.compile(r"^Resolução (CAN-DEN|CAN|DEN) (\d+)/(\d{4})", re.IGNORECASE), "series"),
    (re.compile(r"^Resolução Regional(?: SP)? (\d+)/(\d{4})", re.IGNORECASE), "regional"),
    (re.compile(r"^Instrução Normativa (\d{4})\.(\d+)", re.IGNORECASE), "in-year"),
    (re.compile(r"^Instrução Normativa (\d+)/(\d{4})", re.IGNORECASE), "in"),
    (re.compile(r"^Instrução Normativa (\d+)\b", re.IGNORECASE), "in-undated"),
    (re.compile(r"^Instrução Regional (\d+)/(\d{4})", re.IGNORECASE), "ir"),
)
SERIES_OF_BODY: Final = {
    "conselho_de_administracao_nacional": "can",
    "diretoria_executiva_nacional": "den",
    "diretoria_regional": "regional",
}

Number = tuple[str, int, int]
"""(série, ano, número)."""


def ementa(first_page: str) -> str | None:
    """A ementa: o resto da linha do cabeçalho (a linha que começa por "RESOLUÇÃO" ou "INSTRUÇÃO", com o número) e as
    linhas seguintes até o "Considerando" (no máximo 400 caracteres), sem quebras. Sem cabeçalho na página, `None`."""
    lines = [line.strip() for line in first_page.replace("\r", "").split("\n")]
    for index, line in enumerate(lines):
        match = _HEADING.match(line)
        if match is None:
            continue
        following_lines = lines[index + 1 :]
        if not any(ch.isdigit() for ch in match.group(0)):  # "RESOLUÇÃO No" e o número na linha de baixo
            if not following_lines or not _NUMBER_LINE.match(following_lines[0]):
                continue
            following_lines = following_lines[1:]
        body: list[str] = []
        rest = line[match.end() :].strip(" –—-:")
        if rest and not _EMENTA_END.match(rest):
            body.append(rest)
        for following in following_lines:
            if not following:
                continue
            if _EMENTA_END.match(following) or len(" ".join(body)) >= _EMENTA_MAX:
                break
            body.append(following)
        text = " ".join(" ".join(body).split())
        return text or None
    return None


def number_of(title: str, effective: str | None) -> Number | None:
    """O número da resolução pelo título do catálogo; a instrução normativa sem ano usa o da vigência."""
    for pattern, kind in _TITLE_NUMBER:
        match = pattern.match(title.strip())
        if match is None:
            continue
        if kind == "series":
            return (match.group(1).lower(), int(match.group(3)), int(match.group(2)))
        if kind == "regional":
            return ("regional", int(match.group(2)), int(match.group(1)))
        if kind == "in-year":
            return ("in", int(match.group(1)), int(match.group(2)))
        if kind == "in":
            return ("in", int(match.group(2)), int(match.group(1)))
        if kind == "in-undated":
            year = int(effective[:4]) if effective else None
            return ("in", year, int(match.group(1))) if year else None
        return ("ir", int(match.group(2)), int(match.group(1)))
    return None


def number_aliases(number: Number) -> tuple[str, ...]:
    """Os nomes normalizados pelos quais uma pergunta cita a resolução: com e sem os zeros à esquerda, "no" (nº), e a
    forma ano.número das regionais e das instruções."""
    series, year, num = number
    nums = list(dict.fromkeys((str(num), f"{num:02d}", f"{num:03d}")))
    if series in ("can", "den", "can-den"):
        heads = [f"resolucao {series.replace('-', ' ')}", f"resolucao {series.replace('-', ' ')} no"]
    elif series == "regional":
        heads = ["resolucao regional", "resolucao regional sp", "resolucao regional no"]
    elif series == "in":
        heads = ["instrucao normativa", "instrucao normativa no"]
    else:
        heads = ["instrucao regional"]
    found = [f"{head} {n} {year}" for head in heads for n in nums]
    if series in ("regional", "in"):
        found += [f"{head} {year} {n}" for head in heads for n in nums]
    return tuple(dict.fromkeys(found))


@dataclass(frozen=True, slots=True, kw_only=True)
class Resolution:
    doc_key: str
    document_id: str
    title: str
    number: Number | None
    series: str | None
    date: date | None
    ementa: str | None
    first_chunk: str | None


@dataclass(slots=True)
class Links:
    edges: list[dict[str, Any]] = field(default_factory=list)
    units: set[str] = field(default_factory=set)
    """As unidades que as ligações usam (a regra do POR, o artigo do Estatuto, o artigo da resolução que cita)."""
    dropped: list[str] = field(default_factory=list)


def resolutions_of(documents: Sequence[Mapping[str, Any]]) -> list[Resolution]:
    found: list[Resolution] = []
    for doc in documents:
        fields = doc["fields"]
        if fields.get("document_type") != RESOLUTION:
            continue
        bodies = list(fields.get("bodies") or ())
        effective = fields.get("effective_date")
        pages = doc.get("pages") or [""]
        chunk_ids = doc.get("chunk_ids") or []
        found.append(
            Resolution(
                doc_key=str(doc["doc_key"]),
                document_id=str(fields["document_id"]),
                title=str(fields.get("document_title") or ""),
                number=number_of(str(fields.get("document_title") or ""), effective),
                series=SERIES_OF_BODY.get(bodies[0]) if bodies else None,
                date=date.fromisoformat(effective) if effective else None,
                ementa=ementa(str(pages[0] or "")),
                first_chunk=str(chunk_ids[0]) if chunk_ids else None,
            )
        )
    return found


@dataclass(frozen=True, slots=True, kw_only=True)
class NormUnits:
    """As unidades da norma vigente que uma resolução altera ou cita: número → (unit_id, título normalizado), com a
    vigência da norma."""

    units: Mapping[int, tuple[str, frozenset[str]]]
    effective: date | None


def norm_units(chunks: Sequence[Mapping[str, Any]], doc_key: str | None, kind: str) -> NormUnits:
    """As regras (`kind="regra"`, com o cabeçalho do texto como título) ou os artigos de uma norma, pelo caminho dos
    trechos."""
    found: dict[int, tuple[str, frozenset[str]]] = {}
    effective: date | None = None
    for chunk in chunks:
        if doc_key is None or chunk["doc_key"] != doc_key:
            continue
        if effective is None and chunk.get("effective_date"):
            effective = date.fromisoformat(str(chunk["effective_date"]))
        for step in chunk.get("path") or []:
            if step.get("kind") != kind:
                continue
            number = re.search(r"(\d{1,3})", str(step.get("title") or ""))
            if number is None or int(number.group(1)) in found:
                continue
            title = ""
            if kind == "regra":
                heading = re.search(rf"REGRA\s+0*{int(number.group(1))}\s+([^\n|]+)", chunk["text"])
                title = heading.group(1) if heading else ""
            found[int(number.group(1))] = (str(step["unit_id"]), _words(title))
    return NormUnits(units=found, effective=effective)


def resolution_links(
    resolutions: Sequence[Resolution],
    chunks: Sequence[Mapping[str, Any]],
    rules: NormUnits,
    articles: NormUnits,
) -> Links:
    """As ligações escritas no texto das resoluções: `ALTERA` (regra do POR ou outra resolução), `REVOGA` e `CITA`.
    Uma aresta por (tipo, origem, destino), com a fonte no primeiro trecho que a escreve."""
    links = Links()
    by_number = {r.number: r for r in resolutions if r.number is not None}
    by_doc = {r.doc_key: r for r in resolutions}
    seen: set[tuple[str, str, str]] = set()
    titles: dict[str, dict[int, frozenset[str]]] = {}
    for chunk in chunks:
        resolution = by_doc.get(chunk["doc_key"])
        if resolution is None:
            continue
        for match in _RULE_TITLE.finditer(_CONTEXT.sub("", chunk["text"])):
            titles.setdefault(resolution.doc_key, {}).setdefault(int(match.group(1)), _words(match.group(2)))

    def add(kind: str, start: str, end: str, source: str, **attrs: str) -> None:
        if start == end or (kind, start, end) in seen:
            return
        seen.add((kind, start, end))
        edge: dict[str, Any] = {"kind": "edge", "type": kind, "from": start, "to": end, "source": source}
        edge["origin"] = "regra"
        if attrs:
            edge["attrs"] = attrs
        links.edges.append(edge)

    def between(passive: bool, kind: str, current: Resolution, other: Resolution, source: str) -> None:
        """ "Altera a Resolução X" (a atual → X); "alterada pela Resolução X" (X → a atual). A mais antiga (pelo ano do
        número, e não pela vigência, que no texto consolidado é a da consolidação) não altera nem revoga a mais nova:
        o caso fica de fora, com o motivo."""
        start, end = (other, current) if passive else (current, other)
        if start.number is not None and end.number is not None and start.number[1] < end.number[1]:
            links.dropped.append(f"{kind} {start.doc_key} → {end.doc_key}: a de origem é anterior à de destino")
            return
        add(kind, start.doc_key, end.doc_key, source, numero=_label(other))

    for chunk in chunks:
        resolution = by_doc.get(chunk["doc_key"])
        if resolution is None:
            continue
        text, chunk_id = _CONTEXT.sub("", chunk["text"]), str(chunk["chunk_id"])
        for verb in _ALTERS.finditer(text):
            window = _window(text, verb.end())
            for number in _rule_numbers(window):
                target, how = _rule_target(number, titles.get(resolution.doc_key, {}), rules, resolution.date)
                if target is None:
                    links.dropped.append(f"ALTERA {resolution.doc_key} → Regra {number}: {how}")
                    continue
                links.units.add(target)
                add("ALTERA", resolution.doc_key, target, chunk_id, regra=str(number), casamento=how)
            for other in _references(window, resolution.series, by_number):
                between(_PASSIVE.match(window) is not None, "ALTERA", resolution, other, chunk_id)
        for verb in _REVOKES.finditer(text):
            window = _window(text, verb.end())
            for other in _references(window, resolution.series, by_number):
                between(_PASSIVE.match(window) is not None, "REVOGA", resolution, other, chunk_id)
    altered = {(start, end) for kind, start, end in seen if kind == "ALTERA"}
    for chunk in chunks:  # a citação: a regra ou o artigo que a resolução menciona sem alterar
        resolution = by_doc.get(chunk["doc_key"])
        path = chunk.get("path") or []
        if resolution is None or not path:
            continue
        text, chunk_id = _CONTEXT.sub("", chunk["text"]), str(chunk["chunk_id"])
        citing = str(path[-1]["unit_id"])
        for number in _rule_numbers(text):
            target, how = _rule_target(number, titles.get(resolution.doc_key, {}), rules, resolution.date)
            if target is not None and (resolution.doc_key, target) not in altered:
                links.units.update((citing, target))
                add("CITA", citing, target, chunk_id, regra=str(number), casamento=how)
        for match in _ARTICLE.finditer(text):
            number = int(match.group(1))
            unit = articles.units.get(number)
            if unit is None or not _not_older(resolution.date, articles.effective):
                continue
            links.units.update((citing, unit[0]))
            add("CITA", citing, unit[0], chunk_id, artigo=str(number), casamento="numero")
    return links


@dataclass(slots=True)
class Subjects:
    found: dict[str, tuple[tuple[str, ...], str]] = field(default_factory=dict)
    """doc_key → (as entidades, o trecho em que a frase citada está)."""
    dropped: list[str] = field(default_factory=list)


def checked_subjects(proposals: Mapping[str, Any], chunks: Sequence[Mapping[str, Any]], entities: set[str]) -> Subjects:
    """As propostas de assunto do modelo offline (FR-020), conferidas por código: toda entidade no catálogo curado e o
    trecho citado, normalizado, dentro de um trecho da própria resolução — a fonte da aresta. Proposta que falha fica
    de fora, com o motivo; nenhuma entra sem conferência (FR-003)."""
    texts: dict[str, list[tuple[str, str]]] = {}
    for chunk in chunks:
        texts.setdefault(str(chunk["doc_key"]), []).append(
            (str(chunk["chunk_id"]), normalize(_CONTEXT.sub("", chunk["text"])))
        )
    result = Subjects()
    for entry in proposals.get("propostas") or []:
        if not isinstance(entry, Mapping):
            continue
        key, quote = str(entry.get("doc_key") or ""), normalize(str(entry.get("trecho") or ""))
        named = tuple(str(e) for e in entry.get("entidades") or ())
        unknown = [e for e in named if e not in entities]
        where = next((chunk_id for chunk_id, text in texts.get(key, ()) if quote and f" {quote} " in f" {text} "), None)
        if not named or unknown:
            result.dropped.append(f"assunto de {key}: entidade fora do catálogo {', '.join(unknown) or '(nenhuma)'}")
        elif where is None:
            result.dropped.append(f"assunto de {key}: o trecho citado não está na resolução")
        else:
            result.found[key] = (named, where)
    return result


def lineage_edges(documents: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """`SUBSTITUI`: em cada documento com mais de uma versão, a mais nova → a anterior (pela vigência)."""
    versions: dict[str, list[tuple[str, str]]] = {}
    for doc in documents:
        fields = doc["fields"]
        versions.setdefault(str(fields["document_id"]), []).append(
            (str(fields.get("effective_date") or ""), str(doc["doc_key"]))
        )
    edges: list[dict[str, Any]] = []
    for document_id, found in versions.items():
        ordered = sorted(found)
        for (_, older), (_, newer) in pairwise(ordered):
            edges.append(
                {
                    "kind": "edge",
                    "type": "SUBSTITUI",
                    "from": newer,
                    "to": older,
                    "source": f"{LINEAGE_SOURCE}{document_id}",
                    "origin": "regra",
                }
            )
    return edges


def _window(text: str, start: int) -> str:
    """O trecho depois do verbo, até o fim da frase (no máximo 220 caracteres)."""
    window = text[start : start + _WINDOW]
    end = _WINDOW_END.search(window)
    return window[: end.start()] if end else window


def _rule_numbers(text: str) -> list[int]:
    return [int(n) for match in _RULES.finditer(text) for n in re.findall(r"\d{1,3}", match.group(1))]


def _rule_target(
    number: int, titles: Mapping[int, frozenset[str]], rules: NormUnits, when: date | None
) -> tuple[str | None, str]:
    """A regra do POR vigente pelo título que a resolução reproduz; sem título, pelo número só quando a resolução
    não é anterior ao POR. Sem alvo: `(None, motivo)`."""
    title = titles.get(number)
    if title:
        best, score = None, 0.0
        for candidate, (unit_id, words) in rules.units.items():
            if not words:
                continue
            value = len(title & words) / len(title | words)
            if value > score or (value == score and candidate == number):
                best, score = unit_id, value
        if best is not None and score >= TITLE_MATCH:
            return best, "titulo"
        return None, "título sem regra correspondente no POR vigente"
    if number in rules.units and _not_older(when, rules.effective):
        return rules.units[number][0], "numero"
    return None, "sem título, e a numeração é de antes do POR vigente"


def _references(window: str, series: str | None, by_number: Mapping[Number, Resolution]) -> list[Resolution]:
    found: list[Resolution] = []
    for match in _REFERENCE.finditer(window):
        named = (match.group(1) or match.group(3) or "").lower()
        kind = "regional" if named.startswith("regional") or match.group(0).upper().startswith("RR") else named
        kind = kind or series or ""
        for first, second in _PAIR.findall(match.group(2)):
            a, b = int(first), int(second)
            year, num = (a, b) if a >= 1900 else (b, a)
            if year < 100:
                year += 1900 if year >= 50 else 2000
            other = by_number.get((kind, year, num))
            if other is not None:
                found.append(other)
    return found


def _label(resolution: Resolution) -> str:
    if resolution.number is None:
        return resolution.title
    series, year, num = resolution.number
    return f"{series} {num:02d}/{year}"


def _not_older(when: date | None, norm: date | None) -> bool:
    return when is not None and norm is not None and when >= norm


def _words(title: str) -> frozenset[str]:
    """As palavras de um título, sem as de ligação e sem as letras soltas do corpo que o seguem ("DEFINIÇÃO DO
    ESCOTISMO O E" → definicao, escotismo), no singular grosseiro (sem o "s" final)."""
    upper: list[str] = []
    for word in title.split():  # o título em caixa alta acaba no corpo (minúscula) ou no primeiro item ("I –", "A.")
        if any(ch.islower() for ch in word) or word in _DASHES or _ITEM.match(word):
            break
        upper.append(word)
    words = normalize(" ".join(upper)).split()
    while words and len(words[-1]) <= 2:
        words.pop()
    return frozenset(w.rstrip("s") for w in words if w not in _STOP)
