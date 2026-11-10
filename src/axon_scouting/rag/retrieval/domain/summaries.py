"""Seleção dos resumos das atas para a pergunta de síntese (spec 103; ADR-062).

Sem vetor: os candidatos saem por metadado — a abrangência pedida, os anos citados e o órgão nomeado (os resumos dele;
sem nenhum, os que falam dele, como as eleições da Diretoria Regional nas atas da Assembleia) — e a ordem, pelos termos
da pergunta (plural reduzido ao singular, frequência saturada e peso pela raridade entre os candidatos, como no BM25),
mais bônus: resumo do ano citado, dos anos mais recentes quando a pergunta pede "as últimas", e a parte de abertura da
ficha (data e local) quando ela pergunta "onde". Para as vagas se espalharem pelo período, no máximo
duas partes de cada ano e duas de cada reunião, e a melhor parte de cada uma antes da segunda de qualquer outra. As
linhas do tempo vêm primeiro, em ordem de ano; depois as fichas.
Puro: sem I/O, sem relógio.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from axon_scouting.rag.index.api import SummaryEntry
from axon_scouting.rag.retrieval.domain.terms import bag, idf, normalize, query_terms, score, stem

MAX_TIMELINES: Final = 6
"""Linhas do tempo entregues: seis anos cabem inteiros; mais que isso, as que mais casam com a pergunta."""
MAX_SUMMARIES: Final = 10
"""Resumos entregues no total — metade das vagas da instalação (20); a outra metade fica com os trechos das atas."""
WINDOW_AFTER: Final = 3
"""Com um ano só citado, a janela vai até três anos depois ("o eleito em 2022 foi reeleito na assembleia seguinte?")."""
PER_GROUP: Final = 2
"""Partes de uma mesma linha do tempo (órgão e ano) ou de uma mesma ficha (reunião) entregues no máximo."""
RECENCY: Final = frozenset({"ultima", "ultimo", "ultimas", "ultimos", "recente", "recentes", "atual", "atuais"})
PLACE: Final = frozenset({"onde", "cidade", "cidades", "local", "locais", "lugar"})


@dataclass(frozen=True, slots=True, kw_only=True)
class SynthesisRequest:
    year_from: int | None
    year_until: int | None
    scopes: tuple[str, ...]
    bodies: tuple[str, ...]


def select(entries: Sequence[SummaryEntry], request: SynthesisRequest, query: str) -> list[SummaryEntry]:
    """Até `MAX_SUMMARIES` resumos: as linhas do tempo (até `MAX_TIMELINES`) e as fichas, cada grupo pela pontuação."""
    low, high = _window(request)
    candidates = [
        entry
        for entry in entries
        if (not request.scopes or entry.scope in request.scopes)
        and (low is None or (entry.year is not None and low <= entry.year <= (high or entry.year)))
    ]
    if not candidates:
        return []
    candidates = _of_bodies(candidates, request.bodies)
    words = normalize(query).split()
    terms = query_terms(query)
    bags = {entry.summary_id: _bag(entry) for entry in candidates}
    weights = idf(list(bags.values()), terms)
    years = [entry.year for entry in candidates if entry.year is not None]
    span = (min(years), max(years)) if years and RECENCY & set(words) else None
    place = bool(PLACE & set(words))
    scored = sorted(
        ((_score(entry, bags[entry.summary_id], terms, weights, request, span, place), entry) for entry in candidates),
        key=lambda pair: (-pair[0], pair[1].summary_id),
    )
    timelines = _spread([entry for score, entry in scored if entry.kind == "linha_do_tempo" and score > 0])
    timelines = timelines[:MAX_TIMELINES]
    timelines.sort(key=lambda entry: (entry.year or 0, entry.summary_id))
    cards = _spread([entry for score, entry in scored if entry.kind == "ficha" and score > 0])
    return [*timelines, *cards][:MAX_SUMMARIES]


def _of_bodies(candidates: list[SummaryEntry], bodies: tuple[str, ...]) -> list[SummaryEntry]:
    """Com órgão nomeado: os resumos do órgão; sem nenhum, os que falam dele; sem nenhum, todos."""
    if not bodies:
        return candidates
    own = [entry for entry in candidates if set(bodies) & set(entry.bodies)]
    if own:
        return own
    mentioning = [entry for entry in candidates if any(_mentions(_bag(entry), body) for body in bodies)]
    return mentioning or candidates


def _spread(ranked: list[SummaryEntry]) -> list[SummaryEntry]:
    """A melhor parte de cada linha do tempo ou ficha primeiro, depois as segundas — no máximo `PER_GROUP` por grupo —,
    para um período longo ter todos os anos antes de um ano ter duas partes."""
    rounds: list[list[SummaryEntry]] = [[] for _ in range(PER_GROUP)]
    taken: dict[str, int] = {}
    for entry in ranked:
        group = entry.summary_id.split("~", 1)[0]
        count = taken.get(group, 0)
        if count < PER_GROUP:
            taken[group] = count + 1
            rounds[count].append(entry)
    return [entry for parts in rounds for entry in parts]


def _window(request: SynthesisRequest) -> tuple[int | None, int | None]:
    if request.year_from is None:
        return None, None
    if request.year_until is None or request.year_until == request.year_from:
        return request.year_from - 1, request.year_from + WINDOW_AFTER
    return request.year_from, request.year_until


def _score(
    entry: SummaryEntry,
    counts: dict[str, int],
    terms: list[str],
    weights: dict[str, float],
    request: SynthesisRequest,
    span: tuple[int, int] | None,
    place: bool,
) -> float:
    points = score(counts, terms, weights)
    if points <= 0:
        return 0.0
    if request.year_from is not None and entry.year in (request.year_from, request.year_until):
        points += 0.5
    if span is not None and entry.year is not None and span[1] > span[0]:
        points += 1.5 * (entry.year - span[0]) / (span[1] - span[0])
    if entry.kind == "ficha" and (entry.summary_id.endswith("#ficha") or entry.summary_id.endswith("#ficha~1")):
        points += 1.0 if place else 0.3
    return points


def _mentions(counts: dict[str, int], body: str) -> bool:
    """O resumo fala do órgão nomeado, mesmo sendo de outro: as eleições da Diretoria Regional estão nas atas da
    Assembleia Regional."""
    return all(stem(word) in counts for word in body.split("_") if len(word) >= 3)


def _bag(entry: SummaryEntry) -> dict[str, int]:
    return bag(f"{entry.title} {entry.text}")
