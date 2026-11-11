"""Os eventos das atas (spec 106 G4, US4; design.md, seção 2.2; research R21).

Das fichas conferidas (`corpus/resumos/atas/*.yaml`, ADR-062), por regra, sem modelo:
- **`event`**: um por reunião (a data e o local) e um por item conferido — deliberação ou eleição —, com a página. A
  deliberação cujo tema é uma eleição ("Eleição da Direção Regional gestão 2022-2025") é do tipo `eleicao`. Os
  números ficam só na ficha. O evento leva os trechos da ata na página citada e nas vizinhas (a conferência da ficha
  aceita as vizinhas): é o que a busca entrega e cita;
- **`GERA`**: a versão (a ata) → o evento, com a fonte `<doc_key>#<n>@p<página>`;
- **`DE`**: o evento → as entidades que o `reconhecer` acha no tema e na decisão (na eleição, no órgão e no cargo) e o
  órgão que emite a ata (`EMITIDO_POR`, G3).
Pessoas nunca viram entidade (decisão 2): o nome de quem foi eleito fica só no texto do evento, fora do reconhecimento.
Puro.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from axon_scouting.rag.graph.api import Aliases, reconhecer
from axon_scouting.rag.graph.domain.build.organs import normalize

ELECTION_PREFIXES: Final = ("eleicao", "eleicoes")


@dataclass(slots=True)
class MinutesGraph:
    events: list[dict[str, Any]] = field(default_factory=list)
    edges: list[dict[str, Any]] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)


def events_graph(
    minutes: Sequence[Mapping[str, Any]],
    chunks: Sequence[Mapping[str, Any]],
    versions: Mapping[str, Mapping[str, Any]],
    aliases: Aliases,
    issuers: Mapping[str, Sequence[str]],
) -> MinutesGraph:
    """`versions`: doc_key → {date}; `issuers`: doc_key → os órgãos que emitem a ata."""
    pages: dict[str, list[tuple[int, int, str]]] = {}
    for chunk in chunks:
        start = chunk.get("page_start")
        if isinstance(start, int):
            raw_end = chunk.get("page_end")
            end = raw_end if isinstance(raw_end, int) else start
            pages.setdefault(chunk["doc_key"], []).append((start, end, chunk["chunk_id"]))
    result = MinutesGraph()
    for record in minutes:
        doc_key = str(record.get("doc_key") or "")
        if doc_key not in versions:
            result.dropped.append(f"ficha {doc_key or '?'}: ata fora do catálogo")
            continue
        meeting = record.get("reuniao") or {}
        when = str(meeting.get("data") or versions[doc_key].get("date") or "") or None
        items: list[tuple[str, str, str, int]] = []  # (tipo, texto, texto do reconhecimento, página)
        place = str(meeting.get("local") or "").strip()
        kind = str(meeting.get("tipo") or "").strip()
        items.append(("reuniao", " ".join(p for p in (f"Reunião {kind}".strip(), when or "", place) if p), "", 1))
        for item in record.get("deliberacoes") or []:
            if not item.get("conferido"):
                continue
            tema, decisao = str(item.get("tema") or ""), str(item.get("decisao") or "")
            typed = "eleicao" if normalize(tema).startswith(ELECTION_PREFIXES) else "deliberacao"
            items.append((typed, f"{tema}: {decisao}", f"{tema} {decisao}", _page(item)))
        for item in record.get("eleicoes") or []:
            if not item.get("conferido"):
                continue
            orgao, cargo = str(item.get("orgao") or ""), str(item.get("cargo") or "")
            mandato = str(item.get("mandato") or "").strip()
            text = f"Eleição para {orgao}: {item.get('eleito') or ''} ({cargo})" + (
                f", mandato {mandato}" if mandato else ""
            )
            items.append(("eleicao", text, f"{orgao} {cargo}", _page(item)))
        for number, (typed, text, recognized, page) in enumerate(items):
            ident = f"evento:{doc_key}#{number}"
            source = f"{doc_key}#{number}@p{page}"
            # A página citada e as vizinhas, como a conferência da ficha (ADR-062): o item pode começar na anterior.
            covering = [c for start, end, c in pages.get(doc_key, ()) if start <= page + 1 and page - 1 <= end]
            result.events.append(
                {
                    "kind": "event",
                    "id": ident,
                    "doc_key": doc_key,
                    "date": when,
                    "type": typed,
                    "text": text,
                    "page": page,
                    "chunk_ids": covering,
                    "source": source,
                }
            )
            result.edges.append(_edge("GERA", doc_key, ident, source))
            targets = (
                [*reconhecer(normalize(recognized), aliases), *issuers.get(doc_key, ())]
                if recognized
                else [*issuers.get(doc_key, ())]
            )
            for target in dict.fromkeys(targets):
                result.edges.append(_edge("DE", ident, target, source))
    return result


def _page(item: Mapping[str, Any]) -> int:
    page = item.get("pagina")
    return page if isinstance(page, int) and page > 0 else 1


def _edge(kind: str, start: str, end: str, source: str) -> dict[str, Any]:
    return {"kind": "edge", "type": kind, "from": start, "to": end, "source": source, "origin": "regra"}
