"""Fichas de reunião e linha do tempo: a busca global sobre as atas (ADR-062).

Uma ficha é o registro estruturado de uma ata — tipo, data e local da reunião, deliberações, eleições e números, cada
item com a página da ata — extraído offline por um modelo e guardado em `corpus/resumos/atas/<doc_key>.yaml`. Aqui
ficam as políticas puras sobre ela: ler e validar o registro, **conferir** cada item contra o texto da página (o item
que o texto não sustenta não entra no índice) e montar o que a busca entrega — o texto da ficha e a linha do tempo por
órgão e ano, sem modelo. Sem I/O, sem relógio.
"""

from __future__ import annotations

import dataclasses
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Final

MEETING_KINDS: Final = ("ordinaria", "extraordinaria", "outra")
KIND_LABELS: Final = {"ordinaria": "ordinária", "extraordinaria": "extraordinária", "outra": ""}
EVIDENCE_SHARE: Final = 0.6
"""Fração das palavras de conteúdo do item que precisa aparecer na página citada (ou nas vizinhas)."""
PART_CHARS: Final = 1800
"""Teto de uma parte de ficha ou de linha do tempo: ~450 tokens, dentro do teto dos trechos (ADR-050)."""
TIMELINE_DECISION_CHARS: Final = 180
"""Na linha do tempo, a decisão vai até aqui; o texto inteiro fica na ficha da reunião."""
PROCEDURAL: Final = (
    "mesa diretora",
    "mesa dos trabalhos",
    "presidente da mesa",
    "comissoes assessoras",
    "comissao de credenciamento",
    "comissao de escrutinio",
    "comissao de estilo",
    "ata da reuniao anterior",
    "ata anterior",
    "inversao",
    "ordem do dia",
    "pauta",
)
"""Temas de procedimento da própria reunião: ficam na ficha, não na linha do tempo do órgão."""
_STOPWORDS: Final = frozenset(
    [
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
        "pelas",
        "pelos",
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
        "era",
        "como",
        "mais",
        "menos",
        "ate",
        "apos",
        "entre",
        "sua",
        "seu",
        "suas",
        "seus",
        "sua",
        "este",
        "esta",
        "esse",
        "essa",
        "isso",
        "aos",
        "as",
        "pela",
        "pelo",
    ]
)


class RecordError(ValueError):
    """Ficha fora do formato; a mensagem nomeia o campo."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Deliberation:
    tema: str
    decisao: str
    votacao: str = ""
    pagina: int
    conferido: bool = False


@dataclass(frozen=True, slots=True, kw_only=True)
class Election:
    orgao: str
    cargo: str
    eleito: str
    mandato: str = ""
    pagina: int
    conferido: bool = False


@dataclass(frozen=True, slots=True, kw_only=True)
class Figure:
    """Um número que a ata registra (delegados credenciados, votos, delegados eleitos…)."""

    o_que: str
    valor: str
    pagina: int
    conferido: bool = False


@dataclass(frozen=True, slots=True, kw_only=True)
class MeetingRecord:
    doc_key: str
    tipo: str
    data: date | None
    local: str
    deliberacoes: tuple[Deliberation, ...] = ()
    eleicoes: tuple[Election, ...] = ()
    numeros: tuple[Figure, ...] = ()
    extracao: Mapping[str, str] = field(default_factory=dict)
    """Modelo, data e sha256 do texto de onde a ficha saiu: reextrair só quando o texto muda."""
    revisado: str | None = None
    """Quem revisou e quando; `None` enquanto só a checagem automática passou."""


@dataclass(frozen=True, slots=True, kw_only=True)
class MinutesDocument:
    """O que a ficha precisa saber do documento da ata, tirado do catálogo do índice."""

    doc_key: str
    title: str
    category: str
    bodies: tuple[str, ...]
    scope: str | None
    source_url: str | None
    effective_date: date | None


@dataclass(frozen=True, slots=True, kw_only=True)
class SummaryPart:
    """Uma parte de ficha ou de linha do tempo, pronta para virar `SummaryEntry` no índice."""

    summary_id: str
    kind: str
    title: str
    text: str
    year: int | None
    meeting_date: date | None
    doc_keys: tuple[str, ...]
    bodies: tuple[str, ...]
    scope: str | None
    page: int | None
    source_url: str | None


# --- leitura e escrita ----------------------------------------------------------------------------------------------


def record_from_mapping(raw: Mapping[str, Any]) -> MeetingRecord:
    try:
        doc_key = _text(raw, "doc_key", required=True)
        reuniao = raw.get("reuniao") or {}
        if not isinstance(reuniao, Mapping):
            raise RecordError("reuniao: esperado um mapa")
        tipo = _text(reuniao, "tipo") or "outra"
        if tipo not in MEETING_KINDS:
            raise RecordError(f"reuniao.tipo: esperado um de {', '.join(MEETING_KINDS)}, veio {tipo!r}")
        return MeetingRecord(
            doc_key=doc_key,
            tipo=tipo,
            data=_date(reuniao.get("data")),
            local=_text(reuniao, "local"),
            deliberacoes=tuple(
                Deliberation(
                    tema=_text(item, "tema", required=True),
                    decisao=_text(item, "decisao", required=True),
                    votacao=_text(item, "votacao"),
                    pagina=_page(item),
                    conferido=bool(item.get("conferido", False)),
                )
                for item in _items(raw, "deliberacoes")
            ),
            eleicoes=tuple(
                Election(
                    orgao=_text(item, "orgao"),
                    cargo=_text(item, "cargo", required=True),
                    eleito=_text(item, "eleito", required=True),
                    mandato=_text(item, "mandato"),
                    pagina=_page(item),
                    conferido=bool(item.get("conferido", False)),
                )
                for item in _items(raw, "eleicoes")
            ),
            numeros=tuple(
                Figure(
                    o_que=_text(item, "o_que", required=True),
                    valor=_text(item, "valor", required=True),
                    pagina=_page(item),
                    conferido=bool(item.get("conferido", False)),
                )
                for item in _items(raw, "numeros")
            ),
            extracao={str(k): str(v) for k, v in (raw.get("extracao") or {}).items()},
            revisado=None if raw.get("revisado") in (None, "") else str(raw["revisado"]),
        )
    except RecordError as error:
        raise RecordError(f"{raw.get('doc_key', '?')}: {error}") from None


def record_to_mapping(record: MeetingRecord) -> dict[str, Any]:
    return {
        "doc_key": record.doc_key,
        "reuniao": {
            "tipo": record.tipo,
            "data": None if record.data is None else record.data.isoformat(),
            "local": record.local,
        },
        "deliberacoes": [dataclasses.asdict(item) for item in record.deliberacoes],
        "eleicoes": [dataclasses.asdict(item) for item in record.eleicoes],
        "numeros": [dataclasses.asdict(item) for item in record.numeros],
        "extracao": dict(record.extracao),
        "revisado": record.revisado,
    }


# --- conferência -----------------------------------------------------------------------------------------------------


def confirm(record: MeetingRecord, pages: Sequence[str]) -> MeetingRecord:
    """Marca cada item como conferido quando a página citada (ou uma vizinha) contém a maior parte das palavras de
    conteúdo dele. O texto que a ficha repete precisa estar na ata: é o que barra o que o modelo inventou."""
    normalized = [_normalize(page) for page in pages]
    return dataclasses.replace(
        record,
        deliberacoes=tuple(
            dataclasses.replace(item, conferido=_supported(f"{item.decisao}", item.pagina, normalized))
            for item in record.deliberacoes
        ),
        eleicoes=tuple(
            dataclasses.replace(item, conferido=_supported(f"{item.eleito} {item.cargo}", item.pagina, normalized))
            for item in record.eleicoes
        ),
        numeros=tuple(
            dataclasses.replace(
                item, conferido=_supported(item.valor, item.pagina, normalized) and _has_number(item, normalized)
            )
            for item in record.numeros
        ),
    )


def problems(record: MeetingRecord) -> list[str]:
    """Os itens que não passaram na conferência, para a lista de revisão."""
    found: list[str] = []
    for item in record.deliberacoes:
        if not item.conferido:
            found.append(f"deliberação p. {item.pagina}: {item.tema} — {item.decisao}")
    for election in record.eleicoes:
        if not election.conferido:
            found.append(f"eleição p. {election.pagina}: {election.cargo} — {election.eleito}")
    for figure in record.numeros:
        if not figure.conferido:
            found.append(f"número p. {figure.pagina}: {figure.o_que} — {figure.valor}")
    return found


def _supported(text: str, page: int, pages: Sequence[str]) -> bool:
    words = _content_words(text)
    if not words or not 1 <= page <= len(pages):
        return False
    window = " ".join(pages[max(0, page - 2) : page + 1])
    present = sum(1 for word in words if f" {word} " in f" {window} ")
    return present >= max(1, round(EVIDENCE_SHARE * len(words)))


def _has_number(figure: Figure, pages: Sequence[str]) -> bool:
    digits = "".join(ch for ch in figure.valor if ch.isdigit() or ch == " ").split()
    if not digits:
        return True
    window = " ".join(pages[max(0, figure.pagina - 2) : figure.pagina + 1])
    return all(f" {number} " in f" {window} " for number in digits)


def _content_words(text: str) -> list[str]:
    return [word for word in _normalize(text).split() if (len(word) >= 4 or word.isdigit()) and word not in _STOPWORDS]


def _normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    plain = "".join(char for char in decomposed if not unicodedata.combining(char))
    return " ".join("".join(char if char.isalnum() else " " for char in plain).split())


# --- o que a busca entrega -------------------------------------------------------------------------------------------


def card_parts(record: MeetingRecord, document: MinutesDocument) -> list[SummaryPart]:
    """O texto da ficha da reunião, só com os itens conferidos, em partes que cabem no teto."""
    when = record.data or document.effective_date
    lines = [_meeting_line(record, when)]
    lines += [
        f"- Deliberação: {d.tema}: {d.decisao}{_vote(d.votacao)} (p. {d.pagina})"
        for d in record.deliberacoes
        if d.conferido
    ]
    lines += [
        f"- Eleição{_of(e.orgao)}: {e.eleito} ({e.cargo}){_term(e.mandato)} (p. {e.pagina})"
        for e in record.eleicoes
        if e.conferido
    ]
    lines += [f"- Número: {f.valor} {f.o_que} (p. {f.pagina})" for f in record.numeros if f.conferido]
    title = f"Ficha da reunião · {document.title}"
    return _parts(
        base_id=f"{record.doc_key}#ficha",
        kind="ficha",
        title=title,
        lines=lines,
        year=None if when is None else when.year,
        meeting_date=when,
        doc_keys=(record.doc_key,),
        bodies=document.bodies,
        scope=document.scope,
        page=1,
        source_url=document.source_url,
    )


def timeline_parts(records: Iterable[MeetingRecord], documents: Mapping[str, MinutesDocument]) -> list[SummaryPart]:
    """A linha do tempo por órgão (a categoria da pasta), abrangência e ano, montada das fichas sem modelo: uma linha
    por item conferido, na ordem das datas, com a ata e a página de onde veio. Os itens de procedimento da reunião
    (mesa, comissões assessoras, ata anterior, pauta) ficam de fora, a decisão é encurtada, e a mesma linha registrada
    em duas atas da mesma reunião (as vias de cartório) aparece uma vez só."""
    groups: dict[tuple[str, str | None, int], list[tuple[date, str, str]]] = defaultdict(list)
    seen: set[tuple[tuple[str, str | None, int], str]] = set()

    def add(key: tuple[str, str | None, int], when: date, doc_key: str, line: str, *, dedupe: str) -> None:
        mark = (key, f"{when.isoformat()} {_normalize(dedupe)}")
        if mark not in seen:
            seen.add(mark)
            groups[key].append((when, doc_key, line))

    for record in records:
        document = documents.get(record.doc_key)
        when = record.data or (document.effective_date if document else None)
        if document is None or when is None or not document.bodies:
            continue
        reference = f"ata de {when.strftime('%d/%m/%Y')}"
        key = (document.bodies[0], document.scope, when.year)
        add(key, when, record.doc_key, _meeting_line(record, when), dedupe=_meeting_line(record, when))
        for item in record.deliberacoes:
            if item.conferido and not _procedural(item.tema):
                decision = _clip(item.decisao, TIMELINE_DECISION_CHARS)
                line = f"  {item.tema}: {decision} ({reference}, p. {item.pagina})"
                add(key, when, record.doc_key, line, dedupe=f"{item.tema} {decision}")
        for election in record.eleicoes:
            if election.conferido and not _procedural(f"{election.orgao} {election.cargo}"):
                text = f"eleição{_of(election.orgao)}: {election.eleito} ({election.cargo}){_term(election.mandato)}"
                add(key, when, record.doc_key, f"  {text} ({reference}, p. {election.pagina})", dedupe=text)
        for figure in record.numeros:
            if figure.conferido:
                text = f"{figure.valor} {figure.o_que}"
                add(key, when, record.doc_key, f"  {text} ({reference}, p. {figure.pagina})", dedupe=text)
    parts: list[SummaryPart] = []
    for (body, scope, year), entries in sorted(groups.items(), key=lambda item: (item[0][0], item[0][2])):
        entries.sort(key=lambda entry: (entry[0], entry[1]))
        sample = documents[entries[0][1]]
        label = f"{sample.category}{'' if scope is None else ' · ' + scope}"
        parts += _parts(
            base_id=f"linha-do-tempo-{body.replace('_', '-')}-{scope or 'geral'}#{year}",
            kind="linha_do_tempo",
            title=f"Linha do tempo · {label} · {year}",
            lines=[line for _, _, line in entries],
            year=year,
            meeting_date=None,
            doc_keys=tuple(dict.fromkeys(doc_key for _, doc_key, _ in entries)),
            bodies=sample.bodies,
            scope=scope,
            page=None,
            source_url=None,
        )
    return parts


def _procedural(text: str) -> bool:
    normalized = f" {_normalize(text)} "
    return any(f" {term} " in normalized for term in PROCEDURAL)


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return f"{cut}…"


def _parts(*, base_id: str, kind: str, title: str, lines: list[str], **fields: Any) -> list[SummaryPart]:
    chunks: list[list[str]] = [[]]
    size = 0
    for line in lines:
        if chunks[-1] and size + len(line) > PART_CHARS:
            chunks.append([])
            size = 0
        chunks[-1].append(line)
        size += len(line) + 1
    total = len(chunks)
    return [
        SummaryPart(
            summary_id=base_id if total == 1 else f"{base_id}~{index}",
            kind=kind,
            title=title if total == 1 else f"{title} (parte {index} de {total})",
            text="\n".join(body),
            **fields,
        )
        for index, body in enumerate(chunks, start=1)
    ]


def _meeting_line(record: MeetingRecord, when: date | None) -> str:
    kind = KIND_LABELS.get(record.tipo, "")
    day = "data não registrada" if when is None else when.strftime("%d/%m/%Y")
    where = f", {record.local}" if record.local else ""
    return f"Reunião{' ' + kind if kind else ''} de {day}{where}."


def _vote(text: str) -> str:
    return f" ({text})" if text else ""


def _of(orgao: str) -> str:
    return f" para {orgao}" if orgao else ""


def _term(mandato: str) -> str:
    return f", com mandato de {mandato}" if mandato else ""


def _items(raw: Mapping[str, Any], key: str) -> list[Mapping[str, Any]]:
    value = raw.get(key) or []
    if not isinstance(value, list) or not all(isinstance(item, Mapping) for item in value):
        raise RecordError(f"{key}: esperada uma lista de mapas")
    return value


def _text(raw: Mapping[str, Any], key: str, *, required: bool = False) -> str:
    value = raw.get(key)
    if value is None:
        value = ""
    if not isinstance(value, str | int | float):
        raise RecordError(f"{key}: esperado texto, veio {value!r}")
    text = " ".join(str(value).split())
    if required and not text:
        raise RecordError(f"{key}: obrigatório")
    return text


def _page(raw: Mapping[str, Any]) -> int:
    value = raw.get("pagina")
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise RecordError(f"pagina: esperado inteiro ≥ 1, veio {value!r}")
    return value


def _date(value: object) -> date | None:
    if value in (None, ""):
        return None
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        raise RecordError(f"reuniao.data: esperada data ISO, veio {value!r}") from None
