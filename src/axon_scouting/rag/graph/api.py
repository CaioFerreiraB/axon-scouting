"""Seam *Grafo de domínio* (spec 106, ADR-064; contracts/domain-graph-seam.md).

O que a 103 (o retriever), a 101 (pela composição do `pack.py`) e o `scripts/grafo.py` veem do grafo de domínio: os
tipos da evidência, o esquema do plano (`QueryPlan`), a tabela de papéis (`ROLES`), o reconhecimento das entidades numa
pergunta (`reconhecer`) e a porta `DomainGraphSource`. Puro: não importa nada do resto do módulo nem de outro módulo.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Final, Literal, Protocol, cast, get_args, runtime_checkable

from axon_ai_contracts import JSONValue

Role = Literal["norma", "orientacao", "registro", "comunicacao"]
Mode = Literal["disciplina", "orienta", "registra", "comunica"]
Origin = Literal["curado", "regra", "modelo_conferido"]
QueryName = Literal["sobre", "aplicavel_a", "documentos", "itens", "linha_do_tempo", "ligados"]
Form = Literal["simples", "para_cada", "intersecao"]
EventType = Literal["reuniao", "deliberacao", "eleicao", "alteracao"]

ROLE_ORDER: Final[tuple[Role, ...]] = ("norma", "orientacao", "registro", "comunicacao")
ORIGINS: Final[tuple[Origin, ...]] = get_args(Origin)
QUERIES: Final[tuple[QueryName, ...]] = get_args(QueryName)
FORMS: Final[tuple[Form, ...]] = get_args(Form)

ROLES: Final[Mapping[str, Role]] = {
    "estatuto": "norma",
    "por": "norma",
    "regulamento": "norma",
    "resolucao": "norma",
    "politica": "norma",
    "lei": "norma",
    "manual": "orientacao",
    "planejamento_estrategico": "orientacao",
    "ata": "registro",
    "parecer": "registro",
    "relatorio": "registro",
    "manifestacao": "registro",
    "oficio": "comunicacao",
    "edital": "comunicacao",
    "boletim": "comunicacao",
    "posicionamento": "comunicacao",
    "recomendacao": "comunicacao",
    "apresentacao": "comunicacao",
}
"""O papel de cada tipo de documento da taxonomia, por `taxonomy_id` (research R3). Tipo sem papel falha o build."""

MODE_OF: Final[Mapping[Role, Mode]] = {
    "norma": "disciplina",
    "orientacao": "orienta",
    "registro": "registra",
    "comunicacao": "comunica",
}
"""O modo da aresta `SOBRE` pelo papel do documento da unidade (ADR-064, item 4)."""

PREFIXES: Final = ("tipo_inst", "tipo_orgao", "inst", "orgao", "cargo", "ramo", "secao", "area", "conceito", "doc")
EACH: Final = "$cada"
"""Em `para_cada`, o lugar da entidade nos argumentos da consulta: o executor o troca por cada id."""
ENTITY_ID: Final = re.compile(r"^(\$cada|(" + "|".join(PREFIXES) + r"):[a-z0-9_.-]+)$")
MAX_STEPS: Final = 4
MAX_EACH: Final = 6

Aliases = Mapping[str, tuple[str, ...]]
"""Id da entidade → os termos (já normalizados) que a reconhecem numa pergunta."""


def reconhecer(texto: str, apelidos: Aliases) -> tuple[str, ...]:
    """Os ids das entidades que o texto (já normalizado pela 101) nomeia: pelo termo mais longo — o termo contido num
    termo mais longo já casado não conta ("conselho consultivo" dentro de "conselho consultivo regional") —, sem
    repetição, na ordem da primeira menção. Puro; a regra do ADR-063 (101 R43)."""
    padded = f" {texto} "
    matches = sorted(
        (
            (term, entity, padded.find(f" {term} "))
            for entity, terms in apelidos.items()
            for term in terms
            if term and f" {term} " in padded
        ),
        key=lambda match: -len(match[0]),
    )
    kept: list[tuple[str, str, int]] = []
    for term, entity, position in matches:
        if not any(term in longer for longer, _, _ in kept):
            kept.append((term, entity, position))
    return tuple(dict.fromkeys(entity for _, entity, _ in sorted(kept, key=lambda match: match[2])))


class PlanInvalid(ValueError):
    """Plano fora do esquema (contracts/query-plan.schema.json); `field` nomeia o campo."""

    def __init__(self, field: str, message: str) -> None:
        self.field = field
        super().__init__(f"plano inválido ({field}): {message}")


_ARGS: Final[Mapping[QueryName, frozenset[str]]] = {
    "sobre": frozenset({"entidades", "modo", "papeis", "tipos", "ramo", "periodo"}),
    "aplicavel_a": frozenset({"instituicao", "papeis"}),
    "documentos": frozenset({"tipo", "emissor", "aplica_se_a", "area", "ramo", "periodo", "vigente"}),
    "itens": frozenset({"tipo_de_unidade", "de", "ramo"}),
    "linha_do_tempo": frozenset({"alvo", "eventos", "periodo"}),
    "ligados": frozenset({"alvo", "relacoes"}),
}
_REQUIRED: Final[Mapping[QueryName, frozenset[str]]] = {
    "sobre": frozenset({"entidades"}),
    "aplicavel_a": frozenset({"instituicao"}),
    "documentos": frozenset(),
    "itens": frozenset(),
    "linha_do_tempo": frozenset({"alvo"}),
    "ligados": frozenset({"alvo"}),
}
_ENTITY_ARGS: Final = frozenset({"ramo", "instituicao", "emissor", "aplica_se_a", "area", "de"})
_SLUG: Final = re.compile(r"^[a-z0-9_]+$")


@dataclass(frozen=True, slots=True, kw_only=True)
class QueryStep:
    """Uma consulta do plano, com só os parâmetros da assinatura dela."""

    q: QueryName
    args: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _check_step(self.q, self.args, "consultas[?]")


@dataclass(frozen=True, slots=True, kw_only=True)
class QueryPlan:
    """O plano de uma pergunta (ADR-064, item 6): dados num esquema fechado, que a 101 escreve e o executor roda.
    Plano vazio (`consultas == ()`) = só a busca."""

    version: Literal[1] = 1
    forma: Form = "simples"
    consultas: tuple[QueryStep, ...] = ()
    para_cada: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.version != 1:
            raise PlanInvalid("version", f"versão desconhecida: {self.version!r}")
        if self.forma not in FORMS:
            raise PlanInvalid("forma", f"forma desconhecida: {self.forma!r}")
        if len(self.consultas) > MAX_STEPS:
            raise PlanInvalid("consultas", f"no máximo {MAX_STEPS} consultas")
        if self.para_cada and self.forma != "para_cada":
            raise PlanInvalid("para_cada", "só com a forma para_cada")
        if len(self.para_cada) > MAX_EACH:
            raise PlanInvalid("para_cada", f"no máximo {MAX_EACH} entidades")
        for entity in self.para_cada:
            if not ENTITY_ID.match(entity) or entity == EACH:
                raise PlanInvalid("para_cada", f"id de entidade inválido: {entity!r}")

    @property
    def empty(self) -> bool:
        return not self.consultas

    def to_json(self) -> dict[str, JSONValue]:
        raw: dict[str, JSONValue] = {
            "version": self.version,
            "forma": self.forma,
            "consultas": [{"q": step.q, "args": dict(step.args)} for step in self.consultas],
        }
        if self.para_cada:
            raw["para_cada"] = list(self.para_cada)
        return raw

    @classmethod
    def from_json(cls, raw: Mapping[str, JSONValue]) -> QueryPlan:
        if not isinstance(raw, Mapping):
            raise PlanInvalid("plan", "precisa ser um objeto")
        unknown = set(raw) - {"version", "forma", "consultas", "para_cada"}
        if unknown:
            raise PlanInvalid(sorted(unknown)[0], "campo desconhecido")
        steps = raw.get("consultas")
        if not isinstance(steps, list):
            raise PlanInvalid("consultas", "precisa ser uma lista")
        parsed: list[QueryStep] = []
        for index, step in enumerate(steps):
            where = f"consultas[{index}]"
            if not isinstance(step, Mapping) or set(step) - {"q", "args"} or "q" not in step:
                raise PlanInvalid(where, "precisa ter q e args, e só eles")
            name, args = step.get("q"), step.get("args", {})
            if name not in QUERIES:
                raise PlanInvalid(f"{where}.q", f"consulta desconhecida: {name!r}")
            if not isinstance(args, Mapping):
                raise PlanInvalid(f"{where}.args", "precisa ser um objeto")
            _check_step(cast(QueryName, name), args, where)
            parsed.append(QueryStep(q=cast(QueryName, name), args=dict(args)))
        each = raw.get("para_cada", [])
        if not isinstance(each, list) or not all(isinstance(e, str) for e in each):
            raise PlanInvalid("para_cada", "precisa ser uma lista de ids")
        version = raw.get("version")
        forma = raw.get("forma", "simples")
        if forma not in FORMS:
            raise PlanInvalid("forma", f"forma desconhecida: {forma!r}")
        if version != 1:
            raise PlanInvalid("version", f"versão desconhecida: {version!r}")
        return cls(
            version=1,
            forma=forma,
            consultas=tuple(parsed),
            para_cada=tuple(cast(list[str], each)),
        )


def _check_step(name: QueryName, args: Mapping[str, JSONValue], where: str) -> None:
    if name not in QUERIES:
        raise PlanInvalid(f"{where}.q", f"consulta desconhecida: {name!r}")
    unknown = set(args) - _ARGS[name]
    if unknown:
        raise PlanInvalid(f"{where}.args.{sorted(unknown)[0]}", "parâmetro desconhecido")
    missing = _REQUIRED[name] - set(args)
    if missing:
        raise PlanInvalid(f"{where}.args.{sorted(missing)[0]}", "parâmetro obrigatório")
    for key, value in args.items():
        at = f"{where}.args.{key}"
        if key == "entidades":
            if not isinstance(value, list) or not 1 <= len(value) <= 4:
                raise PlanInvalid(at, "de 1 a 4 ids de entidade")
            for entity in value:
                if not isinstance(entity, str) or not ENTITY_ID.match(entity):
                    raise PlanInvalid(at, f"id de entidade inválido: {entity!r}")
        elif key in _ENTITY_ARGS or (key == "alvo" and name == "linha_do_tempo"):
            if not isinstance(value, str) or not ENTITY_ID.match(value):
                raise PlanInvalid(at, f"id de entidade inválido: {value!r}")
        elif key == "papeis":
            if not isinstance(value, list) or any(role not in ROLE_ORDER for role in value):
                raise PlanInvalid(at, f"papéis: {', '.join(ROLE_ORDER)}")
        elif key == "modo":
            if value not in ("todas", "alguma"):
                raise PlanInvalid(at, "todas ou alguma")
        elif key in ("tipos", "relacoes", "eventos"):
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                raise PlanInvalid(at, "lista de nomes")
        elif key in ("tipo", "tipo_de_unidade"):
            if not isinstance(value, str) or not _SLUG.match(value):
                raise PlanInvalid(at, "um id de tipo")
        elif key == "periodo":
            if not isinstance(value, Mapping) or set(value) - {"de", "ate"}:
                raise PlanInvalid(at, "objeto com de e ate")
        elif key == "vigente":
            if not isinstance(value, bool):
                raise PlanInvalid(at, "booleano")
        elif key == "alvo" and not isinstance(value, str):
            raise PlanInvalid(at, "um id")


@dataclass(frozen=True, slots=True, kw_only=True)
class EdgeRef:
    """A aresta que trouxe a evidência: o tipo, a fonte que a justifica e a origem (ADR-064, item 7)."""

    type: str
    source: str
    origin: Origin


@dataclass(frozen=True, slots=True, kw_only=True)
class EventRef:
    type: EventType
    text: str
    page: int | None


@dataclass(frozen=True, slots=True, kw_only=True)
class Evidence:
    """O que uma consulta devolve: a unidade (ou o documento, ou o evento), com o papel, o nível, a data e a aresta que
    a trouxe. Não traz texto: a 103 pega os trechos e escolhe os que mais casam com a pergunta."""

    query: QueryName
    target: str
    doc_key: str
    unit_id: str | None
    chunk_ids: tuple[str, ...]
    role: Role
    scope: str | None
    date: date | None
    edge: EdgeRef
    event: EventRef | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class FrontGroup:
    """A evidência de um passo do plano (e, em `para_cada`, de uma entidade), ordenada e cortada em documentos."""

    step: int
    query: QueryName
    entity: str | None
    evidence: tuple[Evidence, ...]


Frentes = tuple[FrontGroup, ...]


class GraphUnavailable(Exception):
    """O `graph.jsonl` é de outro build: o chunking difere do índice. A mensagem diz como reconstruir."""


@runtime_checkable
class DomainGraphSource(Protocol):
    """A porta do grafo de domínio (seam da 106), consumida pela 103 (`run`), pela composição da 101 (`aliases`) e pela
    linha de comando (`query`)."""

    async def aliases(self) -> Aliases:
        """Os termos de todas as entidades reconhecíveis; sem grafo, `{}`."""
        ...

    async def run(self, plan: QueryPlan) -> Frentes:
        """Executa o plano; plano vazio → `()`; entidade inexistente → o passo volta vazio."""
        ...

    async def query(self, name: QueryName, /, **args: JSONValue) -> list[Evidence]:
        """Uma consulta solta, para a linha de comando; argumento desconhecido → `PlanInvalid`."""
        ...


def scope_rank(scopes: Sequence[str], scope: str | None) -> int:
    """A posição do nível na ordem de cima para baixo; desconhecido vai por último."""
    return scopes.index(scope) if scope in scopes else len(scopes)
