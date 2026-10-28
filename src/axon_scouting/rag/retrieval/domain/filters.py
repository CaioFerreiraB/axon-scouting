"""Os `retrieval_filters` da 101 → `IndexFilters` da 102 e as facetas (spec 103 FR-002, FR-003, FR-009, FR-011;
research R3, R13; ADR-045).

Sem relógio: sem as datas não há como escolher a vigência, e "hoje" seria um chute. Chave desconhecida é erro — o
contrato da 101 diz que nenhuma chave nova entra sem atualizar a 103. Os temas são aceitos e ignorados na v1. As
facetas (`branches`, dois ramos ou mais) não são filtro do índice: o retriever faz uma busca por ramo. Os tipos
sugeridos (`suggested_types`) também não: só desempatam a ordem (ADR-046). O pedido de lista (`catalog`) é da tool de
lista de documentos (ADR-047), e o de enumeração (`enumeration`), da tool de unidades (ADR-048); a busca por trechos
ignora os dois.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from typing import Final

from axon_ai_contracts import JSONValue

from axon_scouting.rag.index.api import BRANCHES, SCOPES, IndexFilters, taxonomy_id
from axon_scouting.rag.retrieval.domain.catalog import CatalogRequest
from axon_scouting.rag.retrieval.domain.units import EnumerationRequest

KNOWN: Final = frozenset(
    {
        "branch",
        "branches",
        "topics",
        "document_types",
        "suggested_types",
        "bodies",
        "suggested_bodies",
        "scopes",
        "effective_from",
        "effective_until",
        "include_superseded",
        "previous_version",
        "catalog",
        "enumeration",
    }
)


class InvalidFilters(ValueError):
    """Filtros ausentes ou fora do contrato da 101; `key` nomeia a chave (ou `None`, para o todo)."""

    def __init__(self, key: str | None, message: str) -> None:
        self.key = key
        where = f" ({key})" if key else ""
        super().__init__(f"filtros de recuperação inválidos{where}: {message}")


def facets(raw: Mapping[str, JSONValue] | None) -> tuple[str, ...]:
    """As facetas pedidas (`branches`), na ordem recebida; nenhuma, `()`. Os demais filtros são de `index_filters`."""
    value = (raw or {}).get("branches")
    if value is None:
        return ()
    if (
        not isinstance(value, list)
        or len(value) < 2
        or len(set(value)) != len(value)
        or any(not isinstance(item, str) or item not in BRANCHES for item in value)
    ):
        raise InvalidFilters(
            "branches", f"esperados dois ramos ou mais, sem repetir, de {', '.join(BRANCHES)}: {value!r}"
        )
    if (raw or {}).get("branch") is not None:
        raise InvalidFilters("branches", "branch e branches juntos — a derivação publica um ou outro")
    return tuple(str(item) for item in value)


def index_filters(raw: Mapping[str, JSONValue] | None) -> IndexFilters:
    if raw is None:
        raise InvalidFilters(
            None, "ausentes — a preparação de contexto do pack publica a vigência em retrieval_filters"
        )
    if not isinstance(raw, Mapping):
        raise InvalidFilters(None, "esperado um mapa")
    unknown = sorted(set(raw) - KNOWN)
    if unknown:
        raise InvalidFilters(unknown[0], "chave desconhecida — atualize o contrato de filtros da 101 e a 103 juntos")
    start = _date(raw, "effective_from")
    end = _date(raw, "effective_until")
    if start > end:
        raise InvalidFilters("effective_from", f"{start} depois de effective_until ({end})")
    return IndexFilters(
        effective_from=start,
        effective_until=end,
        include_superseded=_flag(raw, "include_superseded"),
        previous_version=_flag(raw, "previous_version"),
        branch=_branch(raw),
        document_types=_ids(raw, "document_types"),
        bodies=_ids(raw, "bodies"),
        scopes=_scopes(raw),
    )


def _date(raw: Mapping[str, JSONValue], key: str) -> date:
    value = raw.get(key)
    if value is None:
        raise InvalidFilters(key, "obrigatória — sem ela não há como escolher a vigência (o retriever não lê relógio)")
    if not isinstance(value, str):
        raise InvalidFilters(key, f"esperada data ISO, veio {value!r}")
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise InvalidFilters(key, f"data ISO inválida: {value!r}") from None


def _flag(raw: Mapping[str, JSONValue], key: str) -> bool:
    value = raw.get(key, False)
    if not isinstance(value, bool):
        raise InvalidFilters(key, f"esperado booleano, veio {value!r}")
    return value


def _branch(raw: Mapping[str, JSONValue]) -> str | None:
    value = raw.get("branch")
    if value is None:
        return None
    if not isinstance(value, str) or value not in BRANCHES:
        raise InvalidFilters("branch", f"ramo desconhecido {value!r} (esperado um de {', '.join(BRANCHES)})")
    return value


def catalog_request(raw: Mapping[str, JSONValue] | None) -> CatalogRequest | None:
    """O pedido de lista de documentos (ADR-047); `None` quando a pergunta não pede uma."""
    value = (raw or {}).get("catalog")
    if value is None:
        return None
    kind = value.get("kind") if isinstance(value, Mapping) else None
    terms = value.get("terms", []) if isinstance(value, Mapping) else None
    if (
        not isinstance(kind, str)
        or not kind.strip()
        or not isinstance(terms, list)
        or any(not isinstance(term, str) for term in terms)
    ):
        raise InvalidFilters("catalog", f"esperado {{'kind': texto, 'terms': [textos]}}, veio {value!r}")
    return CatalogRequest(kind=kind, terms=tuple(str(term) for term in terms))


def enumeration_request(raw: Mapping[str, JSONValue] | None) -> EnumerationRequest | None:
    """O pedido de enumeração (ADR-048); `None` quando a pergunta não pede uma."""
    value = (raw or {}).get("enumeration")
    if value is None:
        return None
    kind = value.get("unit_kind") if isinstance(value, Mapping) else None
    terms = value.get("terms", []) if isinstance(value, Mapping) else None
    if (
        not isinstance(value, Mapping)
        or (kind is not None and (not isinstance(kind, str) or not kind.strip()))
        or not isinstance(terms, list)
        or any(not isinstance(term, str) for term in terms)
    ):
        raise InvalidFilters(
            "enumeration", f"esperado {{'unit_kind': texto ou null, 'terms': [textos]}}, veio {value!r}"
        )
    return EnumerationRequest(unit_kind=kind, terms=tuple(str(term) for term in terms))


def suggested_types(raw: Mapping[str, JSONValue] | None) -> tuple[str, ...]:
    """Os tipos sugeridos pelos temas da pergunta: preferência de ordem, nunca filtro (ADR-046)."""
    return _ids(raw or {}, "suggested_types")


def suggested_bodies(raw: Mapping[str, JSONValue] | None) -> tuple[str, ...]:
    """Os órgãos que a pergunta nomeia sem citar um tipo nem pedir lista: só desempatam (ADR-049)."""
    return _ids(raw or {}, "suggested_bodies")


def _ids(raw: Mapping[str, JSONValue], key: str) -> tuple[str, ...]:
    """Uma lista de ids da taxonomia (ADR-049); id que nenhum documento tem não acha nada, sem erro."""
    value = raw.get(key)
    if value is None:
        return ()
    if not isinstance(value, list) or any(not isinstance(i, str) or not i or taxonomy_id(i) != i for i in value):
        raise InvalidFilters(key, f"esperada lista de ids da taxonomia (minúsculas, dígitos e _), veio {value!r}")
    return tuple(str(item) for item in value)


def _scopes(raw: Mapping[str, JSONValue]) -> tuple[str, ...]:
    value = raw.get("scopes")
    if value is None:
        return ()
    if not isinstance(value, list) or any(item not in SCOPES for item in value):
        raise InvalidFilters("scopes", f"esperada lista de abrangências ({', '.join(SCOPES)}), veio {value!r}")
    return tuple(str(item) for item in value)
