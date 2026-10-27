"""Os `retrieval_filters` da 101 → `IndexFilters` da 102 e as facetas (spec 103 FR-002, FR-003, FR-009, FR-011;
research R3, R13; ADR-045).

Sem relógio: sem as datas não há como escolher a vigência, e "hoje" seria um chute. Chave desconhecida é erro — o
contrato da 101 diz que nenhuma chave nova entra sem atualizar a 103. Os temas são aceitos e ignorados na v1. As
facetas (`branches`, dois ramos ou mais) não são filtro do índice: o retriever faz uma busca por ramo. Os tipos
sugeridos (`suggested_types`) também não: só desempatam a ordem (ADR-046).
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from typing import Final

from axon_ai_contracts import JSONValue

from axon_scouting.rag.index.api import BRANCHES, DocumentType, IndexFilters

KNOWN: Final = frozenset(
    {
        "branch",
        "branches",
        "topics",
        "document_types",
        "suggested_types",
        "effective_from",
        "effective_until",
        "include_superseded",
        "previous_version",
    }
)
_TYPES: Final = frozenset(t.value for t in DocumentType)


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
        document_types=_types(raw),
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


def suggested_types(raw: Mapping[str, JSONValue] | None) -> tuple[DocumentType, ...]:
    """Os tipos sugeridos pelos temas da pergunta: preferência de ordem, nunca filtro (ADR-046)."""
    return _types(raw or {}, "suggested_types")


def _types(raw: Mapping[str, JSONValue], key: str = "document_types") -> tuple[DocumentType, ...]:
    value = raw.get(key)
    if value is None:
        return ()
    if not isinstance(value, list) or any(not isinstance(item, str) or item not in _TYPES for item in value):
        raise InvalidFilters(key, f"esperada lista de tipos conhecidos, veio {value!r}")
    return tuple(DocumentType(item) for item in value)
