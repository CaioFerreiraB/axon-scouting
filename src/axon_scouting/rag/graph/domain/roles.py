"""A tabela de papéis aplicada à taxonomia (spec 106 FR-005; research R3). A tabela fica no `api.py` (`ROLES`); aqui,
a conferência que o build e os invariantes usam."""

from __future__ import annotations

from collections.abc import Iterable

from axon_scouting.rag.graph.api import ROLES


def missing_roles(types: Iterable[str]) -> list[str]:
    """Os tipos de documento (`taxonomy_id`) sem papel; o build falha com eles."""
    return sorted({kind for kind in types if kind not in ROLES})
