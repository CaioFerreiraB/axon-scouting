"""Quantidades do retrieval (spec 103 FR-008; research R6; T005)."""

from __future__ import annotations

import pytest

from axon_scouting.rag.retrieval.domain.config import RetrievalConfig


def test_defaults() -> None:
    config = RetrievalConfig()
    assert (config.default_limit, config.max_limit) == (5, 30)


@pytest.mark.parametrize(("default", "maximo"), [(0, 30), (31, 30), (5, 0)])
def test_faixa_invalida(default: int, maximo: int) -> None:
    with pytest.raises(ValueError):
        RetrievalConfig(default_limit=default, max_limit=maximo)


def test_limite_pedido() -> None:
    config = RetrievalConfig()
    assert config.limit(None) == 5
    assert config.limit(12) == 12
    assert config.limit(50) == 30
    with pytest.raises(ValueError):
        config.limit(0)
