"""Suíte da porta `Clock` contra `SystemClock` e `FixedClock` (spec 102 research R16; T013)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from axon_scouting.rag.ingestion.adapters.clock import SystemClock
from axon_scouting.rag.ingestion.adapters.memory import FixedClock
from axon_scouting.rag.ingestion.ports import Clock

INSTANTE = datetime(2026, 10, 7, 15, 30, tzinfo=UTC)


@pytest.fixture(params=["system", "fixed"])
def clock(request: pytest.FixtureRequest) -> Clock:
    return SystemClock() if request.param == "system" else FixedClock(INSTANTE)


def test_utc_com_fuso(clock: Clock) -> None:
    agora = clock.now()
    assert agora.tzinfo is not None
    assert agora.utcoffset() == timedelta(0)


def test_nunca_anda_para_tras(clock: Clock) -> None:
    primeiro = clock.now()
    assert clock.now() >= primeiro


def test_fixo_e_fixo() -> None:
    assert FixedClock(INSTANTE).now() == INSTANTE
    with pytest.raises(ValueError):
        FixedClock(datetime(2026, 10, 7))


def test_sistema_e_o_agora() -> None:
    assert abs(SystemClock().now() - datetime.now(UTC)) < timedelta(seconds=1)
