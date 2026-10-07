"""`SystemClock`: o relógio do sistema, em UTC (porta `Clock`)."""

from __future__ import annotations

from datetime import UTC, datetime


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)
