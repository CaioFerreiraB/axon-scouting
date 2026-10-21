"""Métricas de domínio do pack scouting (spec 105; ADR-027).

Núcleo funcional puro, declarado pelo `pack.py` no hook `eval_metrics`: sem relógio, sem I/O e sem modelo. Quem roda o
conjunto dourado e lê o trace é o harness da 009; aqui só se mede, item a item.
"""

from axon_scouting.eval.metrics import (
    CONTEXT_FIELDS,
    CONTEXT_KEY,
    ContextAccuracy,
    CurrentDocumentRate,
    ValidityNoticeRate,
    expected_context,
    is_historical,
)

__all__ = [
    "CONTEXT_FIELDS",
    "CONTEXT_KEY",
    "ContextAccuracy",
    "CurrentDocumentRate",
    "ValidityNoticeRate",
    "expected_context",
    "is_historical",
]
