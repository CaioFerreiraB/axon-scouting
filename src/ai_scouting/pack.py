"""`ScoutingPack`: o pack de domínio escoteiro (spec 101).

Composition root do pack (ADR-012): aqui só se compõe — as políticas vivem em `context/` e `prompts/`. A 101 não faz
I/O, então não declara porta secundária (princípio XIII). Na Fase 4 o manifest declara só nome e versão: flow
preferido, coleção exigida e esquema de configuração chegam com a US5, na Fase 5 (clarificação de 2026-10-07).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from ai_core_contracts import JSONValue, PackBase, PackContext, PackManifest, TurnView

from ai_scouting.context.derivation import derive_context, to_slice
from ai_scouting.context.periods import reference_day
from ai_scouting.prompts import SYSTEM_PROMPT

VERSION: Final = "0.1.0"
"""Versão do manifest; igual à da distribuição `ai-scouting` (verificado em tests/test_pack.py)."""


class ScoutingPack(PackBase):
    """Instruções de domínio e a derivação determinística de contexto da pergunta."""

    name = "scouting"
    contract_version = "0.1"

    def manifest(self) -> PackManifest:
        return PackManifest(name=self.name, version=VERSION)

    def system_prompt(self, ctx: PackContext) -> str | None:
        return SYSTEM_PROMPT

    async def prepare_context(self, view: TurnView, ctx: PackContext) -> Mapping[str, JSONValue]:
        """Contexto e filtros de recuperação derivados da pergunta, sem modelo e sem relógio (FR-004, FR-005)."""
        return to_slice(derive_context(view.question, reference_day(view.reference_time)))
