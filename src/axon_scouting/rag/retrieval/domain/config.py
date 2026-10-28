"""Quantidades do retrieval (spec 103 FR-008; research R6). Provisórias — a 105 calibra."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True, kw_only=True)
class RetrievalConfig:
    """`default_limit`: trechos quando o chamador não diz (a tool no loop agêntico); `max_limit`: teto do pedido.

    O número de candidatos e o N após o rerank chegam do chamador (`top_k` — a 005 no `rag_first`). `max_cards`: teto
    de fichas da lista de documentos, para a lista não tomar todas as vagas dos trechos (ADR-047). `unit_tokens`: teto
    de tokens estimados dos irmãos de uma unidade entregues juntos (ADR-048; provisório até a medição da 105).
    """

    default_limit: int = 5
    max_limit: int = 30
    max_cards: int = 15
    unit_tokens: int = 1500

    def __post_init__(self) -> None:
        if not 1 <= self.default_limit <= self.max_limit:
            raise ValueError(
                f"limites inválidos: 1 ≤ default_limit ({self.default_limit}) ≤ max_limit ({self.max_limit})"
            )
        if self.max_cards < 1:
            raise ValueError(f"max_cards precisa ser ≥ 1 (veio {self.max_cards})")

    def limit(self, top_k: int | None) -> int:
        if top_k is None:
            return self.default_limit
        if top_k < 1:
            raise ValueError(f"top_k precisa ser ≥ 1 (veio {top_k})")
        return min(top_k, self.max_limit)
