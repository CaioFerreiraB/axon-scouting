"""`MetadataReranker`: o reranker v1 do scouting (spec 103 US4, FR-005; research R4, R5).

Mantém a ordem da fusão e desfaz empates pelos metadados que o retriever gravou — vigente, específico do ramo, tipo
sugerido —, sem pesos, sem modelo e sem I/O. Lê a ordem dos metadados, não da entrada (a 005 une resultados por
score); chunk sem as marcas vai depois, na ordem em que veio. Custo: ordenar algumas dezenas de itens (SC-003).
"""

from __future__ import annotations

from collections.abc import Sequence

from ai_core_contracts import Chunk

from ai_scouting.rag.retrieval.domain.ordering import ordered


class MetadataReranker:
    """Implementa `Reranker` (001). O v2 (hospedado) e o v3 (cross-encoder local) entram como adapters."""

    async def rerank(self, query: str, documents: Sequence[Chunk], *, top_k: int = 5) -> list[Chunk]:
        if top_k < 1:
            raise ValueError(f"top_k precisa ser ≥ 1 (veio {top_k})")
        return ordered(documents)[:top_k]
