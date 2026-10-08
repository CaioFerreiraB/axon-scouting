"""FR-012: a busca só codifica a pergunta com a codificação registrada no build publicado (spec 102 research R14)."""

from __future__ import annotations

from ai_scouting.rag.index.api import EmbeddingModel, IndexMismatch, IndexVersion


def check_encoders(version: IndexVersion, embedding: EmbeddingModel, sparse: str) -> None:
    """Levanta `IndexMismatch`, nomeando os dois lados, se o embedder ou o codificador esparso divergem."""
    if version.embedding != embedding or version.sparse != sparse:
        raise IndexMismatch(
            index=f"{version.embedding.describe()} + {version.sparse}",
            encoder=f"{embedding.describe()} + {sparse}",
        )
