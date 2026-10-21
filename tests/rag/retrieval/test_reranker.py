"""`MetadataReranker`: ordem da fusão, desempate por metadados, corte em N (spec 103 US4, FR-005; R4, R5; T016)."""

from __future__ import annotations

import asyncio
import inspect
import random

import pytest
from axon_ai_contracts import Chunk, Reranker
from axon_ai_contracts.testing import chunk as chunk_cru

from axon_scouting.rag.retrieval.domain.ordering import BRANCH_MATCH, FUSION_SCORE, TYPE_MATCH, sort_key
from axon_scouting.rag.retrieval.reranker import MetadataReranker


def anotado(chunk_id: str, fusao: float, *, status: str = "vigente", ramo: bool = False, score: float = 0.5) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text="t",
        score=score,
        document="Doc",
        status=status,
        metadata={FUSION_SCORE: fusao, BRANCH_MATCH: ramo, TYPE_MATCH: False},
    )


def rerank(documentos: list[Chunk], top_k: int = 5) -> list[Chunk]:
    return list(asyncio.run(MetadataReranker().rerank("pergunta", documentos, top_k=top_k)))


def trinta() -> list[Chunk]:
    rng = random.Random(7)
    return [
        anotado(
            f"c{n:02d}",
            rng.choice([1.0, 0.833, 0.5, 0.333, 0.25]),
            status=rng.choice(["vigente", "substituido"]),
            ramo=rng.random() < 0.3,
            score=rng.random(),
        )
        for n in range(30)
    ]


def test_us4_cen1_no_maximo_n_na_ordem_da_chave() -> None:
    candidatos = trinta()
    saida = rerank(candidatos, top_k=6)
    assert len(saida) == 6
    assert saida == sorted(candidatos, key=sort_key)[:6]


def test_us4_empatados_na_fusao_o_do_ramo_primeiro() -> None:
    geral, do_ramo = anotado("geral", 0.5), anotado("ramo", 0.5, ramo=True)
    assert [c.chunk_id for c in rerank([geral, do_ramo])] == ["ramo", "geral"]


def test_independe_da_ordem_de_entrada() -> None:
    candidatos = trinta()
    embaralhados = candidatos[:]
    random.Random(1).shuffle(embaralhados)
    por_score = sorted(candidatos, key=lambda c: -c.score)  # como a 005 une resultados de packs
    assert rerank(embaralhados, 10) == rerank(por_score, 10) == rerank(candidatos, 10)


def test_chunks_crus_como_o_kit_da_001() -> None:
    documentos = [chunk_cru("a"), chunk_cru("b"), chunk_cru("c")]
    saida = rerank(documentos, top_k=2)
    assert len(saida) <= 2 and {c.chunk_id for c in saida} <= {"a", "b", "c"}


def test_mistura_crus_depois() -> None:
    saida = rerank([chunk_cru("z"), anotado("b", 0.3), chunk_cru("y"), anotado("a", 0.6)], top_k=10)
    assert [c.chunk_id for c in saida] == ["a", "b", "z", "y"]


def test_limites_e_contrato() -> None:
    with pytest.raises(ValueError):
        rerank([anotado("a", 1.0)], top_k=0)
    assert len(rerank([anotado("a", 1.0), anotado("b", 0.5)], top_k=10)) == 2
    reranker = MetadataReranker()
    assert isinstance(reranker, Reranker)
    assert inspect.iscoroutinefunction(reranker.rerank)
