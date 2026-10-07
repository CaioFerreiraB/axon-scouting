"""Suíte da porta `SparseEncoder` contra o `Bm25Encoder` (spec 102 research R11; T015)."""

from __future__ import annotations

import pytest

from ai_scouting.rag.index.adapters.bm25 import Bm25Encoder
from ai_scouting.rag.index.ports import SparseEncoder


@pytest.fixture
def encoder() -> SparseEncoder:
    return Bm25Encoder()


def test_identidade(encoder: SparseEncoder) -> None:
    assert encoder.identity() == "bm25-pt-v1;k1=1.2;b=0.75;avgdl=256"


def test_ids_estaveis_entre_instancias_e_processos(encoder: SparseEncoder) -> None:
    assert encoder.encode_query("42").indices == (841265288,)  # zlib.crc32(b"42"): igual em qualquer processo
    assert Bm25Encoder().encode_passage("O uniforme escoteiro") == encoder.encode_passage("O uniforme escoteiro")


def test_numeros_preservados_sem_zeros_a_esquerda(encoder: SparseEncoder) -> None:
    artigo = set(encoder.encode_passage("Art. 42. O responsável pela atividade").indices)
    assert set(encoder.encode_query("42").indices) <= artigo
    assert encoder.encode_query("Regra 033").indices == encoder.encode_query("regra 33").indices


def test_acentos_dobrados(encoder: SparseEncoder) -> None:
    assert encoder.encode_query("organização").indices == encoder.encode_query("organizacao").indices
    assert encoder.encode_query("Uniformes").indices == encoder.encode_query("uniforme").indices


def test_stopwords_somem(encoder: SparseEncoder) -> None:
    assert encoder.encode_query("de que o a").indices == ()
    assert encoder.encode_passage("").indices == ()
    assert len(encoder.encode_query("o uniforme de campo").indices) == 2


def test_pergunta_tem_peso_um_por_termo_distinto(encoder: SparseEncoder) -> None:
    vetor = encoder.encode_query("uniforme uniforme campo")
    assert vetor.values == (1.0, 1.0)


def test_passagem_cresce_com_a_frequencia_e_satura(encoder: SparseEncoder) -> None:
    def peso(texto: str) -> float:
        termo = encoder.encode_query("uniforme").indices[0]
        vetor = encoder.encode_passage(texto)
        return vetor.values[vetor.indices.index(termo)]

    um, dois, dez = peso("uniforme"), peso("uniforme uniforme"), peso(" ".join(["uniforme"] * 10))
    assert um < dois < dez < 2.2  # k1 + 1


def test_indices_ordenados_e_unicos(encoder: SparseEncoder) -> None:
    vetor = encoder.encode_passage("O escotismo é um movimento educacional de jovens, e o movimento cresce.")
    assert list(vetor.indices) == sorted(set(vetor.indices))
