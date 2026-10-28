"""Invariantes de qualidade do artefato commitado (spec 102 US6, FR-017 a FR-020; research R26–R29).

O que a revisão do RAG de 2026-10-27 achou no índice — trechos sem corpo, regras do POR só com o título, a marca de
hifenização do PDFium, trechos que o e5 truncava — não volta sem que um destes testes falhe. Os de texto rodam só com o
artefato; o de tokens, onde houver o tokenizador do e5 (`.cache/e5`, como no workflow `eval`).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from axon_scouting.rag.ingestion.domain.chunking import MIN_BODY_LETTERS

ROOT = Path(__file__).resolve().parents[3]
CHUNKS = ROOT / "src" / "axon_scouting" / "data" / "index" / "chunks.jsonl"
TOKENIZER = ROOT / ".cache" / "e5" / "tokenizer.json"
MAX_TOKENS = 512
MAX_TRUNCATED = 0.0
"""Nenhum trecho passa do que o e5 lê (ADR-050, `estrutural-v5`): o teto divide unidade normativa e janela. Antes, 2,1%
na `estrutural-v3` e 2,57% (378 de 14.706) na `estrutural-v4` — unidades normativas inteiras e tabelas de relatório,
que o tokenizador parte em mais tokens por caractere (research R29)."""


def _body(chunk: dict[str, Any]) -> str:
    return str(chunk["text"]).split("\n", 1)[1] if "\n" in chunk["text"] else str(chunk["text"])


@pytest.fixture(scope="module")
def chunks() -> list[dict[str, Any]]:
    return [json.loads(line) for line in CHUNKS.read_text(encoding="utf-8").split("\n") if line]


MAX_STUBS = 5
"""Os que sobram não são títulos de seção: ruído de OCR (uma ata), texto numa fonte sem mapa de caracteres (um
relatório) e um título solto — curadoria (R29); na `estrutural-v5`, 4."""


def test_nenhum_trecho_sem_corpo(chunks: list[dict[str, Any]]) -> None:
    """Corpo sem frase e com poucas letras só como único trecho do documento, salvo o ruído conhecido (R28)."""
    per_document = Counter(chunk["doc_key"] for chunk in chunks)
    stubs = [
        chunk["chunk_id"]
        for chunk in chunks
        if per_document[chunk["doc_key"]] > 1
        and sum(ch.isalpha() for ch in _body(chunk)) < MIN_BODY_LETTERS
        and sum(ch.isdigit() for ch in _body(chunk)) < MIN_BODY_LETTERS  # a tabela de números é conteúdo
        and not any(line.rstrip().endswith((".", "!", "?", ":", ";")) for line in _body(chunk).splitlines())
    ]
    assert len(stubs) <= MAX_STUBS, stubs


def test_sem_marca_de_hifenizacao(chunks: list[dict[str, Any]]) -> None:
    assert [chunk["chunk_id"] for chunk in chunks if "￾" in chunk["text"] or "­" in chunk["text"]] == []


def test_toda_regra_do_por_tem_corpo(chunks: list[dict[str, Any]]) -> None:
    """O título da regra ("REGRA 053 …") e, depois dele, o texto dela (R26)."""
    rules = [chunk for chunk in chunks if chunk["document_type"] == "por" and chunk["article"]]
    assert rules
    title_only = [chunk["chunk_id"] for chunk in rules if len(_body(chunk).split("\n", 1)[-1].strip()) < 30]
    assert title_only == []


def test_nenhum_trecho_alem_do_que_o_e5_le(chunks: list[dict[str, Any]]) -> None:
    if not TOKENIZER.is_file():
        pytest.skip("sem o tokenizador do e5 em .cache/e5 (axon-scouting-corpus model fetch .cache/e5)")
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    lengths = [len(e.ids) for e in tokenizer.encode_batch([f"passage: {chunk['text']}" for chunk in chunks])]
    assert sum(length > MAX_TOKENS for length in lengths) / len(lengths) <= MAX_TRUNCATED
