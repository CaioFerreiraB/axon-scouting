"""Identificadores de chunk e chaves de documento e de conteúdo (spec 102 FR-005; research R9, R13; T022)."""

from __future__ import annotations

import re

import pytest

from ai_scouting.rag.ingestion.domain.ids import chunk_ids, content_key, doc_key, slug


def test_formato_do_design() -> None:
    assert chunk_ids("man-pio-2026", [("4.2.1", 87)]) == ["man-pio-2026#4.2.1@p87"]


def test_repeticao_ganha_sufixo_na_ordem() -> None:
    assert chunk_ids("doc-1", [("s1", 3), ("s1", 3), ("art1", 9), ("s1", 3), ("s1", 4), ("art1", 9)]) == [
        "doc-1#s1@p3",
        "doc-1#s1@p3~2",
        "doc-1#art1@p9",
        "doc-1#s1@p3~3",
        "doc-1#s1@p4",
        "doc-1#art1@p9~2",
    ]


def test_caracteres_permitidos() -> None:
    for chunk_id in chunk_ids("estatuto-2019", [("art42-a", 15), ("anexo-i", 30), ("preambulo", 1), ("s2", 2)]):
        assert re.fullmatch(r"[a-z0-9._~#@-]+", chunk_id)


@pytest.mark.parametrize(
    ("texto", "esperado"),
    [("Art. 42-A", "art-42-a"), ("ANEXO I", "anexo-i"), ("Preâmbulo", "preambulo"), ("4.2.1", "4.2.1"), ("  x  ", "x")],
)
def test_slug(texto: str, esperado: str) -> None:
    assert slug(texto) == esperado


def test_doc_key() -> None:
    assert doc_key("manual-pioneiro", "2026") == "manual-pioneiro-2026"


def test_content_key_muda_com_cada_parte() -> None:
    base = content_key("a" * 64, "manual", "Manual X", "pdfium-5.14.0")
    assert len(base) == 64
    assert base == content_key("a" * 64, "manual", "Manual X", "pdfium-5.14.0")
    assert base != content_key("b" * 64, "manual", "Manual X", "pdfium-5.14.0")
    assert base != content_key("a" * 64, "guia", "Manual X", "pdfium-5.14.0")
    assert base != content_key("a" * 64, "manual", "Manual Y", "pdfium-5.14.0")
    assert base != content_key("a" * 64, "manual", "Manual X", "pdfium-5.15.0")
