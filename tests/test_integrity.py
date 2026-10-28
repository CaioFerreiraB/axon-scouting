"""`citation_integrity` (spec 104 US1, US2, US4; contracts/validators.md casos 1, 2, 3, 6 e 7; T006, T008, T012)."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from axon_ai_contracts import ValidationResult
from axon_ai_contracts.testing import chunk, turn_view, validator_context

from axon_scouting.validators import CitationIntegrityValidator
from axon_scouting.validators.integrity import UNGROUNDED_NOTICE

A = chunk(
    "man-pio-2026#4.2.1@p87", document="Manual Pioneiro", version="2026", status="vigente", section="4.2.1", page=87
)
B = chunk("por-2023#12", document="POR", version="2023", status="vigente", section="12")
C = chunk("guia-esc#3", document="Guia do Escotista")


def validar(answer: Any, *, contexto: tuple[Any, ...] = (A, B), reranked: tuple[Any, ...] = ()) -> ValidationResult:
    view = turn_view(answer=answer, generation_context=contexto, reranked=reranked)
    return asyncio.run(CitationIntegrityValidator().validate(view, validator_context("scouting")))


def test_nome() -> None:
    assert CitationIntegrityValidator.name == "citation_integrity"


# --- US1: citação inventada nunca chega ao usuário -----------------------------------------------------------------


def test_us1_citacao_inventada_e_removida_e_registrada() -> None:
    resultado = validar("Liderança [[man-pio-2026#4.2.1@p87]], regras [[por-2023#12]] e invenção [[man-x#9]].")
    assert resultado.answer == "Liderança [1], regras [2] e invenção."
    assert [citacao["chunk_id"] for citacao in resultado.citations or ()] == [A.chunk_id, B.chunk_id]
    assert resultado.issues == ("citação fora do contexto da geração: man-x#9",)
    assert resultado.flags == {"invalid_citation"}
    assert resultado.ok is False


def test_us1_recuperado_mas_cortado_do_contexto_e_invalido() -> None:
    resultado = validar("Ver o guia [[guia-esc#3]].", contexto=(A,), reranked=(C,))
    assert resultado.answer is not None and "[[" not in resultado.answer
    assert resultado.citations == ()
    assert "ungrounded" in resultado.flags


def test_us1_todas_validas() -> None:
    resultado = validar("Ver [[por-2023#12]].")
    assert resultado == ValidationResult(
        ok=True,
        answer="Ver [1].",
        citations=(
            {
                "chunk_id": B.chunk_id,
                "document": "POR",
                "version": "2023",
                "status": "vigente",
                "section": "12",
                "page": None,
                "url": None,
            },
        ),
    )


# --- casos de borda ---------------------------------------------------------------------------------------------------


def test_borda_formatacao_do_identificador() -> None:
    resultado = validar("Ver [[ POR-2023 #12 ]].")
    assert resultado.answer == "Ver [1]."
    assert (resultado.citations or ())[0]["chunk_id"] == "por-2023#12"  # o id canônico, não a grafia do modelo


def test_borda_citacao_repetida() -> None:
    resultado = validar("Um [[por-2023#12]], outro [[man-pio-2026#4.2.1@p87]], de novo [[por-2023#12]].")
    assert resultado.answer == "Um [1], outro [2], de novo [1]."
    assert len(resultado.citations or ()) == 2


def test_borda_resposta_ausente_nao_muda_nada() -> None:
    assert validar(None) == ValidationResult(ok=True)


@pytest.mark.parametrize(
    "resposta",
    [
        "Liderança [[man-pio-2026#4.2.1@p87]] e invenção [[man-x#9]].",
        "Sem citação nenhuma.",
        "",
    ],
)
def test_borda_deterministico(resposta: str) -> None:
    assert validar(resposta) == validar(resposta)


# --- US2: resposta sem fundamento é marcada ------------------------------------------------------------------------


def test_us2_so_citacoes_invalidas() -> None:
    resultado = validar("Segundo o manual [[man-x#9]].")
    assert resultado.answer == f"Segundo o manual.\n\n{UNGROUNDED_NOTICE}"
    assert resultado.citations == ()
    assert resultado.issues == ("citação fora do contexto da geração: man-x#9", "nenhuma citação válida")
    assert resultado.flags == {"ungrounded", "invalid_citation"}
    assert resultado.ok is False


def test_us2_sem_nenhum_marcador() -> None:
    resultado = validar("Uma resposta qualquer.")
    assert resultado.answer == f"Uma resposta qualquer.\n\n{UNGROUNDED_NOTICE}"
    assert resultado.issues == ("nenhuma citação válida",)
    assert resultado.flags == {"ungrounded"}


def test_us2_resposta_vazia_e_contexto_vazio() -> None:
    resultado = validar("", contexto=())
    assert resultado.answer == UNGROUNDED_NOTICE
    assert resultado.flags == {"ungrounded"}


# --- US4: proveniência exibível ------------------------------------------------------------------------------------


def test_us4_citacao_traz_os_seis_campos_do_trecho() -> None:
    resultado = validar("Ver [[man-pio-2026#4.2.1@p87]].")
    assert list(resultado.citations or ()) == [
        {
            "chunk_id": "man-pio-2026#4.2.1@p87",
            "document": "Manual Pioneiro",
            "version": "2026",
            "status": "vigente",
            "section": "4.2.1",
            "page": 87,
            "url": None,
        }
    ]


@pytest.mark.parametrize(
    ("source_url", "page", "url"),
    [
        ("https://escoteiros.org.br/por.pdf", 87, "https://escoteiros.org.br/por.pdf#page=87"),
        ("https://escoteiros.org.br/por.pdf#toolbar=0", 87, "https://escoteiros.org.br/por.pdf#page=87"),
        ("https://escoteiros.org.br/por.pdf", None, "https://escoteiros.org.br/por.pdf"),
        ("ftp://escoteiros.org.br/por.pdf", 87, None),
        ("javascript:alert(1)", 87, None),
        ("https://", 87, None),
        (None, 87, None),
    ],
)
def test_us4_citacao_leva_o_link_do_original_na_pagina(source_url: Any, page: int | None, url: str | None) -> None:
    """FR-010, ADR-051: o `source_url` dos metadados do trecho, com `#page=N`; só `http(s)`."""
    trecho = chunk("por#r13@p87", document="POR", page=page, metadata={"source_url": source_url})
    resultado = validar("Ver [[por#r13@p87]].", contexto=(trecho,))
    assert (resultado.citations or ())[0]["url"] == url
