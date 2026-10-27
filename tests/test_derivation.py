"""Derivação de contexto da pergunta, sem modelo (spec 101 US2, US3; data-model; research R8, R12; T016 e T022)."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, date, datetime

import pytest
from axon_ai_contracts.testing import pack_context, turn_view

from axon_scouting.context.derivation import ScoutingContext, derive_context, to_slice
from axon_scouting.context.periods import Validity
from axon_scouting.pack import ScoutingPack

DIA = date(2026, 10, 7)
HOJE = Validity.day(DIA)

# --- US2: ramo, temas e tipos -------------------------------------------------------------------------------------


def test_us2_cenario_1() -> None:
    contexto = derive_context("Quais competências do Pioneiro estão relacionadas à liderança?", DIA)
    assert contexto == ScoutingContext(
        branch="pioneiro",
        branches=(),
        topics=("lideranca",),
        document_types=("manual", "programa_educativo"),
        validity=HOJE,
        include_superseded=False,
        previous_version=False,
    )


def test_us2_cenario_2_sem_ramo() -> None:
    contexto = derive_context("Como funciona a promessa?", DIA)
    assert contexto.branch is None
    assert contexto.topics == ("promessa_e_lei",)
    assert contexto.document_types == ("manual", "por")
    assert contexto.validity == HOJE


@pytest.mark.parametrize(
    ("pergunta", "ramo"),
    [
        ("O que um escoteiro deve levar no acampamento?", None),  # "escoteiro" sozinho é ambíguo
        ("Qual a diferença entre lobinho e sênior?", None),  # dois ramos: na dúvida, sem filtro
        ("O que faz um rover?", "pioneiro"),  # sinônimo antigo
        ("Como é a tropa escoteira?", "escoteiro"),
        ("Escoteiros seniores podem acampar sozinhos?", "senior"),
        ("COMPETÊNCIAS DO PIONEIRO", "pioneiro"),
        ("Atividades de pioneirismo para a tropa", None),  # pioneirismo é tema, não o Ramo Pioneiro
    ],
)
def test_ramo(pergunta: str, ramo: str | None) -> None:
    assert derive_context(pergunta, DIA).branch == ramo


def test_tipo_citado_prevalece_sobre_os_sugeridos() -> None:
    contexto = derive_context("O que diz o estatuto sobre liderança?", DIA)
    assert contexto.document_types == ("estatuto",)
    assert contexto.topics == ("lideranca",)


def test_pergunta_vazia_da_o_contexto_default() -> None:
    assert derive_context("", DIA) == ScoutingContext(
        branch=None,
        branches=(),
        topics=(),
        document_types=(),
        validity=HOJE,
        include_superseded=False,
        previous_version=False,
    )


# --- US2: a fatia publicada (R12; contracts/retrieval-filters.md) -------------------------------------------------


def test_fatia_com_ramo() -> None:
    fatia = to_slice(derive_context("Quais competências do Pioneiro estão relacionadas à liderança?", DIA))
    assert fatia == {
        "context": {
            "branch": "pioneiro",
            "branches": [],
            "topics": ["lideranca"],
            "document_types": ["manual", "programa_educativo"],
            "effective_from": "2026-10-07",
            "effective_until": "2026-10-07",
            "include_superseded": False,
            "previous_version": False,
        },
        "retrieval_filters": {
            "branch": "pioneiro",
            "topics": ["lideranca"],
            "document_types": ["manual", "programa_educativo"],
            "effective_from": "2026-10-07",
            "effective_until": "2026-10-07",
            "include_superseded": False,
            "previous_version": False,
        },
    }
    json.dumps(fatia)


def test_filtros_omitem_ramo_e_listas_vazias() -> None:
    fatia = to_slice(derive_context("", DIA))
    assert fatia["context"]["branch"] is None  # type: ignore[index]
    assert fatia["retrieval_filters"] == {
        "effective_from": "2026-10-07",
        "effective_until": "2026-10-07",
        "include_superseded": False,
        "previous_version": False,
    }


def test_prepare_context_publica_a_fatia() -> None:
    pergunta = "Quais competências do Pioneiro estão relacionadas à liderança?"
    view = turn_view(question=pergunta, reference_time=datetime(2026, 10, 8, 1, tzinfo=UTC))  # 22h do dia 7 em Brasília
    fatia = asyncio.run(ScoutingPack().prepare_context(view, pack_context("scouting")))
    assert fatia == to_slice(derive_context(pergunta, DIA))


# --- US3: consulta histórica ---------------------------------------------------------------------------------------


def test_historico_us3_cenario_1() -> None:
    contexto = derive_context("Qual era a regra em 2023?", DIA)
    assert contexto.validity == Validity(start=date(2023, 1, 1), end=date(2023, 12, 31))
    assert contexto.include_superseded is True
    assert contexto.previous_version is False
    assert contexto.document_types == ("por", "resolucao")


def test_historico_us3_cenario_2() -> None:
    contexto = derive_context("O que dizia o manual anterior?", DIA)
    assert contexto.validity == HOJE
    assert contexto.include_superseded is True
    assert contexto.previous_version is True
    assert contexto.document_types == ("manual",)


def test_historico_antigamente_mira_a_versao_anterior() -> None:
    contexto = derive_context("Como era a promessa antigamente?", DIA)
    assert contexto.previous_version is True
    assert contexto.include_superseded is True


def test_historico_filtros_com_o_periodo() -> None:
    filtros = to_slice(derive_context("Qual era a regra em 2023?", DIA))["retrieval_filters"]
    assert filtros == {
        "document_types": ["por", "resolucao"],
        "effective_from": "2023-01-01",
        "effective_until": "2023-12-31",
        "include_superseded": True,
        "previous_version": False,
    }


# --- facetas e o Ramo Filhotes (ADR-045) ---------------------------------------------------------------------------

TODOS = ("filhotes", "lobinho", "escoteiro", "senior", "pioneiro")


@pytest.mark.parametrize(
    ("pergunta", "branch", "branches"),
    [
        ("Qual o marco simbólico dos 5 ramos?", None, TODOS),
        ("Qual o lema de cada ramo?", None, TODOS),
        ("Quais os marcos simbólicos dos cinco ramos?", None, TODOS),
        ("Qual a frase do marco simbólico dos ramos Filhotes, Lobinho, Escoteiro, Sênior e Pioneiro?", None, TODOS),
        ("Qual a diferença entre o marco simbólico do Ramo Lobinho e o do Ramo Sênior?", None, ("lobinho", "senior")),
        ("Qual o marco simbólico do Ramo Filhotes?", "filhotes", ()),
        ("Quantas crianças cabem numa ninhada?", "filhotes", ()),
        ("O que é a promessa escoteira?", None, ()),
    ],
)
def test_facetas(pergunta: str, branch: str | None, branches: tuple[str, ...]) -> None:
    contexto = derive_context(pergunta, DIA)
    assert (contexto.branch, contexto.branches) == (branch, branches)


def test_fatia_com_facetas_nao_tem_branch() -> None:
    filtros = to_slice(derive_context("Qual o lema de cada ramo?", DIA))["retrieval_filters"]
    assert isinstance(filtros, dict) and "branch" not in filtros and filtros["branches"] == list(TODOS)
