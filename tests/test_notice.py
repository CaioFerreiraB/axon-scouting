"""`validity_notice` (spec 104 US3; contracts/validators.md casos 4, 5 e 8; research R6; T010)."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from axon_ai_contracts import Citation, ValidationResult
from axon_ai_contracts.testing import turn_view, validator_context

from axon_scouting.validators import ValidityNoticeValidator
from axon_scouting.validators.notice import NON_CURRENT_NOTICE, UNCONFIRMED_NOTICE


def citacao(
    chunk_id: str, *, status: str | None, document: str = "Manual Pioneiro", version: str | None = "2023"
) -> Citation:
    return Citation(chunk_id=chunk_id, document=document, version=version, status=status, section=None, page=None)


def validar(answer: str, *citacoes: Citation) -> ValidationResult:
    view = turn_view(answer=answer, citations=citacoes)
    return asyncio.run(ValidityNoticeValidator().validate(view, validator_context("scouting")))


def aviso(n: int, source: str, status: str) -> str:
    return NON_CURRENT_NOTICE.format(n=n, source=source, status=status)


def test_nome() -> None:
    assert ValidityNoticeValidator.name == "validity_notice"


def test_us3_cenario_1_acrescenta_o_aviso() -> None:
    resultado = validar("A regra era outra [1].", citacao("a", status="substituído"))
    assert resultado.answer == f"A regra era outra [1].\n\n{aviso(1, 'Manual Pioneiro (2023)', 'substituído')}"
    assert resultado.flags == {"non_current_source"}
    assert resultado.ok is True
    assert resultado.citations is None


def test_us3_cenario_2_aviso_padrao_ja_presente() -> None:
    texto = f"A regra era outra [1].\n\n{aviso(1, 'Manual Pioneiro (2023)', 'substituído')}"
    resultado = validar(texto, citacao("a", status="substituído"))
    assert resultado.answer is None
    assert resultado.flags == {"non_current_source"}


@pytest.mark.parametrize(
    "texto",
    [
        "Pelo manual de 2023 [1], que foi revogado, a regra era outra.",
        "A versão de 2023 [1] já foi substituída pela de 2026.",
        "Essa orientação [1] não está vigente.",
    ],
)
def test_us3_cenario_2_modelo_ja_avisou(texto: str) -> None:
    assert validar(texto, citacao("a", status="substituído")).answer is None


def test_us3_aviso_de_outra_citacao_nao_conta() -> None:
    texto = "A de 2026 [1] substituiu a anterior; a de 2023 [2] dizia outra coisa."
    resultado = validar(texto, citacao("a", status="vigente", version="2026"), citacao("b", status="substituído"))
    assert resultado.answer == f"{texto}\n\n{aviso(2, 'Manual Pioneiro (2023)', 'substituído')}"


@pytest.mark.parametrize("status", [None, "", "arquivado"])
def test_vigencia_nao_confirmada(status: str | None) -> None:
    resultado = validar("Ver [1].", citacao("a", status=status))
    esperado = UNCONFIRMED_NOTICE.format(n=1, source="Manual Pioneiro (2023)")
    assert resultado.answer == f"Ver [1].\n\n{esperado}"
    assert resultado.flags == {"non_current_source"}


@pytest.mark.parametrize("status", ["vigente", "Vigente", " VIGENTE "])
def test_vigente_nao_avisa(status: str) -> None:
    assert validar("Ver [1].", citacao("a", status=status)) == ValidationResult(ok=True)


def test_sem_versao_cita_so_o_documento() -> None:
    resultado = validar("Ver [1].", citacao("a", status="revogado", version=None))
    assert resultado.answer == f"Ver [1].\n\n{aviso(1, 'Manual Pioneiro', 'revogado')}"


def test_dois_avisos_na_ordem() -> None:
    resultado = validar("A [1] e B [2].", citacao("a", status="rascunho"), citacao("b", status=None, document="POR"))
    linhas = (resultado.answer or "").split("\n\n", 1)[1].splitlines()
    assert linhas == [
        aviso(1, "Manual Pioneiro (2023)", "rascunho"),
        UNCONFIRMED_NOTICE.format(n=2, source="POR (2023)"),
    ]


def test_reexecutar_nao_duplica() -> None:
    primeira = validar("Ver [1].", citacao("a", status="revogado"))
    segunda = validar(primeira.answer or "", citacao("a", status="revogado"))
    assert segunda.answer is None


@pytest.mark.parametrize("resposta", ["Ver [1].", ""])
def test_deterministico_e_sem_citacoes(resposta: Any) -> None:
    assert validar(resposta) == ValidationResult(ok=True)
    assert validar(resposta, citacao("a", status="revogado")) == validar(resposta, citacao("a", status="revogado"))
