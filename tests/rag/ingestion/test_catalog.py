"""Validação do catálogo do corpus (spec 102 FR-011; contracts/corpus-catalog.md; research R3; T009)."""

from __future__ import annotations

from datetime import date
from pathlib import PurePosixPath
from typing import Any

import pytest

from ai_scouting.rag.index.api import DocumentType, ValidityStatus
from ai_scouting.rag.ingestion.domain.catalog import CatalogResult, validate_catalog

BASE = PurePosixPath("/corpus")


def entrada(**changes: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "file": "manuais/pioneiro-2026.pdf",
        "document_id": "manual-pioneiro",
        "version": "2026",
        "type": "manual",
        "title": "Manual do Ramo Pioneiro",
        "branch": "pioneiro",
        "jurisdiction": "nacional",
        "effective_date": date(2026, 3, 1),
    }
    base.update(changes)
    return {k: v for k, v in base.items() if v is not ...}


def validar(*entradas: dict[str, Any]) -> CatalogResult:
    return validate_catalog({"documents": list(entradas)}, BASE)


def unico_problema(resultado: CatalogResult) -> str:
    assert not resultado.ok
    assert len(resultado.problems) == 1, resultado.problems
    return resultado.problems[0].message


def test_catalogo_valido_do_contrato() -> None:
    resultado = validar(
        entrada(),
        entrada(
            file="por/por-2025.pdf",
            document_id="por",
            version="2025",
            type="por",
            title="POR",
            branch=...,
            effective_date="2025-01-01",
            supersedes="2023",
        ),
        entrada(
            file="por/por-2023.pdf",
            document_id="por",
            version="2023",
            type="por",
            title="POR",
            branch=...,
            publication_date=date(2023, 3, 1),
            effective_date=date(2023, 3, 1),
        ),
        entrada(
            file="res.pdf",
            document_id="resolucao-2024-03",
            version="1",
            type="resolucao",
            branch=...,
            status="revogado",
            expiry_date=date(2025, 12, 31),
        ),
    )
    assert resultado.ok, resultado.problems
    manual, por2025, por2023, resolucao = resultado.entries
    assert manual.path == PurePosixPath("/corpus/manuais/pioneiro-2026.pdf")
    assert manual.document_type is DocumentType.MANUAL
    assert manual.branch == "pioneiro"
    assert manual.doc_key == "manual-pioneiro-2026"
    assert por2025.effective_date == date(2025, 1, 1)
    assert por2025.supersedes == "2023"
    assert por2025.branch is None
    assert por2023.publication_date == date(2023, 3, 1)
    assert resolucao.status is ValidityStatus.REVOGADO


def test_raiz_sem_documents() -> None:
    assert "documents" in unico_problema(validate_catalog({}, BASE))
    assert "documents" in unico_problema(validate_catalog({"documents": []}, BASE))
    assert "documents" in unico_problema(validate_catalog(["x"], BASE))


@pytest.mark.parametrize("campo", ["file", "document_id", "version", "type", "title", "jurisdiction", "effective_date"])
def test_obrigatorio_ausente(campo: str) -> None:
    resultado = validar(entrada(**{campo: ...}))
    mensagem = unico_problema(resultado)
    assert campo in mensagem
    assert resultado.problems[0].fields == (campo,)


def test_varios_ausentes_numa_mensagem() -> None:
    resultado = validar(entrada(title=..., jurisdiction=...))
    mensagem = unico_problema(resultado)
    assert resultado.problems[0].fields == ("title", "jurisdiction")
    assert "title" in mensagem and "jurisdiction" in mensagem
    assert "entrada 1" in mensagem


def test_campo_desconhecido() -> None:
    assert "expiry_dte" in unico_problema(validar(entrada(expiry_dte=date(2026, 1, 1))))


@pytest.mark.parametrize(
    ("campo", "valor"),
    [
        ("document_id", "Manual_Pioneiro"),
        ("document_id", "-manual"),
        ("version", "2026-1"),
        ("version", "v 2"),
        ("type", "lei"),
        ("branch", "pioneiros"),
        ("status", "vigente"),
        ("status", "substituido"),
        ("effective_date", "2026-13-01"),
        ("effective_date", 20260301),
        ("title", "  "),
        ("supersedes", "2025-1"),
    ],
)
def test_formato_invalido(campo: str, valor: object) -> None:
    resultado = validar(entrada(**{campo: valor}))
    assert campo in unico_problema(resultado)
    assert resultado.problems[0].fields == (campo,)


def test_versao_numerica_vira_texto_e_decimal_pede_aspas() -> None:
    resultado = validar(entrada(version=2026))
    assert resultado.ok
    assert resultado.entries[0].version == "2026"
    mensagem = unico_problema(validar(entrada(version=1.10)))
    assert "aspas" in mensagem


@pytest.mark.parametrize("arquivo", ["/abs/manual.pdf", "../fora.pdf", "manuais/../../fora.pdf"])
def test_arquivo_fora_do_diretorio(arquivo: str) -> None:
    assert "file" in unico_problema(validar(entrada(file=arquivo)))


def test_par_repetido() -> None:
    resultado = validar(entrada(), entrada(file="outro.pdf"))
    mensagem = unico_problema(resultado)
    assert "manual-pioneiro@2026" in mensagem


def test_revogado_exige_fim_de_vigencia() -> None:
    assert "expiry_date" in unico_problema(validar(entrada(status="revogado")))


def test_todos_os_problemas_acumulados() -> None:
    resultado = validar(entrada(title=...), entrada(file="b.pdf", version="2027", type="lei"), entrada(file="/x.pdf"))
    assert not resultado.ok
    assert [p.entry for p in resultado.problems] == [1, 2, 3]
    assert resultado.entries == ()


def test_entradas_validas_e_linhagens_com_entrada_invalida() -> None:
    resultado = validar(
        entrada(),
        entrada(file="b.pdf", document_id="por", version="1", type="lei"),
        entrada(file="/x.pdf", document_id="estatuto"),
    )
    assert not resultado.ok
    assert resultado.entries == ()
    assert [e.doc_key for e in resultado.valid] == ["manual-pioneiro-2026"]
    assert resultado.invalid_documents == frozenset({"por", "estatuto"})
