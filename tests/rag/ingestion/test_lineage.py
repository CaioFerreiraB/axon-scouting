"""Vigência e sucessão derivadas do catálogo (spec 102 FR-007, FR-008, SC-003; research R4; T010)."""

from __future__ import annotations

import itertools
from datetime import date
from pathlib import PurePosixPath
from typing import Any

from axon_scouting.rag.index.api import ValidityStatus
from axon_scouting.rag.ingestion.domain.catalog import CatalogEntry, validate_catalog
from axon_scouting.rag.ingestion.domain.lineage import LineageResult, Validity, derive_lineages


def versoes(*specs: dict[str, Any]) -> tuple[CatalogEntry, ...]:
    raw = []
    for spec in specs:
        entry: dict[str, Any] = {
            "file": f"{spec.get('document_id', 'manual-x')}-{spec['version']}.pdf",
            "document_id": "manual-x",
            "title": "Manual X",
            "jurisdiction": "nacional",
        }
        entry.update(spec)
        raw.append(entry)
    resultado = validate_catalog({"documents": raw}, PurePosixPath("/c"))
    assert resultado.ok, resultado.problems
    return resultado.entries


def derivar(*specs: dict[str, Any]) -> LineageResult:
    return derive_lineages(versoes(*specs))


def problema(resultado: LineageResult) -> str:
    assert not resultado.ok
    assert len(resultado.problems) == 1, resultado.problems
    assert resultado.problems[0].document_id == "manual-x"
    return resultado.problems[0].message


V2023: dict[str, Any] = {"version": "2023", "effective_date": date(2023, 1, 1)}
V2026: dict[str, Any] = {"version": "2026", "effective_date": date(2026, 3, 1), "supersedes": "2023"}


def test_versao_unica_vigente() -> None:
    resultado = derivar(V2023)
    assert resultado.ok
    assert resultado.validities["manual-x-2023"] == Validity(
        status=ValidityStatus.VIGENTE, effective_date=date(2023, 1, 1)
    )


def test_sucessor_substitui_o_antecessor() -> None:
    resultado = derivar(V2023, V2026)
    assert resultado.ok
    assert resultado.validities["manual-x-2023"] == Validity(
        status=ValidityStatus.SUBSTITUIDO,
        effective_date=date(2023, 1, 1),
        expiry_date=date(2026, 2, 28),
        superseded_by="manual-x-2026",
        successor_expiry_date=None,
    )
    assert resultado.validities["manual-x-2026"].status is ValidityStatus.VIGENTE


def test_cadeia_guarda_o_fim_do_sucessor() -> None:
    resultado = derivar(
        {"version": "2019", "effective_date": date(2019, 1, 1)},
        {"version": "2023", "effective_date": date(2023, 1, 1), "supersedes": "2019"},
        V2026,
    )
    assert resultado.ok
    v2019 = resultado.validities["manual-x-2019"]
    assert v2019.status is ValidityStatus.SUBSTITUIDO
    assert v2019.expiry_date == date(2022, 12, 31)
    assert v2019.superseded_by == "manual-x-2023"
    assert v2019.successor_expiry_date == date(2026, 2, 28)


def test_sucessor_rascunho_nao_substitui() -> None:
    resultado = derivar(V2023, {**V2026, "status": "rascunho"})
    assert resultado.ok
    assert resultado.validities["manual-x-2023"].status is ValidityStatus.VIGENTE
    assert resultado.validities["manual-x-2026"].status is ValidityStatus.RASCUNHO


def test_revogado_mantem_o_fim_declarado() -> None:
    resultado = derivar({**V2023, "status": "revogado", "expiry_date": date(2024, 6, 30)})
    assert resultado.ok
    assert resultado.validities["manual-x-2023"] == Validity(
        status=ValidityStatus.REVOGADO, effective_date=date(2023, 1, 1), expiry_date=date(2024, 6, 30)
    )


def test_fim_declarado_coerente_e_aceito() -> None:
    assert derivar({**V2023, "expiry_date": date(2026, 2, 28)}, V2026).ok


def test_erros_de_linhagem() -> None:
    assert "2022" in problema(derivar(V2023, {**V2026, "supersedes": "2022"}))
    mensagem = problema(
        derivar(V2023, V2026, {"version": "2027", "effective_date": date(2027, 1, 1), "supersedes": "2023"})
    )
    assert "2026" in mensagem and "2027" in mensagem
    assert "ciclo" in problema(
        derivar(
            {"version": "2023", "effective_date": date(2023, 1, 1), "supersedes": "2026"},
            V2026,
        )
    )
    assert "2026" in problema(derivar(V2023, {**V2026, "effective_date": date(2023, 1, 1)}))
    assert "expiry_date" in problema(derivar({**V2023, "expiry_date": date(2025, 12, 31)}, V2026))
    mensagem = problema(derivar(V2023, {"version": "2026", "effective_date": date(2026, 3, 1)}))
    assert "vigente" in mensagem and "2023" in mensagem and "2026" in mensagem
    assert "revogad" in problema(derivar({**V2023, "status": "revogado", "expiry_date": date(2024, 1, 1)}, V2026))


def test_linhagens_sao_independentes() -> None:
    resultado = derive_lineages(versoes(V2023, {**V2023, "document_id": "por"}))
    assert resultado.ok
    assert {v.status for v in resultado.validities.values()} == {ValidityStatus.VIGENTE}


def test_no_maximo_uma_vigente_por_linhagem() -> None:
    """Propriedade (SC-003): para toda combinação de três versões e elos de substituição, um catálogo aceito tem no
    máximo uma `vigente` por linhagem."""
    anos = ("2019", "2023", "2026")
    opcoes = (None, *anos)
    for elos in itertools.product(opcoes, repeat=3):
        for rascunhos in itertools.product((False, True), repeat=3):
            specs = []
            for ano, elo, rascunho in zip(anos, elos, rascunhos, strict=True):
                spec: dict[str, Any] = {"version": ano, "effective_date": date(int(ano), 1, 1)}
                if elo is not None:
                    spec["supersedes"] = elo
                if rascunho:
                    spec["status"] = "rascunho"
                specs.append(spec)
            resultado = derivar(*specs)
            if resultado.ok:
                vigentes = [k for k, v in resultado.validities.items() if v.status is ValidityStatus.VIGENTE]
                assert len(vigentes) <= 1, (elos, rascunhos, vigentes)
