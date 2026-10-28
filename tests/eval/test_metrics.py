"""Métricas de domínio do scouting, caso a caso (spec 105 FR-003; contracts/domain-metrics.md; R2, R3; T008)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import pytest
from axon_ai_contracts import Citation
from axon_ai_contracts.eval import EvalChunk, EvalItem, EvalMetric, EvalObservation

from axon_scouting.eval import (
    AbsenceClaimRate,
    ContextAccuracy,
    CurrentDocumentRate,
    FactRecall,
    ValidityNoticeRate,
    expected_facts,
)

BRASILIA = timezone(timedelta(hours=-3))
MEIO_DIA = datetime(2026, 10, 13, 12, tzinfo=BRASILIA)


def _item(
    question: str = "Como é realizada a Roca de Conselho na alcateia?",
    *,
    context: Mapping[str, Any] | None = None,
    reference_time: datetime = MEIO_DIA,
) -> EvalItem:
    extra: dict[str, Any] = {"provisional": True}
    if context is not None:
        extra["context"] = dict(context)
    return EvalItem(id="i1", question=question, reference_time=reference_time, extra=extra)


def _chunk(chunk_id: str, status: str | None) -> EvalChunk:
    return EvalChunk(chunk_id=chunk_id, score=0.9, document="Documento", version="1", status=status, page=1)


def _observation(
    *,
    delivered: Sequence[EvalChunk] = (),
    retrieved: Sequence[EvalChunk] = (),
    citations: Sequence[Citation] = (),
    flags: frozenset[str] = frozenset(),
) -> EvalObservation:
    return EvalObservation(
        answer="resposta", delivered=delivered, retrieved=retrieved, citations=citations, flags=flags
    )


def _citation(status: str | None) -> Citation:
    citation: Citation = {"chunk_id": "c1", "document": "Documento", "version": "1"}
    if status is not None:
        citation = {**citation, "status": status}
    return citation


# --- superfície -----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("metric", "name", "better", "deterministic"),
    [
        (ContextAccuracy(), "context_accuracy", "higher", True),
        (CurrentDocumentRate(), "current_document_rate", "higher", True),
        (ValidityNoticeRate(), "validity_notice_rate", "higher", False),
        (FactRecall({}.get), "fact_recall", "higher", True),
        (AbsenceClaimRate(), "absence_claim_rate", "lower", False),
    ],
)
def test_superficie(metric: EvalMetric, name: str, better: str, deterministic: bool) -> None:
    assert isinstance(metric, EvalMetric)
    assert (metric.name, metric.better, metric.deterministic) == (name, better, deterministic)


# --- context_accuracy -----------------------------------------------------------------------------------------------

ACCURACY = ContextAccuracy()


@pytest.mark.parametrize("context", [None, {}])
def test_contexto_caso_1_sem_contexto_nao_se_aplica(context: Mapping[str, Any] | None) -> None:
    assert ACCURACY.measure(_item(context=context), _observation()) is None


def test_contexto_caso_2_ramo_casado() -> None:
    assert ACCURACY.measure(_item(context={"branch": "lobinho"}), _observation()) == 1.0


def test_contexto_caso_3_sem_ramo_esperado_e_derivado() -> None:
    item = _item("Quando foi fundada a União dos Escoteiros do Brasil?", context={"branch": None})
    assert ACCURACY.measure(item, _observation()) == 1.0


def test_contexto_caso_4_temas_sao_conjunto() -> None:
    item = _item("Qual uniforme usar nas cerimônias?", context={"topics": ["uniforme"]})
    assert ACCURACY.measure(item, _observation()) == 0.0


def test_contexto_caso_5_ordem_dos_tipos_nao_conta() -> None:
    item = _item(
        "O que o estatuto e os princípios, organização e regras dizem sobre a assembleia?",
        context={"document_types": ["estatuto", "por"]},
    )
    assert ACCURACY.measure(item, _observation()) == 1.0


def test_contexto_caso_6_tres_de_quatro() -> None:
    context = {"branch": "lobinho", "topics": [], "include_superseded": False, "document_types": ["manual"]}
    assert ACCURACY.measure(_item(context=context), _observation()) == 0.75


def test_contexto_caso_7_consulta_historica() -> None:
    context = {"include_superseded": True, "effective_from": "2022-01-01", "effective_until": "2022-12-31"}
    item = _item("Qual era a regra de uniforme em 2022?", context=context)
    assert ACCURACY.measure(item, _observation()) == 1.0


def test_contexto_caso_8_dia_de_brasilia() -> None:
    noite = datetime(2026, 10, 13, 22, 30, tzinfo=BRASILIA).astimezone(UTC)
    assert noite.date().isoformat() == "2026-10-14"
    context = {"effective_from": "2026-10-13", "effective_until": "2026-10-13"}
    assert ACCURACY.measure(_item(context=context, reference_time=noite), _observation()) == 1.0


def test_contexto_caso_9_campo_desconhecido_ignorado() -> None:
    assert ACCURACY.measure(_item(context={"branch": "lobinho", "ramo": "x"}), _observation()) == 1.0
    assert ACCURACY.measure(_item(context={"ramo": "x"}), _observation()) is None


# --- current_document_rate ------------------------------------------------------------------------------------------

CURRENT = CurrentDocumentRate()
VIGENTES = tuple(_chunk(f"v{n}", "vigente") for n in range(5))


def test_vigente_caso_1_item_historico_nao_se_aplica() -> None:
    item = _item(context={"include_superseded": True})
    assert CURRENT.measure(item, _observation(delivered=VIGENTES)) is None
    assert CURRENT.measure(_item(context={"previous_version": True}), _observation(delivered=VIGENTES)) is None


def test_vigente_caso_2_sem_entregues_nao_se_aplica() -> None:
    assert CURRENT.measure(_item(), _observation()) is None


def test_vigente_caso_3_todos_vigentes() -> None:
    assert CURRENT.measure(_item(), _observation(delivered=VIGENTES)) == 1.0


def test_vigente_caso_4_um_substituido_em_quatro() -> None:
    delivered = (*VIGENTES[:3], _chunk("s1", "substituido"))
    assert CURRENT.measure(_item(), _observation(delivered=delivered)) == 0.75


def test_vigente_caso_5_status_ausente_nao_e_vigente() -> None:
    delivered = (VIGENTES[0], _chunk("x1", None))
    assert CURRENT.measure(_item(), _observation(delivered=delivered)) == 0.5


def test_vigente_caso_6_mede_o_que_chegou_ao_modelo() -> None:
    retrieved = (_chunk("r1", "revogado"), _chunk("r2", "substituido"), *VIGENTES[:2])
    assert CURRENT.measure(_item(), _observation(delivered=VIGENTES[:2], retrieved=retrieved)) == 1.0


# --- validity_notice_rate -------------------------------------------------------------------------------------------

NOTICE = ValidityNoticeRate()
FLAG = frozenset({"non_current_source"})


def test_aviso_caso_1_sem_citacao_nao_vigente_nao_se_aplica() -> None:
    assert NOTICE.measure(_item(), _observation()) is None
    assert NOTICE.measure(_item(), _observation(citations=(_citation("vigente"),), flags=FLAG)) is None


def test_aviso_caso_2_revogado_com_aviso() -> None:
    citations = (_citation("vigente"), _citation("revogado"))
    assert NOTICE.measure(_item(), _observation(citations=citations, flags=FLAG)) == 1.0


def test_aviso_caso_3_substituido_sem_aviso() -> None:
    assert NOTICE.measure(_item(), _observation(citations=(_citation("substituido"),))) == 0.0


def test_aviso_caso_4_status_ausente_com_aviso() -> None:
    assert NOTICE.measure(_item(), _observation(citations=(_citation(None),), flags=FLAG)) == 1.0


# --- puras ------------------------------------------------------------------------------------------------------------


def test_duas_chamadas_dao_o_mesmo_valor() -> None:
    item = _item(context={"branch": "lobinho", "document_types": ["manual"]})
    observation = _observation(delivered=(VIGENTES[0], _chunk("s1", "substituido")))
    for metric in (ACCURACY, CURRENT, NOTICE):
        assert metric.measure(item, observation) == metric.measure(item, observation)


# --- fatos entregues (FR-014; research R13) ------------------------------------------------------------------------

TEXTOS = {
    "por#s8@p62": "[POR · Capítulo 9]\nII - O marco simbólico do Ramo Lobinho é “Ser Livre como os Lobos”.",
    "manual#cap3@p56": "O marco simbólico do Ramo Sênior, “Viver aventuras, superar desafios”, traduz…",
    "manual#hifen@p1": "Identifica os elementos que com\ufffepõem o marco.",
}


def _fatos(*facts: object) -> EvalItem:
    return EvalItem(id="f1", question="pergunta", reference_time=MEIO_DIA, extra={"facts": list(facts)})


def _entregues(*ids: str) -> EvalObservation:
    return _observation(delivered=[_chunk(chunk_id, "vigente") for chunk_id in ids])


def test_fatos_normalizados_e_alternativas() -> None:
    assert expected_facts({"facts": ["Ser livre como os LOBOS!", ["Viver aventuras", "superar desafios"], 3, []]}) == (
        ("ser livre como os lobos",),
        ("viver aventuras", "superar desafios"),
    )
    assert expected_facts({}) == () and expected_facts({"facts": "texto"}) == ()


@pytest.mark.parametrize(
    ("facts", "delivered", "expected"),
    [
        (["ser livre como os lobos", "viver aventuras, superar desafios"], ["por#s8@p62", "manual#cap3@p56"], 1.0),
        (["ser livre como os lobos", "viver aventuras, superar desafios"], ["por#s8@p62"], 0.5),
        ([["explorar o mundo", "ser livre como os lobos"]], ["por#s8@p62"], 1.0),
        (["ser livre como os lobos"], [], 0.0),
        (["ser livre como os lobos"], ["desconhecido"], 0.0),
        (["livre como os lobo"], ["por#s8@p62"], 0.0),
        (["compõem o marco"], ["manual#hifen@p1"], 0.0),
    ],
    ids=[
        "todos",
        "metade",
        "alternativa",
        "nada-entregue",
        "texto-desconhecido",
        "palavra-inteira",
        "hifen-do-pdfium-conta-como-ausente",
    ],
)
def test_fact_recall(facts: list[object], delivered: list[str], expected: float) -> None:
    assert FactRecall(TEXTOS.get).measure(_fatos(*facts), _entregues(*delivered)) == expected


def test_fact_recall_sem_fatos_nao_conta() -> None:
    assert FactRecall(TEXTOS.get).measure(_item(), _entregues("por#s8@p62")) is None


# --- afirmação de ausência no acervo (FR-017 da 105; research R17) -------------------------------------------------


@pytest.mark.parametrize(
    ("resposta", "esperado"),
    [
        ("Os documentos oficiais não trazem as demais formulações.", 1.0),
        ("Não há, nos documentos consultados, o marco do Ramo Escoteiro.", 1.0),
        ("Nenhum documento oficial foi encontrado para a pergunta.", 1.0),
        ("O marco do Lobinho é “Ser livre como os lobos” [1]. Os demais não apareceram nos trechos recebidos.", 0.0),
        ("Os trechos consultados não trazem o lema do Ramo Sênior.", 0.0),
    ],
)
def test_absence_claim(resposta: str, esperado: float) -> None:
    observacao = EvalObservation(answer=resposta)
    assert AbsenceClaimRate().measure(_fatos("ser livre como os lobos"), observacao) == esperado


def test_absence_claim_sem_fatos_nao_conta() -> None:
    assert AbsenceClaimRate().measure(_item(), EvalObservation(answer="Os documentos não trazem isso.")) is None


def test_contexto_pedido_de_lista() -> None:
    """O `extra` chega congelado (termos em tupla); a espécie e os termos contam (ADR-047)."""
    pergunta = "Quais são as políticas nacionais que existem?"
    certo = {"catalog": {"kind": "politica", "terms": ["nacionais"]}}
    assert ACCURACY.measure(_item(pergunta, context=certo), _observation()) == 1.0
    errado = {"catalog": {"kind": "manual", "terms": ["nacionais"]}}
    assert ACCURACY.measure(_item(pergunta, context=errado), _observation()) == 0.0
    assert ACCURACY.measure(_item(context={"catalog": None}), _observation()) == 1.0
