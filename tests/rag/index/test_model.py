"""Modelo publicado pelo seam do índice (spec 102: data-model, contracts/chunk-index.md; research R14; T004)."""

from __future__ import annotations

import dataclasses
from datetime import UTC, date, datetime

import pytest
from axon_ai_contracts import Chunk

from axon_scouting.rag.index.api import (
    BRANCHES,
    NORMATIVE_TYPES,
    STATUS_LABELS,
    DocumentType,
    EmbeddingModel,
    IndexedChunk,
    IndexFilters,
    IndexMismatch,
    IndexVersion,
    ValidityStatus,
)


def chunk(**changes: object) -> IndexedChunk:
    base = IndexedChunk(
        chunk_id="por-2025#regra33@p9",
        text="[POR · Capítulo 4]\nRegra 033 – O uniforme…",
        doc_key="por-2025",
        document_id="por",
        version="2025",
        document_type=DocumentType.POR,
        document_title="Princípios, Organização e Regras (POR)",
        jurisdiction="nacional",
        page_start=9,
        page_end=10,
        effective_date=date(2025, 1, 1),
        status=ValidityStatus.VIGENTE,
        chapter="Capítulo 4 – Uniformes",
        article="Regra 33",
    )
    return dataclasses.replace(base, **changes)  # type: ignore[arg-type]


def test_vocabularios() -> None:
    assert [s.value for s in ValidityStatus] == ["vigente", "revogado", "substituido", "rascunho"]
    assert STATUS_LABELS[ValidityStatus.SUBSTITUIDO] == "substituído"
    assert set(STATUS_LABELS) == set(ValidityStatus)
    assert [t.value for t in DocumentType] == ["por", "resolucao", "estatuto", "manual", "programa_educativo", "guia"]
    assert {t.value for t in NORMATIVE_TYPES} == {"por", "resolucao", "estatuto"}
    assert BRANCHES == ("lobinho", "escoteiro", "senior", "pioneiro")


def test_filtros_recusam_intervalo_invertido() -> None:
    with pytest.raises(ValueError, match="effective_from"):
        IndexFilters(effective_from=date(2026, 10, 8), effective_until=date(2026, 10, 7))


def test_filtros_normalizam_tipos_e_recusam_desconhecidos() -> None:
    dia = date(2026, 10, 7)
    filtros = IndexFilters(effective_from=dia, effective_until=dia, document_types=("manual", "por"))  # type: ignore[arg-type]
    assert filtros.document_types == (DocumentType.MANUAL, DocumentType.POR)
    with pytest.raises(ValueError):
        IndexFilters(effective_from=dia, effective_until=dia, document_types=("lei",))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="branch"):
        IndexFilters(effective_from=dia, effective_until=dia, branch="pioneiros")


@pytest.mark.parametrize(
    "changes",
    [
        {"page_start": 0},
        {"page_end": 8},
        {"chunk_id": "por-2025#regra 33@p9"},
        {"chunk_id": "por-2025#[regra33]@p9"},
        {"chunk_id": "Por-2025#regra33@p9"},
        {"text": ""},
        {"branch": "pioneiros"},
    ],
)
def test_chunk_invalido(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        chunk(**changes)


def test_to_chunk_cumpre_o_contrato_001() -> None:
    convertido = chunk(
        expiry_date=date(2026, 2, 28),
        superseded_by="por-2026",
        status=ValidityStatus.SUBSTITUIDO,
        branch="senior",
        score=0.5,
    ).to_chunk()
    assert isinstance(convertido, Chunk)
    assert convertido.chunk_id == "por-2025#regra33@p9"
    assert convertido.document == "Princípios, Organização e Regras (POR)"
    assert convertido.version == "2025"
    assert convertido.status == "substituido"
    assert convertido.section == "Regra 33"
    assert convertido.page == 9
    assert convertido.score == 0.5
    assert convertido.metadata["document_id"] == "por"
    assert convertido.metadata["document_type"] == "por"
    assert convertido.metadata["branch"] == "senior"
    assert convertido.metadata["jurisdiction"] == "nacional"
    assert convertido.metadata["chapter"] == "Capítulo 4 – Uniformes"
    assert convertido.metadata["page_end"] == 10
    assert convertido.metadata["effective_date"] == "2025-01-01"
    assert convertido.metadata["expiry_date"] == "2026-02-28"
    assert convertido.metadata["superseded_by"] == "por-2026"


def test_to_chunk_secao_mais_especifica_e_score_ausente() -> None:
    sem_artigo = chunk(article=None, section="Seção I – Do uniforme", subsection=None)
    assert sem_artigo.to_chunk().section == "Seção I – Do uniforme"
    assert chunk(article=None, section=None, subsection=None).to_chunk().section == "Capítulo 4 – Uniformes"
    assert chunk().to_chunk().score == 0.0


def versao(**changes: object) -> IndexVersion:
    base = IndexVersion(
        chunking="estrutural-v2;janela=250-450",
        embedding=EmbeddingModel(name="intfloat/multilingual-e5-small", fingerprint="ca456c06", dimensions=384),
        sparse="bm25-pt-v1;k1=1.2;b=0.75;avgdl=256",
        built_at=datetime(2026, 10, 7, 15, 30, tzinfo=UTC),
    )
    return dataclasses.replace(base, **changes)  # type: ignore[arg-type]


def test_versao_compativel_ignora_data() -> None:
    assert versao().compatible_with(versao(built_at=datetime(2027, 1, 1, tzinfo=UTC)))
    assert not versao().compatible_with(versao(chunking="estrutural-v2;janela=200-400"))
    assert not versao().compatible_with(versao(sparse="bm25-pt-v2"))
    for embedding in (
        EmbeddingModel(name="BAAI/bge-m3", fingerprint="ca456c06", dimensions=384),
        EmbeddingModel(name="intfloat/multilingual-e5-small", fingerprint="outro", dimensions=384),
        EmbeddingModel(name="intfloat/multilingual-e5-small", fingerprint="ca456c06", dimensions=1024),
    ):
        assert not versao().compatible_with(versao(embedding=embedding))


def test_versao_exige_horario_com_fuso() -> None:
    with pytest.raises(ValueError, match="built_at"):
        versao(built_at=datetime(2026, 10, 7, 15, 30))


def test_mismatch_nomeia_os_dois() -> None:
    erro = IndexMismatch(index="e5-small (sha256 ca456c06)", encoder="hashing-test (sha256 fixo)")
    assert "e5-small (sha256 ca456c06)" in str(erro)
    assert "hashing-test (sha256 fixo)" in str(erro)
    assert erro.index == "e5-small (sha256 ca456c06)"
    assert erro.encoder == "hashing-test (sha256 fixo)"


def test_similaridade_valida_e_fora_da_conversao() -> None:
    assert chunk(similarity=0.42).similarity == 0.42
    assert chunk(similarity=1.0000001).similarity == 1.0000001  # tolerância de arredondamento
    for fora in (1.01, -1.01):
        with pytest.raises(ValueError, match="similarity"):
            chunk(similarity=fora)
    assert chunk(score=0.5, similarity=0.9).to_chunk().score == 0.5  # to_chunk usa a fusão (ADR-025)
