"""Ruído de OCR e transcrição (spec 102 FR-033, FR-034; ADR-050)."""

from __future__ import annotations

from pathlib import Path

from axon_scouting.rag.ingestion.adapters.transcription import ParserBySuffix, TranscriptionParser
from axon_scouting.rag.ingestion.domain import ocr_noise
from axon_scouting.rag.ingestion.domain.text import Line

VOCABULARIO = ocr_noise.vocabulary(
    [
        ["O presidente da assembleia colocou em votação a ata e encerrou a reunião anterior."],
        ["A assembleia aprovou a ata e o presidente encerrou a reunião com a votação."],
    ]
)


def test_carimbo_repetido_sai_e_conteudo_repetido_fica() -> None:
    linhas = [
        Line(text=texto, page=pagina)
        for pagina in (1, 2, 3)
        for texto in ("(XX) 41 - 3225-3905 CURITIBA", "RAMO LOBINHO", "O presidente colocou em votação a ata.")
    ]
    limpas = [linha.text for linha in ocr_noise.clean(linhas, VOCABULARIO)]
    assert "(XX) 41 - 3225-3905 CURITIBA" not in limpas
    assert limpas.count("RAMO LOBINHO") == 3


def test_linha_ilegivel_e_rabo_ilegivel_saem() -> None:
    linhas = [
        Line(text="OJsiBoy sejla14 sou1er obÚpor Jeuo!lBay", page=1),
        Line(text="O presidente encerrou a reunião pagéstoNO liZadoO «;", page=1),
    ]
    assert [linha.text for linha in ocr_noise.clean(linhas, VOCABULARIO)] == ["O presidente encerrou a reunião"]


def test_transcricao_por_pagina(tmp_path: Path) -> None:
    arquivo = tmp_path / "ata.txt"
    arquivo.write_text("Página um.\n\fPágina dois.\n\f", encoding="utf-8")
    lido = ParserBySuffix(TranscriptionParser()).parse(arquivo)
    assert lido.pages == ("Página um.\n", "Página dois.\n") and lido.parser == "transcricao-v1"
