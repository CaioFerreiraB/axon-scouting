"""Fichas de reunião e linha do tempo (ADR-062): leitura, conferência contra a página e o que a busca entrega."""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from axon_scouting.rag.ingestion.domain.minutes import (
    MinutesDocument,
    RecordError,
    card_parts,
    confirm,
    problems,
    record_from_mapping,
    record_to_mapping,
    timeline_parts,
)

PAGINAS = [
    "ATA DA XXV REUNIÃO ORDINÁRIA DA ASSEMBLEIA ESCOTEIRA REGIONAL. PIRACICABA/SP, 17 E 18 DE MARÇO DE 2018. "
    "Foi aprovada a Mesa Diretora com o chefe Renato Breneizer.",
    "As contas e balanço do exercício fiscal de 2017 foram aprovadas com 337 votos a favor. "
    "Estavam 404 delegados credenciados.",
    "Foi eleita para a Diretoria Regional a chapa Caminhando Juntos, "
    "composta por Alexandre Banchi (Diretor-Presidente).",
]


def bruto(**extra: Any) -> dict[str, Any]:
    return {
        "doc_key": "sp-ata-are-2018-ordinaria-1",
        "reuniao": {"tipo": "ordinaria", "data": "2018-03-17", "local": "Piracicaba/SP"},
        "deliberacoes": [
            {
                "tema": "Contas de 2017",
                "decisao": "as contas e balanço do exercício fiscal de 2017 foram aprovadas",
                "votacao": "337 votos a favor",
                "pagina": 2,
            },
            {"tema": "Mesa Diretora", "decisao": "foi aprovada a Mesa Diretora", "votacao": "", "pagina": 1},
            {
                "tema": "Sede de 2019",
                "decisao": "aprovada a cidade de Ribeirão Pires como sede",
                "votacao": "",
                "pagina": 3,
            },
        ],
        "eleicoes": [
            {
                "orgao": "Diretoria Regional",
                "cargo": "Diretor-Presidente",
                "eleito": "Alexandre Banchi",
                "mandato": "",
                "pagina": 3,
            },
        ],
        "numeros": [
            {"o_que": "delegados credenciados", "valor": "404", "pagina": 2},
            {"o_que": "delegados credenciados", "valor": "999", "pagina": 2},
        ],
        **extra,
    }


DOCUMENTO = MinutesDocument(
    doc_key="sp-ata-are-2018-ordinaria-1",
    title="Ata da Reunião Ordinária da Assembleia Regional Escoteira de São Paulo — 2018",
    category="Assembleias",
    bodies=("assembleias", "atas"),
    scope="regional",
    source_url="https://exemplo.org/ata-2018.pdf",
    effective_date=date(2018, 3, 17),
)


def test_leitura_e_escrita_ida_e_volta() -> None:
    registro = record_from_mapping(bruto())
    assert registro.data == date(2018, 3, 17) and registro.tipo == "ordinaria"
    assert record_from_mapping(record_to_mapping(registro)) == registro


@pytest.mark.parametrize(
    ("mudanca", "campo"),
    [
        ({"reuniao": {"tipo": "festa"}}, "reuniao.tipo"),
        ({"reuniao": {"data": "17/03/2018"}}, "reuniao.data"),
        ({"deliberacoes": [{"tema": "x", "decisao": "y", "pagina": 0}]}, "pagina"),
        ({"eleicoes": [{"cargo": "Presidente", "eleito": "", "pagina": 1}]}, "eleito"),
    ],
)
def test_ficha_fora_do_formato_nomeia_o_campo(mudanca: dict[str, Any], campo: str) -> None:
    with pytest.raises(RecordError, match=campo):
        record_from_mapping(bruto(**mudanca))


def test_conferencia_contra_a_pagina() -> None:
    """O que a página não sustenta fica fora: a sede de 2019 não está na p. 3, e 999 delegados não estão na p. 2."""
    registro = confirm(record_from_mapping(bruto()), PAGINAS)
    contas, mesa, sede = registro.deliberacoes
    assert contas.conferido and mesa.conferido and not sede.conferido
    assert registro.eleicoes[0].conferido
    assert [numero.conferido for numero in registro.numeros] == [True, False]
    assert len(problems(registro)) == 2


def test_pagina_inexistente_nao_confere() -> None:
    registro = confirm(
        record_from_mapping(bruto(deliberacoes=[{"tema": "x", "decisao": "contas aprovadas", "pagina": 9}])), PAGINAS
    )
    assert not registro.deliberacoes[0].conferido


def test_ficha_so_com_itens_conferidos() -> None:
    (parte,) = card_parts(confirm(record_from_mapping(bruto()), PAGINAS), DOCUMENTO)
    assert parte.summary_id == "sp-ata-are-2018-ordinaria-1#ficha" and parte.kind == "ficha"
    assert parte.text.startswith("Reunião ordinária de 17/03/2018, Piracicaba/SP.")
    assert "404 delegados credenciados (p. 2)" in parte.text
    assert "Alexandre Banchi (Diretor-Presidente)" in parte.text
    assert "Ribeirão Pires" not in parte.text and "999" not in parte.text
    assert (parte.year, parte.page, parte.source_url) == (2018, 1, "https://exemplo.org/ata-2018.pdf")


def test_linha_do_tempo_sem_procedimento_e_sem_repetir_a_mesma_reuniao() -> None:
    """A segunda via da mesma ata (outro doc_key, mesma reunião) não repete linhas; a mesa diretora fica na ficha."""
    primeira = confirm(record_from_mapping(bruto()), PAGINAS)
    segunda = confirm(record_from_mapping(bruto(doc_key="sp-ata-are-2018-ordinaria-cnpj-1")), PAGINAS)
    (parte,) = timeline_parts([primeira, segunda], _documentos())
    assert parte.summary_id == "linha-do-tempo-assembleias-regional#2018"
    assert parte.title == "Linha do tempo · Assembleias · regional · 2018"
    assert "Mesa Diretora" not in parte.text
    assert parte.text.count("404 delegados credenciados") == 1
    assert "(ata de 17/03/2018, p. 2)" in parte.text
    assert parte.doc_keys == ("sp-ata-are-2018-ordinaria-1",)


def _documentos() -> dict[str, MinutesDocument]:
    import dataclasses

    outro = dataclasses.replace(DOCUMENTO, doc_key="sp-ata-are-2018-ordinaria-cnpj-1")
    return {DOCUMENTO.doc_key: DOCUMENTO, outro.doc_key: outro}


def test_partes_cabem_no_teto() -> None:
    muitas = [
        {
            "tema": f"Tema {i}",
            "decisao": "as contas e balanço do exercício fiscal de 2017 foram aprovadas " * 3,
            "votacao": "",
            "pagina": 2,
        }
        for i in range(20)
    ]
    partes = card_parts(confirm(record_from_mapping(bruto(deliberacoes=muitas)), PAGINAS), DOCUMENTO)
    assert len(partes) > 1 and all(len(p.text) <= 2100 for p in partes)
    assert partes[1].summary_id == "sp-ata-are-2018-ordinaria-1#ficha~2" and "(parte 2 de" in partes[1].title
