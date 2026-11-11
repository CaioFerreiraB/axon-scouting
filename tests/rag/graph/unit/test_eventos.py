"""Os eventos das fichas das atas (spec 106 G4, US4; research R21)."""

from __future__ import annotations

from typing import Any

from axon_scouting.rag.graph.domain.build.minutes import events_graph

FICHA: dict[str, Any] = {
    "doc_key": "ata-ar-2022",
    "reuniao": {"tipo": "ordinaria", "data": "2022-03-20", "local": "online"},
    "deliberacoes": [
        {
            "tema": "Eleição da Direção Regional gestão 2022-2025",
            "decisao": "Eleita a chapa X.",
            "pagina": 10,
            "conferido": True,
        },
        {"tema": "Alteração do Regulamento Regional", "decisao": "Aprovada a minuta.", "pagina": 7, "conferido": True},
        {"tema": "Item não conferido", "decisao": "Nada.", "pagina": 3, "conferido": False},
    ],
    "eleicoes": [
        {
            "orgao": "Direção Regional",
            "cargo": "Diretor-Presidente",
            "eleito": "Fulano de Tal",
            "mandato": "31/03/2022 a 31/03/2025",
            "pagina": 10,
            "conferido": True,
        },
    ],
    "numeros": [{"o_que": "delegados", "valor": "18", "pagina": 9, "conferido": True}],
}
TRECHOS = [
    {"chunk_id": "ata-ar-2022#p1", "doc_key": "ata-ar-2022", "page_start": 1, "page_end": 1},
    {"chunk_id": "ata-ar-2022#p7", "doc_key": "ata-ar-2022", "page_start": 6, "page_end": 7},
    {"chunk_id": "ata-ar-2022#p9", "doc_key": "ata-ar-2022", "page_start": 9, "page_end": 10},
    {"chunk_id": "ata-ar-2022#p10", "doc_key": "ata-ar-2022", "page_start": 10, "page_end": 10},
]
APELIDOS = {
    "tipo_orgao:diretoria_regional": ("diretoria regional", "direcao regional"),
    "doc:regulamento-regional-sp": ("regulamento regional",),
    "cargo:fulano": ("fulano de tal",),  # um nome de pessoa que, por acaso, fosse apelido
}
GRAFO = events_graph(
    [FICHA], TRECHOS, {"ata-ar-2022": {"date": "2022-03-20"}}, APELIDOS, {"ata-ar-2022": ["orgao:ar_sp"]}
)


def _de(event_id: str) -> list[str]:
    return [e["to"] for e in GRAFO.edges if e["type"] == "DE" and e["from"] == event_id]


def test_um_evento_por_reuniao_e_por_item_conferido() -> None:
    assert [(e["type"], e["page"]) for e in GRAFO.events] == [
        ("reuniao", 1),
        ("eleicao", 10),  # a deliberação cujo tema é uma eleição
        ("deliberacao", 7),
        ("eleicao", 10),
    ]
    assert all(e["date"] == "2022-03-20" for e in GRAFO.events)


def test_o_evento_leva_os_trechos_da_pagina_e_das_vizinhas() -> None:
    assert GRAFO.events[1]["chunk_ids"] == ["ata-ar-2022#p9", "ata-ar-2022#p10"]  # p. 10: a 9 e a 11 são vizinhas
    assert GRAFO.events[2]["chunk_ids"] == ["ata-ar-2022#p7"]  # p. 7: a 6 a 8; o trecho da 9 fica de fora


def test_gera_com_a_fonte_do_item() -> None:
    gera = [e for e in GRAFO.edges if e["type"] == "GERA"]
    assert [(e["from"], e["to"], e["source"]) for e in gera][2] == (
        "ata-ar-2022",
        "evento:ata-ar-2022#2",
        "ata-ar-2022#2@p7",
    )


def test_de_pelo_tema_e_pelo_orgao_da_ata() -> None:
    assert _de("evento:ata-ar-2022#0") == ["orgao:ar_sp"]  # a reunião: só o órgão da ata
    assert _de("evento:ata-ar-2022#1") == ["tipo_orgao:diretoria_regional", "orgao:ar_sp"]
    assert _de("evento:ata-ar-2022#2") == ["doc:regulamento-regional-sp", "orgao:ar_sp"]


def test_pessoa_nunca_vira_entidade() -> None:
    """O nome do eleito fica no texto do evento e fora do reconhecimento."""
    eleicao = GRAFO.events[3]
    assert "Fulano de Tal" in eleicao["text"] and "31/03/2022 a 31/03/2025" in eleicao["text"]
    assert "cargo:fulano" not in _de(eleicao["id"])
    assert _de(eleicao["id"]) == ["tipo_orgao:diretoria_regional", "orgao:ar_sp"]


def test_ficha_de_ata_fora_do_catalogo_fica_de_fora() -> None:
    fora = events_graph([{**FICHA, "doc_key": "outra"}], TRECHOS, {"ata-ar-2022": {}}, APELIDOS, {})
    assert fora.events == [] and fora.dropped
