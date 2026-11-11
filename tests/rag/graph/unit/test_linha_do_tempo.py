"""A consulta `linha_do_tempo` (spec 106 G4, US4; research R21)."""

from __future__ import annotations

from typing import Any

from axon_scouting.rag.graph.domain.model import from_records
from axon_scouting.rag.graph.domain.queries import linha_do_tempo


def _records() -> list[dict[str, Any]]:
    def entity(id_: str) -> dict[str, Any]:
        return {
            "kind": "entity",
            "id": id_,
            "type": id_.split(":")[0],
            "name": id_,
            "terms": [],
            "source": "x",
            "origin": "curado",
        }

    def version(doc_key: str) -> dict[str, Any]:
        return {
            "kind": "version",
            "doc_key": doc_key,
            "document_id": doc_key,
            "type": "ata",
            "role": "registro",
            "status": "vigente",
            "scope": "regional",
        }

    def event(ident: str, doc_key: str, when: str | None, kind: str) -> dict[str, Any]:
        return {
            "kind": "event",
            "id": ident,
            "doc_key": doc_key,
            "date": when,
            "type": kind,
            "text": ident,
            "page": 2,
            "chunk_ids": [f"{doc_key}#p2"],
        }

    def edge(kind: str, from_: str, to: str) -> dict[str, Any]:
        return {"kind": "edge", "type": kind, "from": from_, "to": to, "source": "x", "origin": "regra"}

    return [
        {"kind": "meta", "version": 2, "chunking": "x"},
        *(entity(e) for e in ("tipo_orgao:diretoria_regional", "orgao:dr_sp", "orgao:ar_sp")),
        *(version(d) for d in ("ata-2016", "ata-2019", "ata-2022", "ata-sem-data")),
        event("evento:ata-2022#1", "ata-2022", "2022-03-20", "eleicao"),
        event("evento:ata-2016#1", "ata-2016", "2016-03-12", "eleicao"),
        event("evento:ata-2019#3", "ata-2019", "2019-03-16", "deliberacao"),
        event("evento:ata-sem-data#1", "ata-sem-data", None, "eleicao"),
        edge("INSTANCIA_DE", "orgao:dr_sp", "tipo_orgao:diretoria_regional"),
        edge("DE", "evento:ata-2022#1", "tipo_orgao:diretoria_regional"),
        edge("DE", "evento:ata-2016#1", "orgao:dr_sp"),  # pela instância
        edge("DE", "evento:ata-2019#3", "tipo_orgao:diretoria_regional"),
        edge("DE", "evento:ata-sem-data#1", "tipo_orgao:diretoria_regional"),
        edge("DE", "evento:ata-2022#1", "orgao:ar_sp"),
    ]


G = from_records(_records())


def _ids(**args: Any) -> list[str]:
    return [e.event.text for e in linha_do_tempo(G, "tipo_orgao:diretoria_regional", **args) if e.event]


def test_por_data_com_as_instancias_e_os_sem_data_por_ultimo() -> None:
    assert _ids() == ["evento:ata-2016#1", "evento:ata-2019#3", "evento:ata-2022#1", "evento:ata-sem-data#1"]


def test_filtra_pelo_tipo_do_evento() -> None:
    assert _ids(eventos=["eleicao"]) == ["evento:ata-2016#1", "evento:ata-2022#1", "evento:ata-sem-data#1"]


def test_filtra_pelo_periodo_e_tira_os_sem_data() -> None:
    assert _ids(periodo={"de": "2017-01-01", "ate": "2022-12-31"}) == ["evento:ata-2019#3", "evento:ata-2022#1"]
    assert _ids(periodo={"de": "2020-01-01"}) == ["evento:ata-2022#1"]


def test_a_evidencia_traz_a_ata_os_trechos_e_o_evento() -> None:
    (first, *_) = linha_do_tempo(G, "orgao:ar_sp")
    assert (first.doc_key, first.chunk_ids, first.role, first.edge.type) == (
        "ata-2022",
        ("ata-2022#p2",),
        "registro",
        "DE",
    )
    assert first.event is not None and first.event.type == "eleicao" and first.event.page == 2


def test_instancia_nao_traz_os_eventos_do_tipo() -> None:
    assert [e.doc_key for e in linha_do_tempo(G, "orgao:dr_sp")] == ["ata-2016"]
