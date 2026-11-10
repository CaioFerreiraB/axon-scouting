"""O export para o arrows.app (spec 106 T033): a forma do arquivo do desenho do mantenedor."""

from __future__ import annotations

import json
from pathlib import Path

from axon_scouting.rag.graph.adapters.arrows import neighbourhood, to_arrows
from axon_scouting.rag.graph.domain.model import from_records

from ..grafo_sintetico import registros

DESENHO = Path(__file__).resolve().parents[4] / "corpus" / "Organização Escoteira.revisado.json"


def test_a_forma_do_arrows() -> None:
    g = from_records(registros())
    data = to_arrows(g, neighbourhood(g, "tipo_orgao:diretoria_regional", 1))
    desenho = json.loads(DESENHO.read_text(encoding="utf-8"))
    assert set(data) == set(desenho)
    assert set(data["nodes"][0]) == set(desenho["nodes"][0])
    assert set(data["relationships"][0]) == set(desenho["relationships"][0])
    ids = {n["id"] for n in data["nodes"]}
    assert all(r["fromId"] in ids and r["toId"] in ids for r in data["relationships"])
    assert any(r["type"] == "ELEGE" and r["properties"]["fonte"] for r in data["relationships"])


def test_sem_entidade_so_as_entidades() -> None:
    g = from_records(registros())
    assert neighbourhood(g, None, 1) == set(g.entities)
    assert neighbourhood(g, "tipo_orgao:diretoria_regional", 1, units=False) <= set(g.entities)
