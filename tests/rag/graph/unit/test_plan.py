"""O plano de consultas (spec 106 FR-011; contracts/query-plan.schema.json)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from axon_scouting.rag.graph.api import PlanInvalid, QueryPlan, QueryStep

SCHEMA = json.loads((Path(__file__).resolve().parents[1] / "query-plan.schema.json").read_text(encoding="utf-8"))

VALIDOS: list[dict[str, Any]] = [
    {"version": 1, "consultas": []},
    {
        "version": 1,
        "consultas": [{"q": "sobre", "args": {"entidades": ["tipo_orgao:diretoria_regional"], "papeis": ["norma"]}}],
    },
    {
        "version": 1,
        "forma": "intersecao",
        "consultas": [
            {"q": "sobre", "args": {"entidades": ["ramo:pioneiro", "cargo:escotista"], "modo": "todas"}},
            {"q": "aplicavel_a", "args": {"instituicao": "inst:ueb"}},
        ],
    },
    {
        "version": 1,
        "forma": "para_cada",
        "para_cada": ["ramo:lobinho", "ramo:escoteiro"],
        "consultas": [{"q": "sobre", "args": {"entidades": ["$cada", "cargo:chefe_de_secao"]}}],
    },
]


@pytest.mark.parametrize("raw", VALIDOS)
def test_ida_e_volta_e_valido_pelo_esquema(raw: dict[str, Any]) -> None:
    jsonschema = pytest.importorskip("jsonschema")
    plano = QueryPlan.from_json(raw)
    assert QueryPlan.from_json(plano.to_json()) == plano
    jsonschema.validate(plano.to_json(), SCHEMA)
    jsonschema.validate(raw, SCHEMA)


@pytest.mark.parametrize(
    ("raw", "campo"),
    [
        ({"version": 2, "consultas": []}, "version"),
        ({"version": 1, "consultas": [{"q": "cypher", "args": {}}]}, "consultas[0].q"),
        (
            {"version": 1, "consultas": [{"q": "sobre", "args": {"entidades": ["ramo:x"], "x": 1}}]},
            "consultas[0].args.x",
        ),
        ({"version": 1, "consultas": [{"q": "sobre", "args": {}}]}, "consultas[0].args.entidades"),
        (
            {"version": 1, "consultas": [{"q": "sobre", "args": {"entidades": ["diretoria"]}}]},
            "consultas[0].args.entidades",
        ),
        (
            {"version": 1, "consultas": [{"q": "sobre", "args": {"entidades": ["ramo:x"], "papeis": ["lei"]}}]},
            "consultas[0].args.papeis",
        ),
        ({"version": 1, "consultas": [], "extra": 1}, "extra"),
        ({"version": 1, "consultas": [{"q": "sobre", "args": {"entidades": ["ramo:x"]}}] * 5}, "consultas"),
        ({"version": 1, "para_cada": ["ramo:x"], "consultas": []}, "para_cada"),
        (
            {"version": 1, "forma": "para_cada", "para_cada": [f"ramo:r{i}" for i in range(7)], "consultas": []},
            "para_cada",
        ),
    ],
)
def test_plano_invalido_nomeia_o_campo(raw: dict[str, Any], campo: str) -> None:
    with pytest.raises(PlanInvalid) as erro:
        QueryPlan.from_json(raw)
    assert erro.value.field == campo


def test_plano_vazio() -> None:
    assert QueryPlan().empty
    assert not QueryPlan(consultas=(QueryStep(q="sobre", args={"entidades": ["ramo:lobinho"]}),)).empty
