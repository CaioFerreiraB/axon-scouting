"""O conjunto de regressão da recuperação, só com os contratos (spec 105 US6, FR-015; research R15).

Confere o formato que o harness da 009 lê e as regras do domínio: todo item declara fatos em `extra.facts`, e todo fato
existe no texto do artefato do índice — um fato que nenhum trecho contém tornaria o item impossível, e a métrica
mediria o conjunto, não a recuperação.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
import yaml

from axon_scouting.context.text import contains_term, normalize
from axon_scouting.eval import FACTS_KEY

ROOT = Path(__file__).resolve().parents[2]
REGRESSION = ROOT / "eval" / "regression.yaml"
CHUNKS = ROOT / "src" / "axon_scouting" / "data" / "index" / "chunks.jsonl"
GROUPS = frozenset({"individual", "agregado", "comparativo", "normativo"})


def _problems(regression: Mapping[str, Any], texts: list[str]) -> list[str]:
    problems: list[str] = []
    if regression.get("version") != 1 or regression.get("assistant") != "scouting":
        problems.append("cabeçalho: version 1 e assistant scouting")
    seen: set[str] = set()
    for raw in regression.get("items") or []:
        item_id = raw.get("id")
        tag = f"item {item_id!r}"
        if not isinstance(item_id, str) or item_id in seen:
            problems.append(f"{tag}: id ausente ou repetido")
        seen.add(str(item_id))
        if not isinstance(raw.get("question"), str) or not raw["question"].strip():
            problems.append(f"{tag}: question ausente")
        try:
            moment = datetime.fromisoformat(str(raw.get("reference_time")))
        except ValueError:
            moment = None
        if moment is None or moment.utcoffset() is None:
            problems.append(f"{tag}: reference_time precisa ser ISO 8601 com fuso")
        extra = raw.get("extra") or {}
        if extra.get("group") not in GROUPS:
            problems.append(f"{tag}: extra.group fora de {sorted(GROUPS)}: {extra.get('group')!r}")
        facts = extra.get(FACTS_KEY)
        if not isinstance(facts, list) or not facts:
            problems.append(f"{tag}: extra.facts precisa ser uma lista não vazia")
            continue
        for fact in facts:
            alternatives = [fact] if isinstance(fact, str) else fact
            if not isinstance(alternatives, list) or not all(isinstance(a, str) and normalize(a) for a in alternatives):
                problems.append(f"{tag}: fato precisa ser texto ou lista de textos: {fact!r}")
                continue
            if not any(contains_term(text, normalize(a)) for a in alternatives for text in texts):
                problems.append(f"{tag}: fato que nenhum trecho do artefato contém: {fact!r}")
    return problems


@pytest.fixture(scope="module")
def regression() -> Mapping[str, Any]:
    loaded: Mapping[str, Any] = yaml.safe_load(REGRESSION.read_text(encoding="utf-8"))
    return loaded


@pytest.fixture(scope="module")
def texts() -> list[str]:
    return [normalize(json.loads(line)["text"]) for line in CHUNKS.read_text(encoding="utf-8").split("\n") if line]


def test_regressao_sem_problemas(regression: Mapping[str, Any], texts: list[str]) -> None:
    assert _problems(regression, texts) == []


def test_regressao_cobre_os_padroes(regression: Mapping[str, Any]) -> None:
    """Os padrões da revisão: o mesmo conceito por ramo, agregado, comparação e regra normativa (FR-015)."""
    groups = {item["extra"]["group"] for item in regression["items"]}
    assert groups == GROUPS
    agregados = [item for item in regression["items"] if item["extra"]["group"] == "agregado"]
    assert len({len(item["extra"][FACTS_KEY]) for item in agregados}) == 1, "agregados cobrem os cinco ramos"


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        ({"extra": {"group": "individual", "facts": ["frase que não existe em trecho nenhum"]}}, "nenhum trecho"),
        ({"extra": {"group": "individual", "facts": []}}, "lista não vazia"),
        ({"extra": {"group": "outro", "facts": ["ser livre como os lobos"]}}, "extra.group"),
        ({"reference_time": "2026-10-27T12:00:00"}, "fuso"),
    ],
    ids=["fato-inexistente", "sem-fatos", "grupo-desconhecido", "sem-fuso"],
)
def test_problemas_nomeiam_o_item(change: dict[str, Any], expected: str) -> None:
    item = {
        "id": "x",
        "question": "Qual o marco simbólico do Ramo Lobinho?",
        "reference_time": "2026-10-27T12:00:00-03:00",
        "extra": {"group": "individual", "facts": ["ser livre como os lobos"]},
    } | change
    regression = {"version": 1, "assistant": "scouting", "items": [item]}
    problems = _problems(regression, ["o marco e ser livre como os lobos"])
    assert len(problems) == 1 and "item 'x'" in problems[0] and expected in problems[0]
