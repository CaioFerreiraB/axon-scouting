"""A saída de `list_documents` e `list_units` para as perguntas de todos os conjuntos de avaliação, com o artefato do
pacote e o e5 (spec 106 G6, T060). `python -m tests.rag.retrieval.list_reference > arquivo` grava a referência; o teste
`test_list_facades.py` refaz e compara."""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from axon_ai_contracts import RETRIEVAL_FILTERS_KEY
from axon_ai_contracts.testing import pack_context, turn_view

from axon_scouting.pack import ScoutingPack

ROOT = Path(__file__).resolve().parents[3]
E5 = ROOT / ".cache" / "e5"
SETS = ("acervo", "regression", "sintese", "programa-adultos", "golden")


def questions() -> list[tuple[str, str, str]]:
    found: list[tuple[str, str, str]] = []
    for name in SETS:
        data = yaml.safe_load((ROOT / "eval" / f"{name}.yaml").read_text(encoding="utf-8"))
        for item in data.get("items") or []:
            found.append(
                (f"{name}:{item['id']}", item["question"], str(item.get("reference_time") or "2026-11-01T12:00:00"))
            )
    return found


async def outputs() -> dict[str, Any]:
    pack = ScoutingPack()
    ctx = pack_context("scouting", {"embedding_dir": str(E5)})
    tools = {tool.name: tool for tool in pack.retrievers(ctx)}
    found: dict[str, Any] = {}
    for key, question, day in questions():
        when = datetime.fromisoformat(day)
        when = when if when.tzinfo else when.replace(tzinfo=UTC)
        update = await pack.prepare_context(turn_view(question=question, reference_time=when), ctx)
        filters = update.get(RETRIEVAL_FILTERS_KEY) if isinstance(update, dict) else None
        if not isinstance(filters, dict) or not (filters.get("catalog") or filters.get("enumeration")):
            continue
        entry: dict[str, Any] = {}
        for name in ("list_documents", "list_units"):
            chunks = await tools[name].retriever.search(question, top_k=20, filters=filters)
            entry[name] = [[c.chunk_id, c.section, c.text] for c in chunks]
        found[key] = entry
    return found


if __name__ == "__main__":
    json.dump(asyncio.run(outputs()), sys.stdout, ensure_ascii=False, indent=1, sort_keys=True)
