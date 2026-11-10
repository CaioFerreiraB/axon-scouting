"""`ArtifactGraphStore` e o escritor do `graph.jsonl` v2 (spec 106, research R10; contracts/graph-jsonl-v2.md).

O arquivo fica ao lado do índice, fora do manifesto. Ausente, o grafo é vazio (como no ADR-063); de outro build (o
`chunking` do `meta` difere do `manifest.json` do mesmo diretório), a carga levanta `GraphUnavailable`.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Final

from axon_scouting.rag.graph.api import GraphUnavailable
from axon_scouting.rag.graph.ports import GraphRecord

GRAPH: Final = "graph.jsonl"
MANIFEST: Final = "manifest.json"
KIND_ORDER: Final = ("meta", "entity", "version", "unit", "edge", "event")


class ArtifactGraphStore:
    def __init__(self, root: Path) -> None:
        self._root = root

    def records(self) -> Iterable[GraphRecord]:
        path = self._root / GRAPH
        if not path.exists():
            return []
        with path.open(encoding="utf-8") as lines:
            records = [json.loads(line) for line in lines if line.strip()]
        meta = next((r for r in records if r.get("kind") == "meta"), None)
        manifest = self._root / MANIFEST
        if meta is None:
            raise GraphUnavailable(f"{path}: sem registro meta (grafo v1?) — rode `scripts/grafo.py build`")
        if manifest.exists():
            chunking = json.loads(manifest.read_text(encoding="utf-8")).get("chunking")
            if meta.get("chunking") != chunking:
                raise GraphUnavailable(
                    f"{path}: o grafo é do chunking {meta.get('chunking')!r}, e o índice é {chunking!r} — rode "
                    "`scripts/grafo.py build`"
                )
        return records


def write_graph_v2(path: Path, records: Sequence[GraphRecord]) -> None:
    """Grava os registros na ordem do contrato (`meta`, `entity`, `version`, `unit`, `edge`, `event`), mantendo a ordem
    de cada bloco como veio (as entidades na ordem do catálogo; o resto já vem ordenado pelo build)."""
    ordered = sorted(records, key=lambda record: KIND_ORDER.index(str(record["kind"])))
    with path.open("w", encoding="utf-8") as out:
        for record in ordered:
            out.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
