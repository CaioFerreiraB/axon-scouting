"""`ArtifactGraph`: o grafo das normas por órgão gravado ao lado do índice (ADR-063).

`graph.jsonl` fica no diretório do artefato: uma linha `{"kind": "organ", ...}` por tipo de órgão ou instituição, com
os termos que o reconhecem, e uma `{"kind": "section", ...}` por seção que o disciplina. Escrito por
`scripts/grafo.py build` a partir de `corpus/organizacao.yaml` e da árvore dos trechos; não entra no manifesto, e o
arquivo ausente vale como grafo vazio.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

from axon_scouting.rag.index.api import NormSection

GRAPH: Final = "graph.jsonl"


class ArtifactGraph:
    """Implementa `NormGraph` sobre `<root>/graph.jsonl`, lido uma vez."""

    def __init__(self, root: Path) -> None:
        self._path = root / GRAPH
        self._loaded: tuple[dict[str, tuple[str, ...]], dict[str, list[NormSection]]] | None = None
        self._lock = threading.Lock()

    async def organ_terms(self) -> Mapping[str, tuple[str, ...]]:
        return dict(self._load()[0])

    async def sections(self, organs: Sequence[str]) -> list[NormSection]:
        by_organ = self._load()[1]
        return [section for organ in organs for section in by_organ.get(organ, [])]

    def _load(self) -> tuple[dict[str, tuple[str, ...]], dict[str, list[NormSection]]]:
        with self._lock:
            if self._loaded is None:
                terms: dict[str, tuple[str, ...]] = {}
                by_organ: dict[str, list[NormSection]] = {}
                if self._path.exists():
                    with self._path.open(encoding="utf-8") as lines:
                        for line in lines:
                            if not line.strip():
                                continue
                            raw = json.loads(line)
                            if raw.get("kind") == "organ":
                                terms[str(raw["id"])] = tuple(str(t) for t in raw.get("terms") or ())
                            elif raw.get("kind") == "section":
                                section = NormSection.from_json(raw)
                                by_organ.setdefault(section.organ, []).append(section)
                self._loaded = (terms, by_organ)
            return self._loaded


def write_graph(path: Path, organs: Mapping[str, tuple[str, ...]], sections: Sequence[NormSection]) -> None:
    with path.open("w", encoding="utf-8") as out:
        for organ, terms in organs.items():
            out.write(json.dumps({"kind": "organ", "id": organ, "terms": list(terms)}, ensure_ascii=False) + "\n")
        for section in sections:
            out.write(json.dumps({"kind": "section", **section.to_json()}, ensure_ascii=False) + "\n")
