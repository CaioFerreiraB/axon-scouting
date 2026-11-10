"""`ArtifactSummaries`: os resumos das atas gravados ao lado do índice (ADR-062).

`summaries.jsonl` fica no diretório do artefato, um `SummaryEntry` por linha, escrito por `scripts/fichas.py build` a
partir das fichas conferidas de `corpus/resumos/atas/`. Não tem vetores nem entra no manifesto: os resumos não passam
pela busca densa, e o arquivo ausente vale como nenhum resumo. Vale nos dois armazenamentos (memória e Qdrant): o
artefato vem no pacote nos dois casos.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Final

from axon_scouting.rag.index.api import SummaryEntry

SUMMARIES: Final = "summaries.jsonl"


class ArtifactSummaries:
    """Implementa `MeetingSummaries` sobre `<root>/summaries.jsonl`, lido uma vez."""

    def __init__(self, root: Path) -> None:
        self._path = root / SUMMARIES
        self._entries: list[SummaryEntry] | None = None
        self._lock = threading.Lock()

    async def summaries(self) -> list[SummaryEntry]:
        return list(self._load())

    def _load(self) -> list[SummaryEntry]:
        with self._lock:
            if self._entries is None:
                self._entries = read_summaries(self._path)
            return self._entries


def read_summaries(path: Path) -> list[SummaryEntry]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as lines:
        return [SummaryEntry.from_json(json.loads(line)) for line in lines if line.strip()]


def write_summaries(path: Path, entries: list[SummaryEntry]) -> None:
    with path.open("w", encoding="utf-8") as out:
        for entry in entries:
            out.write(json.dumps(entry.to_json(), ensure_ascii=False, sort_keys=True) + "\n")


def summary_texts(root: Path) -> dict[str, str]:
    """O texto de cada resumo por id, como a busca o entrega — para a métrica de fatos da 105 ler o que chegou."""
    return {entry.summary_id: f"[{entry.title}]\n{entry.text}" for entry in read_summaries(root / SUMMARIES)}
