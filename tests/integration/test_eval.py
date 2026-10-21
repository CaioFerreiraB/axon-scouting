"""O harness da 009 com o pack scouting, pelo comando `axon eval run` (spec 105 US1, US2; contracts/golden-set.md
regra 8 e contracts/domain-metrics.md verificação 4; research R5, R8; T012).

Grupo opcional `platform`. Usa só a superfície que a 011 publica: as variáveis `AXON_*` e o comando `axon` do
próprio ambiente, num processo por rodada — que solta a trava do Qdrant embutido ao sair. O corpus sintético da 103 é
publicado com o embedder de teste; a geração é o provider simulado do perfil `test` (resposta fixa, sem citação).
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

pytest.importorskip("axon_ai", reason="axon-ai não instalado: rode com o grupo platform (Python 3.14)")

from qdrant_client import AsyncQdrantClient
from rag.retrieval.corpus import publicar

from axon_scouting.rag.index.adapters.bm25 import Bm25Encoder
from axon_scouting.rag.index.adapters.memory import HashingEmbedder
from axon_scouting.rag.index.adapters.qdrant import QdrantIndex

ROOT = Path(__file__).resolve().parents[2]
AXON = Path(sys.executable).with_name("axon")
METRICS = ("scouting:context_accuracy", "scouting:current_document_rate", "scouting:validity_notice_rate")

SINTETICO = """
version: 1
name: sintetico
assistant: scouting
items:
  - id: a1
    question: "O que diz a Regra 33 do POR sobre o nome do grupo?"
    reference_time: "2026-10-13T12:00:00-03:00"
    expected_documents: [{document: "POR", chunk_id: "por-2025#regra33@p38"}]
    reference_answer: "O nome do grupo é escolhido."
    expected_citations: [{document: "POR"}]
    extra: {provisional: true, context: {branch: null}}
  - id: a2
    question: "Como é o acampamento da tropa?"
    reference_time: "2026-10-13T12:00:00-03:00"
    expected_documents: [{document: "Manual X"}]
    reference_answer: "O acampamento da tropa exige cardápio."
    expected_citations: [{document: "Manual X"}]
    extra: {provisional: true, context: {topics: [acampamento]}}
"""


@pytest.fixture(scope="module")
def ambiente(tmp_path_factory: pytest.TempPathFactory) -> dict[str, str]:
    base = tmp_path_factory.mktemp("eval")
    qdrant = base / "qdrant"

    async def publish() -> None:
        client = AsyncQdrantClient(path=str(qdrant))
        try:
            embedder, sparse = HashingEmbedder(), Bm25Encoder()
            await publicar(QdrantIndex(client, "scouting", embedder, sparse), embedder, sparse)
        finally:
            await client.close()  # o modo embutido trava o diretório: o comando abre depois

    asyncio.run(publish())
    packs = base / "packs.yaml"
    packs.write_text(
        f"packs:\n  scouting:\n    enabled: true\n    config:\n      store: qdrant\n      qdrant_path: {qdrant}\n"
        "      embedder: hashing\n",
        encoding="utf-8",
    )
    env = {
        name: value for name, value in os.environ.items() if not name.startswith(("AXON_", "AI_CORE_", "AI_SCOUTING_"))
    }
    env |= {
        "AXON_PROFILE": "test",
        "AXON_PACKS_FILE": str(packs),
        "AXON_ASSISTANTS_FILE": str(ROOT / "eval" / "assistants.yaml"),
    }
    return env


def _eval_run(env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(AXON), "eval", "run", *args], env=env, cwd=ROOT, capture_output=True, text=True, timeout=300
    )


def _report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(report: dict[str, Any], name: str) -> dict[str, Any]:
    (found,) = [metric for metric in report["metrics"] if metric["name"] == name]
    return found


def test_metricas_de_dominio_no_relatorio(ambiente: dict[str, str], tmp_path: Path) -> None:
    golden, report = tmp_path / "golden.yaml", tmp_path / "report.json"
    golden.write_text(SINTETICO, encoding="utf-8")

    result = _eval_run(ambiente, "--golden", str(golden), "--report", str(report))

    assert result.returncode == 0, result.stdout + result.stderr
    data = _report(report)
    assert data["invalid"] == []
    for name in METRICS:
        assert name in result.stdout
    accuracy, current, notice = (_metric(data, name) for name in METRICS)
    assert (accuracy["value"], accuracy["n"], accuracy["deterministic"]) == (1.0, 2, True)
    assert (current["value"], current["n"], current["deterministic"]) == (1.0, 2, True)
    assert (notice["value"], notice["n"], notice["deterministic"]) == (None, 0, False), "sem citação, não se aplica"


def test_semente_roda_sem_item_invalido(ambiente: dict[str, str], tmp_path: Path) -> None:
    report = tmp_path / "report.json"

    result = _eval_run(ambiente, "--golden", "eval/golden.yaml", "--report", str(report))

    assert result.returncode == 0, result.stdout + result.stderr
    data = _report(report)
    assert data["golden_set"] == "scouting-semente"
    assert data["invalid"] == []
    semente = yaml.safe_load((ROOT / "eval" / "golden.yaml").read_text(encoding="utf-8"))
    assert [item["id"] for item in data["items"]] == [item["id"] for item in semente["items"]]
    assert all(item["failure"] is None for item in data["items"]), data["items"]


def test_segunda_rodada_compara_com_a_linha_de_base(ambiente: dict[str, str], tmp_path: Path) -> None:
    golden, base = tmp_path / "golden.yaml", tmp_path / "base.json"
    golden.write_text(SINTETICO, encoding="utf-8")
    first = _eval_run(ambiente, "--golden", str(golden), "--baseline", str(base), "--update-baseline")
    assert first.returncode == 0, first.stdout + first.stderr
    assert base.exists()

    second = _eval_run(ambiente, "--golden", str(golden), "--baseline", str(base))

    assert second.returncode == 0, second.stdout + second.stderr
    assert "REGRESSÃO" not in second.stdout
    assert "scouting:current_document_rate" in second.stdout
