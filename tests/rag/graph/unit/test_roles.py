"""O papel de cada tipo de documento (spec 106 FR-005, research R3)."""

from __future__ import annotations

import json
from pathlib import Path

from axon_scouting.rag.graph.api import MODE_OF, ROLES
from axon_scouting.rag.graph.domain.roles import missing_roles

DOCUMENTS = Path(__file__).resolve().parents[4] / "src" / "axon_scouting" / "data" / "index" / "documents.jsonl"


def test_a_tabela_da_r3() -> None:
    assert {kind for kind, role in ROLES.items() if role == "norma"} == {
        "estatuto",
        "por",
        "regulamento",
        "resolucao",
        "politica",
        "lei",
    }
    assert {kind for kind, role in ROLES.items() if role == "orientacao"} == {"manual", "planejamento_estrategico"}
    assert {kind for kind, role in ROLES.items() if role == "registro"} == {
        "ata",
        "parecer",
        "relatorio",
        "manifestacao",
    }
    assert {kind for kind, role in ROLES.items() if role == "comunicacao"} == {
        "oficio",
        "edital",
        "boletim",
        "posicionamento",
        "recomendacao",
        "apresentacao",
    }
    assert MODE_OF == {
        "norma": "disciplina",
        "orientacao": "orienta",
        "registro": "registra",
        "comunicacao": "comunica",
    }


def test_todo_tipo_do_acervo_tem_papel() -> None:
    tipos = {
        json.loads(line)["fields"]["document_type"]
        for line in DOCUMENTS.read_text(encoding="utf-8").split("\n")
        if line
    }
    assert missing_roles(tipos) == []


def test_tipo_sem_papel_e_acusado() -> None:
    assert missing_roles(["estatuto", "carta", "ata", "carta"]) == ["carta"]
