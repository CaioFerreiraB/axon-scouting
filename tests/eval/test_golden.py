"""O conjunto dourado contra o catálogo do corpus e o léxico, só com os contratos (spec 105 US1; contracts/golden-set.md
regras 1–7; research R4, R5; T006).

O core não conhece o corpus: é aqui que se confere que toda evidência nomeia um documento do `corpus/corpus.yaml` e que
o contexto esperado fala a língua da derivação da 101. A regra 8 (o harness lê o arquivo sem item inválido) roda no
grupo `platform` (`tests/integration/test_eval.py`).
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml

from axon_scouting.context.derivation import derive_context
from axon_scouting.context.lexicon import LEXICON
from axon_scouting.context.periods import reference_day
from axon_scouting.eval import CONTEXT_FIELDS, expected_context, is_historical

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "eval" / "golden.yaml"
CATALOG = ROOT / "corpus" / "corpus.yaml"

TOPICS = frozenset(topic.id for topic in LEXICON.topics)
DATES = ("effective_from", "effective_until")
FLAGS = ("include_superseded", "previous_version")


def _load(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _moment(value: object) -> datetime | None:
    """`reference_time` como o harness o lê: texto ISO ou o `datetime` do YAML, com fuso."""
    if isinstance(value, datetime):
        moment = value
    elif isinstance(value, str):
        try:
            moment = datetime.fromisoformat(value)
        except ValueError:
            return None
    else:
        return None
    return moment if moment.utcoffset() is not None else None


def _problems(golden: Mapping[str, Any], catalog: Mapping[str, Any]) -> list[str]:
    """Os problemas das regras 1–7 de contracts/golden-set.md, cada um nomeando o item e o valor."""
    prefixes = {entry["title"]: f"{entry['document_id']}-{entry['version']}#" for entry in catalog["documents"]}
    problems: list[str] = []
    seen: set[str] = set()
    for raw in golden.get("items") or []:
        item_id = raw.get("id")
        tag = f"item {item_id!r}"
        if not isinstance(item_id, str) or not item_id.strip():
            problems.append(f"{tag}: id ausente")
        elif item_id in seen:
            problems.append(f"{tag}: id repetido")
        else:
            seen.add(item_id)

        question = raw.get("question")
        if not isinstance(question, str) or not question.strip():
            problems.append(f"{tag}: question ausente")
        moment = _moment(raw.get("reference_time"))
        if moment is None:
            problems.append(f"{tag}: reference_time precisa ser ISO 8601 com fuso: {raw.get('reference_time')!r}")
        answer = raw.get("reference_answer")
        if not isinstance(answer, str) or not answer.strip():
            problems.append(f"{tag}: reference_answer ausente")

        documents, citations = raw.get("expected_documents") or [], raw.get("expected_citations") or []
        if bool(documents) != bool(citations):
            problems.append(f"{tag}: expected_documents e expected_citations precisam ser ambas vazias ou ambas não")
        for field in ("expected_documents", "expected_citations"):
            for evidence in raw.get(field) or []:
                document, chunk_id = evidence.get("document"), evidence.get("chunk_id")
                prefix = prefixes.get(document)
                if prefix is None:
                    problems.append(f"{tag}: {field}: documento fora do catálogo do corpus: {document!r}")
                elif chunk_id is not None and not str(chunk_id).startswith(prefix):
                    problems.append(f"{tag}: {field}: chunk_id {chunk_id!r} não é de {document!r} ({prefix}…)")

        extra = raw.get("extra") or {}
        if extra.get("provisional") is not True:
            problems.append(f"{tag}: extra.provisional precisa ser true na semente")
        context = extra.get("context")
        if context is not None and not isinstance(context, Mapping):
            problems.append(f"{tag}: extra.context precisa ser um mapeamento")
            continue
        problems.extend(f"{tag}: {problem}" for problem in _context_problems(context or {}))

        derived = derive_context(question, reference_day(moment)) if isinstance(question, str) and moment else None
        historical = is_historical(extra) or (
            derived is not None and (derived.include_superseded or derived.previous_version)
        )
        if historical:
            declared = expected_context(extra)
            missing = [key for key in DATES if key not in declared]
            if not any(key in declared for key in FLAGS):
                missing.insert(0, "include_superseded ou previous_version")
            if missing:
                problems.append(f"{tag}: pergunta histórica sem {', '.join(missing)} em extra.context")
    return problems


def _context_problems(context: Mapping[str, Any]) -> list[str]:
    problems = [f"campo desconhecido em extra.context: {key!r}" for key in context if key not in CONTEXT_FIELDS]
    branch = context.get("branch")
    if "branch" in context and branch is not None and branch not in LEXICON.branches:
        problems.append(f"ramo fora do léxico: {branch!r}")
    for key, known in (("topics", TOPICS), ("document_types", frozenset(LEXICON.document_types))):
        if key in context:
            values = context[key]
            if not isinstance(values, list):
                problems.append(f"{key} precisa ser uma lista: {values!r}")
                continue
            problems.extend(f"{key}: id fora do léxico: {value!r}" for value in values if value not in known)
    for key in DATES:
        if key in context:
            value = context[key]
            if not isinstance(value, str):
                problems.append(f"{key} precisa ser texto ISO entre aspas: {value!r}")
                continue
            try:
                date.fromisoformat(value)
            except ValueError:
                problems.append(f"{key} ilegível: {value!r}")
    problems.extend(
        f"{key} precisa ser booleano: {context[key]!r}"
        for key in FLAGS
        if key in context and not isinstance(context[key], bool)
    )
    return problems


@pytest.fixture(scope="module")
def catalog() -> Mapping[str, Any]:
    return _load(CATALOG)


@pytest.fixture(scope="module")
def semente() -> Mapping[str, Any]:
    return _load(GOLDEN)


# --- a semente -----------------------------------------------------------------------------------------------------


def test_semente_sem_problemas(semente: Mapping[str, Any], catalog: Mapping[str, Any]) -> None:
    assert _problems(semente, catalog) == []


def test_semente_no_formato_da_009(semente: Mapping[str, Any]) -> None:
    assert semente["version"] == 1
    assert semente["name"] == "scouting-semente"
    assert semente["assistant"] == "scouting"


def test_semente_tem_entre_8_e_12_itens(semente: Mapping[str, Any]) -> None:
    assert 8 <= len(semente["items"]) <= 12


COBERTOS = frozenset({"por", "estatuto", "resolucao", "manual"})
"""Os tipos que a semente cobre (FR-002: fontes normativas). Com o catálogo de 593 documentos, cobrir todos os títulos
deixou de fazer sentido (research R14)."""


def test_semente_cobre_os_tipos_normativos_um_historico_e_um_fora_do_corpus(
    semente: Mapping[str, Any], catalog: Mapping[str, Any]
) -> None:
    items = semente["items"]
    type_of = {entry["title"]: entry["type"] for entry in catalog["documents"]}
    cited = {type_of.get(evidence["document"]) for item in items for evidence in item.get("expected_documents") or []}
    assert cited >= COBERTOS
    assert any(is_historical(item.get("extra") or {}) for item in items), "FR-002: ao menos uma consulta histórica"
    assert any(not item.get("expected_documents") for item in items), "ao menos uma pergunta fora do corpus"


# --- as regras, uma a uma ------------------------------------------------------------------------------------------


def _item(**changes: Any) -> dict[str, Any]:
    item: dict[str, Any] = {
        "id": "x1",
        "question": "Quem pode ser patrono de um grupo escoteiro?",
        "reference_time": "2026-10-13T12:00:00-03:00",
        "expected_documents": [
            {"document": "Princípios, Organização e Regras (POR)", "chunk_id": "por-2025.2.1#regra31@p37"}
        ],
        "reference_answer": "Uma pessoa já falecida, de conduta exemplar.",
        "expected_citations": [{"document": "Princípios, Organização e Regras (POR)"}],
        "extra": {"provisional": True, "context": {"branch": None}},
    }
    item.update(changes)
    return item


def _golden(*items: dict[str, Any]) -> dict[str, Any]:
    return {"version": 1, "name": "teste", "items": list(items)}


def test_item_valido_nao_tem_problema(catalog: Mapping[str, Any]) -> None:
    assert _problems(_golden(_item()), catalog) == []


def test_documento_fora_do_catalogo_nomeia_item_e_documento(catalog: Mapping[str, Any]) -> None:
    item = _item(expected_documents=[{"document": "Regulamento Inventado"}])
    (problem,) = _problems(_golden(item), catalog)
    assert "'x1'" in problem and "'Regulamento Inventado'" in problem


def test_chunk_id_de_outro_documento(catalog: Mapping[str, Any]) -> None:
    item = _item(
        expected_documents=[{"document": "Princípios, Organização e Regras (POR)", "chunk_id": "estatuto-2024#art1@p3"}]
    )
    (problem,) = _problems(_golden(item), catalog)
    assert "estatuto-2024#art1@p3" in problem


@pytest.mark.parametrize(
    ("context", "expected"),
    [
        ({"ramo": "lobinho"}, "campo desconhecido"),
        ({"topics": ["culinaria"]}, "fora do léxico: 'culinaria'"),
        ({"branch": "castor"}, "ramo fora do léxico"),
        ({"document_types": "por"}, "precisa ser uma lista"),
        ({"effective_from": date(2026, 10, 13)}, "texto ISO entre aspas"),
        ({"effective_until": "13/10/2026"}, "ilegível"),
        ({"previous_version": "sim"}, "booleano"),
    ],
)
def test_extra_context_invalido(catalog: Mapping[str, Any], context: dict[str, Any], expected: str) -> None:
    (problem,) = _problems(_golden(_item(extra={"provisional": True, "context": context})), catalog)
    assert expected in problem


def test_item_historico_declarado_sem_effective_until(catalog: Mapping[str, Any]) -> None:
    context = {"include_superseded": True, "effective_from": "2022-01-01"}
    (problem,) = _problems(_golden(_item(extra={"provisional": True, "context": context})), catalog)
    assert "effective_until" in problem


def test_pergunta_historica_pela_derivacao_sem_extra_context(catalog: Mapping[str, Any]) -> None:
    item = _item(question="Qual era a regra de uniforme em 2022?", extra={"provisional": True})
    (problem,) = _problems(_golden(item), catalog)
    assert "pergunta histórica" in problem and "include_superseded ou previous_version" in problem


def test_pergunta_historica_completa(catalog: Mapping[str, Any]) -> None:
    context = {"include_superseded": True, "effective_from": "2022-01-01", "effective_until": "2022-12-31"}
    item = _item(question="Qual era a regra de uniforme em 2022?", extra={"provisional": True, "context": context})
    assert _problems(_golden(item), catalog) == []


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"reference_time": "2026-10-13T12:00:00"}, "fuso"),
        ({"reference_answer": None}, "reference_answer"),
        ({"expected_citations": []}, "ambas vazias"),
        ({"extra": {"context": {"branch": None}}}, "provisional"),
        ({"question": " "}, "question"),
    ],
)
def test_formato_minimo(catalog: Mapping[str, Any], changes: dict[str, Any], expected: str) -> None:
    (problem,) = _problems(_golden(_item(**changes)), catalog)
    assert expected in problem


def test_pergunta_fora_do_corpus_sem_evidencia_e_valida(catalog: Mapping[str, Any]) -> None:
    item = _item(expected_documents=[], expected_citations=[], reference_answer="Nenhum documento oficial.")
    assert _problems(_golden(item), catalog) == []


def test_id_repetido(catalog: Mapping[str, Any]) -> None:
    (problem,) = _problems(_golden(_item(), copy.deepcopy(_item())), catalog)
    assert "repetido" in problem
