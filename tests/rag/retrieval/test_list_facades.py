"""`list_documents` e `list_units` como fachadas de `documentos` e `itens` do grafo de domínio (spec 106 G6, US6;
research R23): o grafo decide quem pode entrar, e a saída não muda."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from axon_ai_contracts import JSONValue

from axon_scouting.rag.graph.api import Aliases, EdgeRef, Evidence, Frentes, QueryName, QueryPlan
from axon_scouting.rag.retrieval.catalog import CatalogRetriever
from axon_scouting.rag.retrieval.units import UnitsRetriever

from . import test_units
from .corpus import indice
from .test_catalog import pedido

ROOT = Path(__file__).resolve().parents[3]
REFERENCE = Path(__file__).parent / "data" / "list_facades_reference.json"


class FakeGraph:
    """Um grafo que responde `documentos` e `itens` com os ids dados, e registra as consultas."""

    def __init__(self, *, docs: Sequence[str] = (), units: Sequence[str] = ()) -> None:
        self.docs, self.units = list(docs), list(units)
        self.asked: list[tuple[str, dict[str, Any]]] = []

    async def aliases(self) -> Aliases:
        return {}

    async def run(self, plan: QueryPlan) -> Frentes:
        return ()

    async def query(self, name: QueryName, /, **args: JSONValue) -> list[Evidence]:
        self.asked.append((name, dict(args)))
        ids = self.docs if name == "documentos" else self.units
        return [_evidence(name, item) for item in ids]


def _evidence(name: QueryName, item: str) -> Evidence:
    return Evidence(
        query=name,
        target="x",
        doc_key=item.split("#")[0],
        unit_id=item if "#" in item else None,
        chunk_ids=(),
        role="norma",
        scope=None,
        date=None,
        edge=EdgeRef(type="CONTEM", source=item, origin="regra"),
    )


def _catalog(graph: FakeGraph | None, filters: dict[str, object]) -> list[str]:
    async def go() -> list[str]:
        retriever = CatalogRetriever(await indice(), graph=graph)  # type: ignore[arg-type]
        return [chunk.chunk_id for chunk in await retriever.search("lista", top_k=20, filters=filters)]

    return asyncio.run(go())


def _units(graph: FakeGraph | None) -> list[str]:
    async def go() -> list[str]:
        index = await test_units.indice()
        filters = test_units.filtros(enumeration={"unit_kind": "especialidade", "terms": []})
        found = await UnitsRetriever(index, index, graph=graph).search(  # type: ignore[arg-type]
            "quais são as especialidades", top_k=20, filters=filters
        )
        return [chunk.text for chunk in found]

    return asyncio.run(go())


def test_o_grafo_decide_quais_documentos_entram_na_lista() -> None:
    filters = pedido("documento")
    today = _catalog(None, filters)
    assert len(today) >= 2
    keys = [chunk_id.removesuffix("#acervo") for chunk_id in today]
    assert _catalog(FakeGraph(docs=keys), filters) == today  # o grafo com os mesmos documentos: a mesma saída
    graph = FakeGraph(docs=keys[:1])
    assert _catalog(graph, filters) == today[:1]
    assert graph.asked == [("documentos", {"vigente": False})]  # "documento" = qualquer tipo


def test_o_tipo_pedido_vai_ao_grafo() -> None:
    graph = FakeGraph(docs=["outro-1"])
    assert _catalog(graph, pedido("politica")) == []
    assert graph.asked[0] == ("documentos", {"vigente": False, "tipo": "politica"})


def test_sem_grafo_a_lista_de_hoje() -> None:
    """Um índice sem `graph.jsonl` ao lado (a suíte de integração, um Qdrant de fora): o grafo vazio não esvazia as
    listas."""
    filters = pedido("documento")
    assert _catalog(FakeGraph(), filters) == _catalog(None, filters)
    assert _units(FakeGraph()) == _units(None)


def test_o_grafo_decide_quais_unidades_entram_na_ficha() -> None:
    today = _units(None)
    ecoturismo = f"{test_units.EIXO.unit_id}/especialidade-ecoturismo"
    sustentabilidade = f"{test_units.EIXO.unit_id}/especialidade-sustentabilidade"
    assert _units(FakeGraph(units=[ecoturismo, sustentabilidade])) == today
    (card,) = _units(FakeGraph(units=[ecoturismo]))
    assert "Ecoturismo" in card and "Sustentabilidade" not in card


@pytest.mark.skipif(not (ROOT / ".cache" / "e5" / "model.onnx").exists(), reason="sem o e5 em .cache/e5")
def test_a_saida_das_listas_e_a_de_antes_do_g6() -> None:
    """T060: as perguntas de lista dos cinco conjuntos, com o artefato do pacote e o e5, dão a mesma saída que antes da
    fachada (a referência foi gravada antes da mudança)."""
    from .list_reference import outputs

    assert asyncio.run(outputs()) == json.loads(REFERENCE.read_text(encoding="utf-8"))
