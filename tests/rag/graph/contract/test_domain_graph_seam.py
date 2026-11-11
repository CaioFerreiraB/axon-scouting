"""Conformidade do seam *Grafo de domínio* (spec 106 T014; contracts/domain-graph-seam.md, garantias 1–8)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from axon_scouting.rag.graph.adapters.artifact import ArtifactGraphStore
from axon_scouting.rag.graph.adapters.memory import MemoryGraphStore
from axon_scouting.rag.graph.api import DomainGraphSource, GraphUnavailable, PlanInvalid, QueryPlan, QueryStep
from axon_scouting.rag.graph.service import DomainGraphService

from ..grafo_sintetico import registros

INDEX = Path(__file__).resolve().parents[4] / "src" / "axon_scouting" / "data" / "index"


def _plano(*entidades: str) -> QueryPlan:
    return QueryPlan(consultas=(QueryStep(q="sobre", args={"entidades": list(entidades), "papeis": ["norma"]}),))


@pytest.fixture(params=["memoria", "artefato"])
def grafo(request: pytest.FixtureRequest) -> DomainGraphSource:
    if request.param == "memoria":
        return DomainGraphService(MemoryGraphStore(registros()))
    return DomainGraphService(ArtifactGraphStore(INDEX))


def _entidade(grafo: DomainGraphSource) -> str:
    return "tipo_orgao:diretoria_regional"


def test_e_o_protocolo(grafo: DomainGraphSource) -> None:
    assert isinstance(grafo, DomainGraphSource)


def test_determinismo_e_ordem(grafo: DomainGraphSource) -> None:
    plano = _plano(_entidade(grafo))
    uma, outra = asyncio.run(grafo.run(plano)), asyncio.run(grafo.run(plano))
    assert uma == outra and uma
    escopos = [e.scope for e in uma[0].evidence]
    ordem = ["mundial", "interamericano", "nacional", "regional", "local"]
    assert [ordem.index(s) for s in escopos if s in ordem] == sorted(ordem.index(s) for s in escopos if s in ordem)


def test_a_norma_de_cima_primeiro_em_cada_grupo(grafo: DomainGraphSource) -> None:
    ordem = ["mundial", "interamericano", "nacional", "regional", "local"]
    for grupo in asyncio.run(grafo.run(_plano(_entidade(grafo), "tipo_orgao:assembleia_regional"))):
        for entidade in {e.target for e in grupo.evidence}:
            niveis = [ordem.index(e.scope) for e in grupo.evidence if e.target == entidade and e.scope in ordem]
            assert niveis == sorted(niveis)


def test_governanca_nunca_no_run(grafo: DomainGraphSource) -> None:
    plano = QueryPlan(consultas=(QueryStep(q="ligados", args={"alvo": _entidade(grafo)}),))
    assert all(not g.evidence for g in asyncio.run(grafo.run(plano)))
    assert any(e.edge.type == "ELEGE" for e in asyncio.run(grafo.query("ligados", alvo=_entidade(grafo))))


def test_toda_evidencia_tem_fonte_e_origem(grafo: DomainGraphSource) -> None:
    for grupo in asyncio.run(grafo.run(_plano(_entidade(grafo)))):
        for e in grupo.evidence:
            assert e.edge.source and e.edge.origin in ("curado", "regra", "modelo_conferido")


def test_plano_vazio_e_entidade_inexistente(grafo: DomainGraphSource) -> None:
    assert asyncio.run(grafo.run(QueryPlan())) == ()
    assert asyncio.run(grafo.run(_plano("tipo_orgao:nao_existe")))[0].evidence == ()


def test_reconhecimento_pelos_termos(grafo: DomainGraphSource) -> None:
    termos = asyncio.run(grafo.aliases())
    assert "diretoria regional" in termos[_entidade(grafo)]


def test_argumento_desconhecido(grafo: DomainGraphSource) -> None:
    with pytest.raises((PlanInvalid, TypeError)):
        asyncio.run(grafo.query("sobre", entidades=[_entidade(grafo)], cor="azul"))


def test_sem_grafo_e_grafo_de_outro_build(tmp_path: Path) -> None:
    vazio = DomainGraphService(ArtifactGraphStore(tmp_path))
    assert asyncio.run(vazio.aliases()) == {} and asyncio.run(vazio.run(_plano("tipo_orgao:x"))) == ()
    (tmp_path / "manifest.json").write_text(json.dumps({"chunking": "novo"}), encoding="utf-8")
    (tmp_path / "graph.jsonl").write_text(json.dumps(registros(chunking="velho")[0]) + "\n", encoding="utf-8")
    with pytest.raises(GraphUnavailable):
        asyncio.run(DomainGraphService(ArtifactGraphStore(tmp_path)).aliases())
