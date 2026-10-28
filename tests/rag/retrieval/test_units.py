"""`UnitsRetriever`, a tool `list_units`, a junção dos repetidos e a ordem (spec 103 US8, FR-015, FR-016; ADR-048)."""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import Mapping
from datetime import date
from typing import Any

import pytest
from axon_ai_contracts import Chunk, NamedRetriever, Retriever

from axon_scouting.rag.index.adapters.bm25 import Bm25Encoder
from axon_scouting.rag.index.adapters.memory import HashingEmbedder, MemoryIndex, MemoryStorage
from axon_scouting.rag.index.api import UnitKind, UnitNode, UnitRef
from axon_scouting.rag.retrieval.catalog import CatalogRetriever
from axon_scouting.rag.retrieval.domain.filters import InvalidFilters
from axon_scouting.rag.retrieval.domain.ordering import ALSO_IN, STRUCTURE_RANK
from axon_scouting.rag.retrieval.domain.units import units_card_key, with_unit_cards
from axon_scouting.rag.retrieval.reranker import MetadataReranker
from axon_scouting.rag.retrieval.retriever import ScoutingRetriever
from axon_scouting.rag.retrieval.units import UnitsRetriever, list_units

from ..index.kit import campos, codificar, registro, trecho, versao
from .corpus import filtros

GUIA = campos("guia-sp", "1", document_title="Guia de Especialidades", effective_date=date(2025, 12, 1))
MANUAL = campos("manual-x", "2026", document_title="Manual X", effective_date=date(2026, 3, 1))
EIXO = UnitRef(unit_id="guia-sp-1#u/eixo-meio-ambiente", kind="eixo", title="Meio Ambiente")
UNIDADES = (
    UnitNode(
        unit_id=EIXO.unit_id, kind="eixo", title="Meio Ambiente", page_start=1, page_end=3, branches=("pioneiro",)
    ),
    UnitNode(
        unit_id=f"{EIXO.unit_id}/especialidade-ecoturismo",
        kind="especialidade",
        title="Ecoturismo",
        page_start=1,
        page_end=1,
        parent=EIXO.unit_id,
        branches=("pioneiro",),
    ),
    UnitNode(
        unit_id=f"{EIXO.unit_id}/especialidade-sustentabilidade",
        kind="especialidade",
        title="Sustentabilidade",
        page_start=2,
        page_end=3,
        parent=EIXO.unit_id,
        branches=("pioneiro",),
    ),
)
ETAPAS = "O projeto não tem uma duração determinada e pode ser feito em equipe de interesse."
TRECHOS = [
    trecho(
        GUIA,
        "s1",
        1,
        "[Guia · Eixo: Meio Ambiente]\nAs especialidades do eixo: • Ecoturismo • Sustentabilidade • Agricultura "
        "• Clima",
        shape="list",
        branches=("pioneiro",),
    ),
    trecho(GUIA, "s2", 2, f"[Guia · Insígnia A · Etapas do projeto]\n{ETAPAS}", branches=("pioneiro",)),
    trecho(GUIA, "s3", 3, f"[Guia · Insígnia B · Etapas do projeto]\n{ETAPAS}", branches=("pioneiro",)),
]
MANUAL_TRECHOS = [trecho(MANUAL, "s1", 1, "[Manual X]\nAs especialidades no projeto de vida do jovem.")]
TIPOS = (UnitKind(name="especialidade", label="Especialidade", plurals=("especialidades",)),)


async def indice() -> MemoryIndex:
    embedder, sparse = HashingEmbedder(), Bm25Encoder()
    index = MemoryIndex("scouting", embedder, sparse, storage=MemoryStorage())
    build = await index.create(versao(embedder, sparse, unit_kinds=TIPOS))
    guia = dataclasses.replace(registro(GUIA, TRECHOS, ["p1", "p2", "p3"]), units=UNIDADES)
    await index.write(build.name, guia, await codificar(TRECHOS, embedder, sparse))
    await index.write(
        build.name, registro(MANUAL, MANUAL_TRECHOS, ["p"]), await codificar(MANUAL_TRECHOS, embedder, sparse)
    )
    await index.publish(build.name)
    return index


def unidades(raw: Mapping[str, Any] | None, pergunta: str = "quais são as especialidades") -> list[Chunk]:
    async def go() -> list[Chunk]:
        index = await indice()
        return list(await UnitsRetriever(index, index).search(pergunta, top_k=20, filters=raw))

    return asyncio.run(go())


def pedido(kind: str | None, *terms: str, **extra: object) -> dict[str, object]:
    return filtros(enumeration={"unit_kind": kind, "terms": list(terms)}, **extra)


def test_tool_list_units() -> None:
    async def go() -> None:
        index = await indice()
        tool = list_units(UnitsRetriever(index, index))
        assert isinstance(tool, NamedRetriever) and isinstance(tool.retriever, Retriever)
        assert tool.name == "list_units" and tool.collections == frozenset({"scouting"})

    asyncio.run(go())


def test_com_o_tipo_uma_ficha_por_documento_com_as_unidades() -> None:
    (ficha,) = unidades(pedido("especialidade", branch="pioneiro"))
    assert ficha.chunk_id == "guia-sp-1#unidades/especialidade"
    assert units_card_key(ficha.chunk_id) == ("guia-sp-1", "especialidade")
    assert ficha.text.splitlines() == [
        "Guia de Especialidades",
        "Lista de especialidades do documento (2):",
        "- Ecoturismo — Meio Ambiente (p. 1)",
        "- Sustentabilidade — Meio Ambiente (p. 2–3)",
    ]
    assert ficha.section == "lista de especialidades, 1 de 1" and ficha.page is None
    assert ficha.metadata[STRUCTURE_RANK] == 0


def test_o_ramo_da_unidade_filtra() -> None:
    assert unidades(pedido("especialidade", branch="lobinho")) == []


def test_sem_tipo_os_trechos_com_forma_de_lista() -> None:
    (lista,) = unidades(pedido(None, "eixo"))
    assert lista.chunk_id == "guia-sp-1#s1@p1" and lista.metadata[STRUCTURE_RANK] == 0


def test_sem_pedido_nada_e_pedido_malformado_falha() -> None:
    assert unidades(filtros()) == []
    with pytest.raises(InvalidFilters, match="enumeration"):
        unidades(filtros(enumeration={"unit_kind": 3}))


def test_trechos_de_texto_identico_ficam_um_so() -> None:
    async def go() -> list[Chunk]:
        return list(
            await ScoutingRetriever(await indice()).search("projeto equipe de interesse", top_k=10, filters=filtros())
        )

    achados = asyncio.run(go())
    etapas = [c for c in achados if c.text.endswith(ETAPAS)]
    assert len(etapas) == 1 and len(etapas[0].metadata[ALSO_IN]) == 1  # type: ignore[arg-type]


def test_o_reranker_poe_a_estrutura_depois_das_fichas_e_sem_repetir() -> None:
    async def go() -> list[Chunk]:
        index = await indice()
        raw = pedido(None, "eixo")
        busca = await ScoutingRetriever(index).search("especialidades eixo", top_k=10, filters=raw)
        estrutura = await UnitsRetriever(index, index).search("especialidades eixo", top_k=10, filters=raw)
        fichas = await CatalogRetriever(index).search(
            "lista", top_k=10, filters=filtros(catalog={"kind": "manual", "terms": ["especialidades"]})
        )
        return list(await MetadataReranker().rerank("q", [*busca, *estrutura, *fichas], top_k=10))

    ordem = [c.chunk_id for c in asyncio.run(go())]
    assert ordem[:2] == ["guia-sp-1#acervo", "guia-sp-1#s1@p1"]  # a ficha do acervo, depois a estrutura
    assert ordem.index("guia-sp-1#s1@p1") < min(ordem.index(i) for i in ordem if i.endswith("@p2") or i.endswith("@p3"))
    assert len(ordem) == len(set(ordem))


def test_a_ficha_se_refaz_da_arvore_para_a_metrica_de_fatos() -> None:
    async def go() -> tuple[str, Any]:
        index = await indice()
        ficha = (await UnitsRetriever(index, index).search("q", top_k=5, filters=pedido("especialidade")))[0]
        return ficha.text, await index.outline("guia-sp-1")

    texto, arvore = asyncio.run(go())
    text_of = with_unit_cards(
        lambda _id: None, lambda key: arvore if key == "guia-sp-1" else [], {"especialidade": TIPOS[0]}
    )
    assert text_of("guia-sp-1#unidades/especialidade") == texto
    assert text_of("guia-sp-1#s1@p1") is None


def test_tipo_espalhado_por_documentos_demais_cai_para_as_listas() -> None:
    """ "Quais são os blocos", sem ramo: a pergunta não disse de onde listar (MAX_DOCUMENTS)."""

    class Espalhado:
        def __init__(self, index: MemoryIndex) -> None:
            self._index = index

        async def unit_kinds(self) -> tuple[UnitKind, ...]:
            return TIPOS

        async def units(self, kind: str, *, filters: Any) -> list[Any]:
            (entrada,) = [u for u in await self._index.outline("guia-sp-1") if u.unit.kind == "eixo"]
            return [
                dataclasses.replace(entrada, document=dataclasses.replace(entrada.document, doc_key=f"d-{n}"))
                for n in range(4)
            ]

        async def outline(self, doc_key: str) -> list[Any]:
            return []

    async def go() -> list[Chunk]:
        index = await indice()
        return list(await UnitsRetriever(index, Espalhado(index)).search("eixo", top_k=5, filters=pedido("eixo")))

    (lista,) = asyncio.run(go())
    assert lista.chunk_id == "guia-sp-1#s1@p1"


def _trecho(chunk_id: str, page: int, group: str | None, size: int = 40) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text="x" * size,
        score=0.5,
        document="Guia",
        version="1",
        status="vigente",
        section=None,
        page=page,
        metadata={} if group is None else {"unit_group": group},
    )


def test_irmaos_da_mesma_unidade_vao_juntos_na_ordem_do_documento() -> None:
    from axon_scouting.rag.retrieval.domain.ordering import by_unit

    entregues = [
        _trecho("g#fazer@p51", 51, "esp-gastronomia"),
        _trecho("m#x@p3", 3, None),
        _trecho("v#a@p70", 70, "esp-viagens"),
        _trecho("g#temas@p49", 49, "esp-gastronomia"),
    ]
    assert [c.chunk_id for c in by_unit(entregues, max_tokens=1500)] == [
        "g#temas@p49",
        "g#fazer@p51",
        "m#x@p3",
        "v#a@p70",
    ]


def test_o_teto_por_unidade_deixa_o_resto_no_lugar() -> None:
    from axon_scouting.rag.retrieval.domain.ordering import by_unit

    entregues = [
        _trecho("g#a@p1", 1, "u", size=4000),
        _trecho("m#x@p9", 9, None),
        _trecho("g#b@p2", 2, "u", size=4000),
    ]
    assert [c.chunk_id for c in by_unit(entregues, max_tokens=1500)] == ["g#a@p1", "m#x@p9", "g#b@p2"]


def test_sem_tipo_e_com_facetas_nada() -> None:
    """A pergunta por ramo já espalha as vagas pelas facetas; a rota sem tipo as tiraria (103 R17)."""
    assert unidades(pedido(None, "etapas", branches=["lobinho", "escoteiro"])) == []
