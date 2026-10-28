"""`CatalogRetriever`, a tool `list_documents` e as fichas da lista de documentos (spec 103 FR-013, FR-014; ADR-047)."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from datetime import date
from typing import Any

import pytest
from axon_ai_contracts import Chunk, NamedRetriever, Retriever

from axon_scouting.rag.index.api import CatalogEntry, IndexFilters, ValidityStatus
from axon_scouting.rag.retrieval.catalog import CatalogRetriever, list_documents
from axon_scouting.rag.retrieval.domain.catalog import (
    CatalogRequest,
    card,
    card_doc_key,
    card_text,
    select,
    singular,
    with_cards,
)
from axon_scouting.rag.retrieval.domain.config import RetrievalConfig
from axon_scouting.rag.retrieval.domain.filters import InvalidFilters
from axon_scouting.rag.retrieval.domain.ordering import CATALOG_RANK, FUSION_SCORE
from axon_scouting.rag.retrieval.reranker import MetadataReranker
from axon_scouting.rag.retrieval.retriever import ScoutingRetriever

from .corpus import filtros, indice

HOJE = date(2026, 10, 7)


def listar(
    raw: Mapping[str, Any] | None, *, top_k: int | None = 20, config: RetrievalConfig | None = None
) -> list[Chunk]:
    async def go() -> list[Chunk]:
        return list(await CatalogRetriever(await indice(), config).search("lista", top_k=top_k, filters=raw))

    return asyncio.run(go())


def pedido(kind: str, *terms: str, **extra: object) -> dict[str, object]:
    return filtros(catalog={"kind": kind, "terms": list(terms)}, **extra)


def entrada(title: str, **changes: Any) -> CatalogEntry:
    base: dict[str, Any] = {
        "doc_key": "doc-1",
        "document_id": "doc",
        "version": "1",
        "document_type": "manual",
        "document_title": title,
        "jurisdiction": "nacional",
        "effective_date": date(2024, 1, 1),
        "status": ValidityStatus.VIGENTE,
    }
    return CatalogEntry(**{**base, **changes})


# --- a tool ------------------------------------------------------------------------------------------------------


def test_sem_pedido_de_lista_nao_devolve_nada() -> None:
    assert listar(filtros()) == []


def test_filtros_ausentes_sao_erro_como_na_busca() -> None:
    with pytest.raises(InvalidFilters):
        listar(None)


def test_lista_os_vigentes_da_especie_do_mais_recente_ao_mais_antigo() -> None:
    """Rascunho nunca; a versão substituída fica fora num pedido de hoje; empate de data, pelo título."""
    fichas = listar(pedido("manual"))
    assert [f.chunk_id for f in fichas] == ["manual-x-2026#acervo", "manual-esc-1#acervo", "manual-pio-1#acervo"]
    assert [f.section for f in fichas] == [f"ficha do acervo, {n} de 3" for n in (1, 2, 3)]
    assert all(f.page is None for f in fichas)


def test_qualificador_escolhe_os_titulos_que_o_tem() -> None:
    (ficha,) = listar(pedido("manual", "pioneiros"))
    assert ficha.chunk_id == "manual-pio-1#acervo" and ficha.score == 1.0
    assert ficha.document == "Manual Pioneiro" and ficha.status == "vigente"
    assert ficha.metadata["source_url"] == "https://exemplo.org/manual-pio-1.pdf"


def test_qualificador_que_nenhum_titulo_tem_nao_lista_nada() -> None:
    assert listar(pedido("manual", "uniforme")) == []


def test_filtro_de_ramo_vale_como_na_busca() -> None:
    fichas = listar(pedido("manual", branch="escoteiro"))
    assert {f.chunk_id for f in fichas} == {"manual-x-2026#acervo", "manual-esc-1#acervo"}


def test_consulta_historica_lista_a_versao_da_epoca() -> None:
    fichas = listar(
        pedido("manual", "x", effective_from="2023-06-01", effective_until="2023-06-01", include_superseded=True)
    )
    assert [f.chunk_id for f in fichas] == ["manual-x-2023#acervo"]
    assert fichas[0].status == "substituido"


def test_teto_de_fichas_com_o_total_na_secao() -> None:
    fichas = listar(pedido("manual"), config=RetrievalConfig(max_cards=2))
    assert len(fichas) == 2 and fichas[-1].section == "ficha do acervo, 2 de 3"
    assert len(listar(pedido("manual"), top_k=1)) == 1


@pytest.mark.parametrize(
    "valor",
    ["manual", {"kind": "", "terms": []}, {"kind": "manual", "terms": "x"}, {"terms": []}, {"kind": "m", "terms": [1]}],
)
def test_pedido_malformado_e_erro_nomeado(valor: object) -> None:
    with pytest.raises(InvalidFilters) as erro:
        listar(filtros(catalog=valor))
    assert erro.value.key == "catalog"


def test_a_busca_por_trechos_aceita_e_ignora_o_pedido() -> None:
    async def go() -> list[Chunk]:
        return list(await ScoutingRetriever(await indice()).search("acampamento", filters=pedido("manual")))

    assert all(not c.chunk_id.endswith("#acervo") for c in asyncio.run(go()))


def test_tool_list_documents() -> None:
    async def go() -> NamedRetriever:
        return list_documents(CatalogRetriever(await indice()), collection="scouting")

    tool = asyncio.run(go())
    assert tool.name == "list_documents" and tool.collections == frozenset({"scouting"})
    assert isinstance(tool.retriever, Retriever)


def test_fichas_vem_antes_dos_trechos_no_reranker() -> None:
    async def go() -> list[Chunk]:
        index = await indice()
        raw = pedido("manual")
        trechos = await ScoutingRetriever(index).search("acampamento da tropa", top_k=10, filters=raw)
        fichas = await CatalogRetriever(index).search("acampamento da tropa", top_k=10, filters=raw)
        return await MetadataReranker().rerank("acampamento da tropa", [*trechos, *fichas], top_k=5)

    ordem = [c.chunk_id for c in asyncio.run(go())]
    assert ordem[:3] == ["manual-x-2026#acervo", "manual-esc-1#acervo", "manual-pio-1#acervo"]
    assert not any(c.endswith("#acervo") for c in ordem[3:])


# --- domínio -----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("plural", "singular_"),
    [
        ("politicas", "politica"),
        ("resolucoes", "resolucao"),
        ("manuais", "manual"),
        ("nacionais", "nacional"),
        ("jovens", "jovem"),
        ("lideres", "lider"),
        ("especialidades", "especialidade"),
        ("atas", "ata"),
        ("2023", "2023"),
        ("can", "can"),
    ],
)
def test_singular(plural: str, singular_: str) -> None:
    assert singular(plural) == singular_


def test_maior_casamento_vence() -> None:
    """ "Atas da assembleia nacional": só as que têm as duas palavras, não as que têm uma."""
    entradas = [
        entrada(
            "Ata da Assembleia Nacional — 2024", doc_key="a-1", document_type="ata", effective_date=date(2024, 4, 1)
        ),
        entrada("Ata do Conselho Nacional", doc_key="b-1", document_type="ata"),
        entrada("Ata da 30ª Assembleia Nacional", doc_key="c-1", document_type="ata", effective_date=date(2025, 4, 1)),
        entrada("Política Nacional de Adultos", doc_key="d-1", document_type="politica"),
    ]
    escolhidos = select(entradas, CatalogRequest(kind="ata", terms=("assembleia", "nacional")))
    assert [(e.doc_key, s) for e, s in escolhidos] == [("c-1", 1.0), ("a-1", 1.0)]


def test_a_classificacao_qualifica_como_o_titulo() -> None:
    """ADR-049: "atas da assembleia regional" acha a ata da categoria Assembleias de nível Regional sem as palavras no
    título; e a espécie é o tipo da taxonomia, não uma palavra do título."""
    entradas = [
        entrada("Ata ARE 2024", doc_key="a-1", document_type="ata", category="Assembleias", level="Regional"),
        entrada("Ata do CAN 70", doc_key="b-1", document_type="ata", category="Conselho de Administração Nacional"),
        entrada("Ata assembleia regional (cópia)", doc_key="c-1", document_type="manual"),
    ]
    escolhidos = select(entradas, CatalogRequest(kind="ata", terms=("assembleia", "regional")))
    assert [e.doc_key for e, _ in escolhidos] == ["a-1"]


def test_qualquer_titulo_com_documento() -> None:
    entradas = [entrada("Guia de Especialidades", doc_key="g-1"), entrada("Manual do Lobinho", doc_key="m-1")]
    escolhidos = select(entradas, CatalogRequest(kind="documento", terms=("especialidades",)))
    assert [e.doc_key for e, _ in escolhidos] == ["g-1"]


def test_texto_da_ficha() -> None:
    vigente = entrada(
        "Política Nacional de Adultos", branch="lobinho", source_url="https://exemplo.org/p.pdf", version="2018"
    )
    assert card_text(vigente) == (
        "Política Nacional de Adultos\n"
        "Documento do acervo, versão 2018: em vigor desde 01/01/2024.\n"
        "Abrangência nacional; Ramo Lobinho.\n"
        "Original: https://exemplo.org/p.pdf"
    )
    antiga = entrada(
        "Manual X", status=ValidityStatus.SUBSTITUIDO, expiry_date=date(2025, 12, 31), superseded_by="manual-x-2026"
    )
    assert "substituído; em vigor de 01/01/2024 a 31/12/2025" in card_text(antiga)
    assert "todos os ramos" in card_text(antiga) and "Original" not in card_text(antiga)


def test_ficha_como_chunk_marcada_para_a_ordem() -> None:
    filtros_ = IndexFilters(effective_from=HOJE, effective_until=HOJE)
    ficha = card(entrada("Manual X", doc_key="manual-x-1"), rank=2, total=7, score=0.5, filters=filtros_)
    assert ficha.chunk_id == "manual-x-1#acervo" and ficha.section == "ficha do acervo, 3 de 7"
    assert ficha.metadata[CATALOG_RANK] == 2 and ficha.metadata[FUSION_SCORE] == 0.5


def test_texto_por_id_de_trecho_ou_de_ficha() -> None:
    entradas = {"manual-x-1": entrada("Manual X", doc_key="manual-x-1")}
    texto = with_cards({"t#s1@p1": "o trecho"}.get, entradas.get)
    assert texto("t#s1@p1") == "o trecho"
    assert texto("manual-x-1#acervo") == card_text(entradas["manual-x-1"])
    assert texto("outro-1#acervo") is None
    assert card_doc_key("manual-x-1#acervo") == "manual-x-1" and card_doc_key("t#s1@p1") is None
