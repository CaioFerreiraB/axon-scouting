"""O grafo das normas por órgão (ADR-063): catálogo, ligação pelo título e o regulamento inteiro."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from axon_scouting.rag.graph.domain.build.organs import (
    ChunkPlace,
    OrganizationError,
    organ_of_title,
    organs_from_mapping,
    sections,
)

ROOT = Path(__file__).resolve().parents[4]
CATALOGO = {
    "tipos_de_instituicao": [{"id": "regiao", "nome": "Região Escoteira", "aliases": ["regiao"]}],
    "tipos_de_orgao": [
        {"id": "diretoria_regional", "nome": "Diretoria Regional", "nivel": "regional"},
        {"id": "conselho_consultivo_nacional", "nome": "Conselho Consultivo", "aliases": ["conselho consultivo"]},
        {"id": "conselho_consultivo_regional", "nome": "Conselho Consultivo Regional"},
        {"id": "conselho_de_administracao_nacional", "nome": "Conselho de Administração Nacional"},
    ],
}
ORGAOS = organs_from_mapping(CATALOGO)


def lugar(
    chunk_id: str,
    doc_key: str,
    *path: tuple[str, str, str],
    tipo: str = "estatuto",
    titulo: str = "Estatuto",
    escopo: str = "nacional",
    vigente: bool = True,
) -> ChunkPlace:
    return ChunkPlace(
        chunk_id=chunk_id,
        doc_key=doc_key,
        document_type=tipo,
        document_title=titulo,
        scope=escopo,
        vigente=vigente,
        path=path,
    )


def test_o_nome_mais_longo_vence() -> None:
    assert organ_of_title("Capítulo V – Do Conselho Consultivo Regional", ORGAOS) == "conselho_consultivo_regional"
    assert organ_of_title("Seção V – Do Conselho Consultivo", ORGAOS) == "conselho_consultivo_nacional"
    assert organ_of_title("Capítulo 4 – Do Nível Regional", ORGAOS) is None


def test_a_secao_mais_funda_que_nomeia_o_orgao() -> None:
    capitulo = ("e#cap4", "capitulo", "Capítulo 4 – Do Nível Regional")
    secao = ("e#s3", "secao", "Seção III – Da Diretoria Regional")
    artigo = ("e#art27", "artigo", "Art. 27")
    regulamento = ("r#cap3", "capitulo", "Capítulo III – Da Diretoria Regional")
    sub = ("r#s1", "secao", "Seção I – Do Diretor-presidente")
    achadas = sections(
        [
            lugar("e#art27@p19", "estatuto", capitulo, secao, artigo),
            lugar(
                "r#art26@p9",
                "regulamento-sp",
                regulamento,
                tipo="regulamento",
                titulo="Regulamento da Região",
                escopo="regional",
            ),
            lugar(
                "r#art28@p10",
                "regulamento-sp",
                regulamento,
                sub,
                tipo="regulamento",
                titulo="Regulamento da Região",
                escopo="regional",
            ),
            lugar("velho#art1", "estatuto-1999", capitulo, secao, vigente=False),
            lugar("m#1", "manual", secao, tipo="manual"),
        ],
        ORGAOS,
    )
    assert [(s.organ, s.doc_key, s.chunk_ids) for s in achadas] == [
        ("diretoria_regional", "estatuto", ("e#art27@p19",)),  # a norma de cima primeiro
        ("diretoria_regional", "regulamento-sp", ("r#art26@p9", "r#art28@p10")),  # a seção I sobe ao capítulo
    ]


def test_regulamento_cujo_titulo_nomeia_o_orgao_vale_inteiro() -> None:
    titulo = "Regulamento Interno do Conselho de Administração Nacional"
    achadas = sections(
        [
            lugar(
                "can#art13@p6",
                "reg-can",
                ("can#cap2", "capitulo", "Capítulo II – Das Reuniões"),
                tipo="regulamento",
                titulo=titulo,
            ),
            lugar(
                "sp#art1",
                "reg-sp",
                ("sp#c1", "capitulo", "Capítulo I"),
                tipo="regulamento",
                titulo="Regulamento da Região Escoteira de São Paulo",
                escopo="regional",
            ),
        ],
        ORGAOS,
    )
    assert [(s.organ, s.unit_id) for s in achadas] == [("conselho_de_administracao_nacional", "reg-can#documento")]


def test_catalogo_invalido() -> None:
    with pytest.raises(OrganizationError):
        organs_from_mapping({"tipos_de_orgao": [{"nome": "Sem id"}]})
    with pytest.raises(OrganizationError):
        organs_from_mapping({"tipos_de_orgao": [{"id": "a", "nome": "Órgão A"}, {"id": "a", "nome": "Órgão B"}]})


def test_o_catalogo_do_corpus_carrega() -> None:
    organs = organs_from_mapping(yaml.safe_load((ROOT / "corpus" / "organizacao.yaml").read_text(encoding="utf-8")))
    ids = {organ.id for organ in organs}
    assert {"diretoria_regional", "assembleia_regional", "conselho_de_administracao_nacional"} <= ids


def test_relacoes_so_com_fonte_conferida_e_presente() -> None:
    from axon_scouting.rag.graph.domain.build.organs import cargo_terms, relations_from_mapping

    bruto = {
        "relacoes": [
            {"tipo": "ELEGE", "de": "assembleia_regional", "para": "diretoria_regional", "fonte": "e#art24@p17"},
            {"tipo": "NOMEIA", "de": "can", "para": "den", "fonte": "a verificar"},
            {"tipo": "FISCALIZA", "de": "cfr", "para": "dr", "fonte": "inexistente#x"},
        ],
        "cargos": [
            {"id": "c1", "nome": "Conselheiro Nacional", "orgao": "conselho_de_administracao_nacional"},
            {"id": "c2", "nome": "Vice", "orgao": "diretoria_regional"},
        ],
    }
    (relacao,) = relations_from_mapping(bruto, {"e#art24@p17"})
    assert (relacao.kind, relacao.source, relacao.target) == ("ELEGE", "assembleia_regional", "diretoria_regional")
    assert cargo_terms(bruto) == {"conselho_de_administracao_nacional": ("conselheiro nacional",)}
