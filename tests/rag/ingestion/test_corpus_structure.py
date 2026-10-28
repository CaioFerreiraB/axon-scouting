"""Taxonomia e estrutura dos documentos (spec 102 FR-023, FR-024; ADR-048; research R33, R34)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from axon_scouting.rag.ingestion.domain.corpus_structure import (
    CorpusStructure,
    bodies_of,
    canonical_axis,
    canonical_block,
    parse_structure,
    scope_of,
    type_of,
)

CORPUS = Path(__file__).resolve().parents[3] / "corpus"


def taxonomia(*entradas: dict[str, Any]) -> dict[str, Any]:
    return {"version": 1, "entries": list(entradas)}


def estrutura(**changes: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "version": 1,
        "unit_kinds": {
            "capitulo": {"label": "Capítulo"},
            "secao": {"label": "Seção"},
            "artigo": {"label": "Artigo"},
            "ata": {},
            "documento": {},
            "eixo": {"label": "Eixo"},
            "bloco": {"label": "Bloco"},
            "especialidade": {"label": "Especialidade", "plural": "especialidades"},
        },
        "vocabulary": {
            "eixo": {"Meio Ambiente": ["Mudanças Climáticas"], "Paz e Desenvolvimento": ["Valores"]},
            "aliases": {"Paz e Compreensão": "Paz e Desenvolvimento"},
        },
        "profiles": {
            "artigos": {"sources": ["marcadores"], "units": ["capitulo", "secao", "artigo"]},
            "capitulos": {"sources": ["bookmarks", "títulos"], "units": ["capitulo", "secao"]},
            "especialidades": {"sources": ["sumário", "rodapé"], "units": ["eixo", "especialidade"]},
            "comunicado": {"sources": [], "units": ["documento"]},
        },
        "profile_by_type": {"Regulamento": "artigos", "Manual": "capitulos", "Ofício": "comunicado"},
        "documents": [
            {
                "match": ["guia-especialidades-*"],
                "profile": "especialidades",
                "branches": ["senior", "pioneiro"],
                "units": [{"kind": "especialidade", "pages": "24-221", "source": "sumário", "parent": "eixo"}],
            },
            {"folder": ["programa-educativo/guias"], "profile": "capitulos"},
        ],
    }
    base.update(changes)
    return base


def entradas() -> list[dict[str, Any]]:
    return [
        {"categoria": "Assembleias", "nivel": "Nacional", "tipo": "Regulamento", "pasta": "assembleias/regulamentos"},
        {"categoria": "Programa Educativo", "tipo": "Manual", "pasta": "programa-educativo/guias"},
        {
            "categoria": "Diretoria Regional",
            "tipo": "Ofício",
            "pasta": "diretoria-regional/oficios",
            "origem": "inferido",
        },
    ]


def montar(**changes: Any) -> CorpusStructure:
    structure, problems = parse_structure(taxonomia(*entradas()), estrutura(**changes))
    assert structure is not None, problems
    return structure


def test_a_pasta_classifica_com_a_regiao_e_o_ocr() -> None:
    structure = montar()
    assert structure.classify("assembleias/regulamentos/reg.pdf").semantic_type == "Regulamento"  # type: ignore[union-attr]
    regional = structure.classify("diretoria-regional/oficios/sao-paulo/of.pdf")
    assert regional is not None and regional.folder == "diretoria-regional/oficios" and regional.inferred
    assert structure.classify("ocr/assembleias/regulamentos/reg.pdf") is not None
    assert structure.classify("outra/pasta/x.pdf") is None


def test_perfil_pelo_tipo_semantico_e_pelas_excecoes() -> None:
    structure = montar()
    regulamento = structure.classify("assembleias/regulamentos/reg.pdf")
    guias = structure.classify("programa-educativo/guias/g.pdf")
    assert regulamento is not None and guias is not None
    assert structure.resolve("regulamento-x", regulamento).profile.name == "artigos"  # o tipo
    especialidades = structure.resolve("guia-especialidades-senior", guias)  # a exceção por documento vence
    assert especialidades.profile.name == "especialidades"
    assert especialidades.rule is not None and especialidades.rule.branches == ("senior", "pioneiro")
    assert especialidades.rule.units[0].pages == (24, 221)
    assert structure.resolve("guia-povo-livre", guias).profile.name == "capitulos"  # a exceção por pasta


def test_a_impressao_digital_muda_com_a_declaracao() -> None:
    structure = montar()
    guias = structure.classify("programa-educativo/guias/g.pdf")
    assert guias is not None
    antes = structure.resolve("guia-especialidades-x", guias).fingerprint()
    outra = montar(
        documents=[{"match": ["guia-especialidades-*"], "profile": "especialidades", "branches": ["senior"]}]
    )
    assert outra.resolve("guia-especialidades-x", guias).fingerprint() != antes
    assert structure.resolve("guia-especialidades-x", guias).fingerprint() == antes


def test_problemas_acumulados() -> None:
    ruim = estrutura(
        profiles={"artigos": {"sources": ["marcadores", "telepatia"], "units": ["artigo", "alinea"]}, "sonho": {}},
        profile_by_type={"Regulamento": "artigos", "Ofício": "inexistente"},
        documents=[
            {"profile": "artigos"},
            {
                "match": ["x"],
                "profile": "artigos",
                "branches": ["adulto"],
                "units": [{"kind": "especialidade", "pages": "30-10"}, {"kind": "nada", "pattern": "("}],
            },
        ],
    )
    tax = taxonomia(*entradas(), {"categoria": "X", "tipo": "Ata", "pasta": "assembleias/regulamentos"})
    structure, problems = parse_structure(tax, ruim)
    assert structure is None
    texto = "\n".join(problems)
    for esperado in (
        "pasta repetida",
        "fonte desconhecida 'telepatia'",
        "tipo de unidade não declarado 'alinea'",
        "perfil 'sonho' sem implementação",
        "perfil inexistente 'inexistente'",
        "o tipo 'Manual'",
        "precisa de `match` ou `folder`",
        "ramo desconhecido 'adulto'",
        "faixa de páginas inválida",
        "tipo de unidade não declarado 'nada'",
        "padrão inválido",
    ):
        assert esperado in texto, (esperado, problems)


def test_tipo_orgaos_e_abrangencia_vem_da_taxonomia() -> None:
    """ADR-049: sem `type` no catálogo, o tipo, os órgãos e a abrangência saem da entrada da pasta."""
    structure = montar()
    entrada = structure.classify("assembleias/regulamentos/r.pdf")
    assert entrada is not None
    resolucao = structure.resolve("regulamento-x", entrada)
    assert type_of(resolucao) == "regulamento"
    assert bodies_of(resolucao) == ("assembleias",)
    assert scope_of(resolucao, "sao-paulo") == "nacional"  # o nível da taxonomia vence a jurisdição
    sem_nivel = structure.resolve("manual-x", structure.classify("programa-educativo/guias/m.pdf"))  # type: ignore[arg-type]
    assert scope_of(sem_nivel, "sao-paulo") == "regional" and scope_of(sem_nivel, "mundial") == "mundial"


def test_eixo_e_bloco_pelo_vocabulario() -> None:
    structure = montar()
    guias = structure.classify("programa-educativo/guias/g.pdf")
    assert guias is not None
    resolution = structure.resolve("guia-povo-livre", guias)
    assert canonical_axis("24 Meio Ambiente", resolution) == "Meio Ambiente"
    assert canonical_axis("Paz e Compreensão", resolution) == "Paz e Desenvolvimento"  # o sinônimo
    assert canonical_axis("Habilidades", resolution) is None
    assert canonical_block("MUDANÇAS CLIMÁTICAS MEIO AMBIENTE", resolution) == ("Meio Ambiente", "Mudanças Climáticas")
    assert canonical_block("EDUCAÇÃO EM VALORES", resolution) is None  # o bloco precisa começar o título


# --- os arquivos do corpus ------------------------------------------------------------------------------------------


def test_os_arquivos_do_corpus_valem_e_cobrem_o_catalogo() -> None:
    def ler(nome: str) -> object:
        return yaml.safe_load((CORPUS / nome).read_text(encoding="utf-8"))

    structure, problems = parse_structure(ler("taxonomia.yaml"), ler("estrutura.yaml"))
    assert structure is not None, problems
    catalogo = ler("corpus.yaml")
    assert isinstance(catalogo, dict)
    perfis: Counter[str] = Counter()
    for entrada in catalogo["documents"]:
        classificacao = structure.classify(entrada["file"])
        assert classificacao is not None, f"{entrada['file']}: pasta fora da taxonomia"
        perfis[structure.resolve(entrada["document_id"], classificacao).profile.name] += 1
    assert sum(perfis.values()) == len(catalogo["documents"])
    assert perfis["especialidades"] == 2 and perfis["regras"] == 1 and perfis["artigos"] >= 281
