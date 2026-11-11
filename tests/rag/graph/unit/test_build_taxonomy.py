"""As instâncias do `organizacao.yaml` e `APLICA_SE_A`/`EMITIDO_POR` pela taxonomia (spec 106 G3, US3; research R20)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from axon_scouting.rag.graph.domain.build.instances import instances_from_mapping, type_parts
from axon_scouting.rag.graph.domain.build.organs import OrganizationError
from axon_scouting.rag.graph.domain.build.taxonomy import applies_to, folder_of, issued_by, subfolder

ROOT = Path(__file__).resolve().parents[4]
CATALOGO = {
    "tipos_de_instituicao": [
        {"id": "ueb", "nivel": "nacional"},
        {"id": "regiao", "nivel": "regional"},
        {"id": "grupo_escoteiro", "nivel": "local"},
    ],
    "tipos_de_orgao": [
        {
            "id": "diretoria_regional",
            "nivel": "regional",
            "instituicao": "regiao",
            "taxonomia": "diretoria_regional",
            "instituido_em": "estatuto#art27",
        },
        {"id": "assembleia_regional", "nivel": "regional", "instituicao": "regiao", "taxonomia": "assembleias"},
        {
            "id": "assembleia_de_grupo",
            "nivel": "local",
            "instituicao": "grupo_escoteiro",
            "instituido_em": "a verificar",
        },
    ],
    "instituicoes": [
        {"id": "ueb", "instancia_de": "ueb", "fonte": "estatuto#art1"},
        {
            "id": "regiao_sp",
            "instancia_de": "regiao",
            "parte_de": ["ueb"],
            "jurisdicao": "sao-paulo",
            "aliases": ["Região de São Paulo"],
            "fonte": "regulamento#art1",
        },
        {
            "id": "ge_x",
            "instancia_de": "grupo_escoteiro",
            "parte_de": "regiao_sp",
            "pasta": "ge-x-1-sp",
            "fonte": "estatuto-ge#art1",
            "parte_de_fonte": "estatuto-ge#preambulo",
        },
        {"id": "ge_sem_fonte", "instancia_de": "grupo_escoteiro", "parte_de": "regiao_sp", "fonte": "a verificar"},
    ],
    "orgaos": [
        {"id": "dr_sp", "instancia_de": "diretoria_regional", "parte_de": "regiao_sp", "fonte": "regulamento#art2"},
        {"id": "ar_sp", "instancia_de": "assembleia_regional", "parte_de": "regiao_sp", "fonte": "regulamento#art2"},
    ],
}
TRECHOS = {
    "estatuto#art1",
    "estatuto#art27",
    "regulamento#art1",
    "regulamento#art2",
    "estatuto-ge#art1",
    "estatuto-ge#preambulo",
}
INST = instances_from_mapping(CATALOGO, TRECHOS)


def test_instancias_com_tipo_partes_e_fonte() -> None:
    ge = next(i for i in INST.found if i.id == "inst:ge_x")
    assert (ge.kind_of, ge.parents, ge.level, ge.parent_source) == (
        "tipo_inst:grupo_escoteiro",
        ("inst:regiao_sp",),
        "local",
        "estatuto-ge#preambulo",
    )
    regiao = next(i for i in INST.found if i.id == "inst:regiao_sp")
    assert regiao.terms == ("regiao de sao paulo",)
    dr = next(i for i in INST.found if i.id == "orgao:dr_sp")
    assert (dr.kind_of, dr.taxonomy, dr.level) == ("tipo_orgao:diretoria_regional", "diretoria_regional", "regional")


def test_sem_fonte_fica_fora() -> None:
    assert "inst:ge_sem_fonte" not in {i.id for i in INST.found}
    assert any("inst:ge_sem_fonte" in line for line in INST.dropped)


def test_tipo_desconhecido_e_erro_do_catalogo() -> None:
    errado = {**CATALOGO, "orgaos": [{"id": "x", "instancia_de": "nenhum", "fonte": "estatuto#art1"}]}
    with pytest.raises(OrganizationError, match="instancia_de"):
        instances_from_mapping(errado, TRECHOS)


def test_partes_entre_os_tipos_com_a_fonte_quando_conferida() -> None:
    assert type_parts(CATALOGO, TRECHOS) == [
        ("tipo_orgao:diretoria_regional", "tipo_inst:regiao", "estatuto#art27"),
        ("tipo_orgao:assembleia_regional", "tipo_inst:regiao", "organizacao.yaml:tipos_de_orgao/assembleia_regional"),
        (
            "tipo_orgao:assembleia_de_grupo",
            "tipo_inst:grupo_escoteiro",
            "organizacao.yaml:tipos_de_orgao/assembleia_de_grupo",
        ),
    ]


@pytest.mark.parametrize(
    ("campos", "arquivo", "esperada"),
    [
        ({"scope": "nacional", "jurisdiction": "nacional"}, "gestao/estatutos/nacional/E.pdf", "inst:ueb"),
        ({"scope": "nacional"}, "conselho-de-administracao-nacional/resolucoes/R.pdf", "inst:ueb"),
        (
            {"scope": "regional", "jurisdiction": "sao-paulo"},
            "diretoria-regional/resolucoes/sao-paulo/R.pdf",
            "inst:regiao_sp",
        ),
        ({"scope": "regional", "jurisdiction": "rio"}, "diretoria-regional/resolucoes/rio/R.pdf", None),
        ({"scope": "local", "jurisdiction": "sao-paulo"}, "gestao/estatutos/local/ge-x-1-sp/E.pdf", "inst:ge_x"),
        ({"scope": "local", "jurisdiction": "sao-paulo"}, "ocr/gestao/estatutos/local/ge-x-1-sp/E.pdf", "inst:ge_x"),
        ({"scope": "local", "jurisdiction": "sao-paulo"}, "gestao/estatutos/local/outro-ge/E.pdf", None),
    ],
)
def test_aplica_se_a_pelo_nivel_e_pela_subpasta(campos: dict[str, str], arquivo: str, esperada: str | None) -> None:
    assert applies_to(campos, arquivo, INST.institutions()) == esperada


def test_emitido_pela_pasta_do_tipo_e_pela_instituicao() -> None:
    organs = INST.organs()
    assert issued_by({"bodies": ["diretoria_regional", "resolucoes"]}, "inst:regiao_sp", organs) == ["orgao:dr_sp"]
    assert issued_by({"bodies": ["assembleias", "atas"]}, "inst:regiao_sp", organs) == ["orgao:ar_sp"]
    assert issued_by({"bodies": ["assembleias", "atas"]}, "inst:ueb", organs) == []  # a AN não está no catálogo
    assert issued_by({"bodies": ["gestao"]}, "inst:regiao_sp", organs) == []
    assert issued_by({"bodies": ["diretoria_regional"]}, None, organs) == []


def test_pasta_e_subpasta() -> None:
    assert folder_of("ocr/a/b/local/ge/x.pdf") == "a/b/local/ge"
    assert subfolder("a/b/local/ge/x.pdf") == "ge"
    assert subfolder("a/b/nacional/x.pdf") is None
    assert subfolder("a/b/x.pdf") is None


def test_o_catalogo_do_corpus_tem_as_instancias_do_g3() -> None:
    raw = yaml.safe_load((ROOT / "corpus" / "organizacao.yaml").read_text(encoding="utf-8"))
    ids = {f"inst:{e['id']}" for e in raw["instituicoes"]} | {f"orgao:{e['id']}" for e in raw["orgaos"]}
    assert {"inst:ueb", "inst:regiao_sp", "inst:ge_inah", "orgao:dr_sp", "orgao:ar_sp", "orgao:can"} <= ids
    assert all(e.get("fonte") for e in [*raw["instituicoes"], *raw["orgaos"]])
