"""Derivação de contexto da pergunta, sem modelo (spec 101 US2, US3; data-model; research R8, R12; T016 e T022)."""

from __future__ import annotations

import asyncio
import dataclasses
import json
from datetime import UTC, date, datetime
from types import MappingProxyType

import pytest
from axon_ai_contracts.testing import pack_context, turn_view

from axon_scouting.context.derivation import EnumerationRequest, ScoutingContext, derive_context, to_slice
from axon_scouting.context.lexicon import LEXICON
from axon_scouting.context.periods import Validity
from axon_scouting.pack import ScoutingPack

DIA = date(2026, 10, 7)
HOJE = Validity.day(DIA)

# --- US2: ramo, temas e tipos -------------------------------------------------------------------------------------


def test_us2_cenario_1() -> None:
    contexto = derive_context("Quais competências do Pioneiro estão relacionadas à liderança?", DIA)
    assert contexto == ScoutingContext(
        branch="pioneiro",
        branches=(),
        topics=("lideranca",),
        document_types=(),
        suggested_types=("manual",),
        validity=HOJE,
        include_superseded=False,
        previous_version=False,
        enumeration=EnumerationRequest(unit_kind=None, terms=("competencias", "relacionadas", "lideranca")),
    )


def test_us2_cenario_2_sem_ramo() -> None:
    contexto = derive_context("Como funciona a promessa?", DIA)
    assert contexto.branch is None
    assert contexto.topics == ("promessa_e_lei",)
    assert contexto.document_types == ()
    assert contexto.suggested_types == ("manual", "por")
    assert contexto.validity == HOJE


@pytest.mark.parametrize(
    ("pergunta", "ramo"),
    [
        ("O que um escoteiro deve levar no acampamento?", None),  # "escoteiro" sozinho é ambíguo
        ("Qual a diferença entre lobinho e sênior?", None),  # dois ramos: na dúvida, sem filtro
        ("O que faz um rover?", "pioneiro"),  # sinônimo antigo
        ("Como é a tropa escoteira?", "escoteiro"),
        ("Escoteiros seniores podem acampar sozinhos?", "senior"),
        ("COMPETÊNCIAS DO PIONEIRO", "pioneiro"),
        ("Atividades de pioneirismo para a tropa", None),  # pioneirismo é tema, não o Ramo Pioneiro
    ],
)
def test_ramo(pergunta: str, ramo: str | None) -> None:
    assert derive_context(pergunta, DIA).branch == ramo


def test_tipo_citado_prevalece_sobre_os_sugeridos() -> None:
    contexto = derive_context("O que diz o estatuto sobre liderança?", DIA)
    assert contexto.document_types == ("estatuto",)
    assert contexto.suggested_types == ()
    assert contexto.topics == ("lideranca",)


def test_pergunta_vazia_da_o_contexto_default() -> None:
    assert derive_context("", DIA) == ScoutingContext(
        branch=None,
        branches=(),
        topics=(),
        document_types=(),
        suggested_types=(),
        validity=HOJE,
        include_superseded=False,
        previous_version=False,
    )


# --- US2: a fatia publicada (R12; contracts/retrieval-filters.md) -------------------------------------------------


def test_fatia_com_ramo() -> None:
    fatia = to_slice(derive_context("Quais competências do Pioneiro estão relacionadas à liderança?", DIA))
    assert fatia == {
        "context": {
            "branch": "pioneiro",
            "branches": [],
            "topics": ["lideranca"],
            "document_types": [],
            "suggested_types": ["manual"],
            "effective_from": "2026-10-07",
            "effective_until": "2026-10-07",
            "include_superseded": False,
            "previous_version": False,
            "bodies": [],
            "scopes": [],
            "suggested_bodies": [],
            "catalog": None,
            "enumeration": {"unit_kind": None, "terms": ["competencias", "relacionadas", "lideranca"]},
            "anchor": None,
            "entities": [],
            "plan": None,
            "synthesis": None,
        },
        "retrieval_filters": {
            "branch": "pioneiro",
            "topics": ["lideranca"],
            "suggested_types": ["manual"],
            "effective_from": "2026-10-07",
            "effective_until": "2026-10-07",
            "include_superseded": False,
            "previous_version": False,
            "enumeration": {"unit_kind": None, "terms": ["competencias", "relacionadas", "lideranca"]},
        },
    }
    json.dumps(fatia)


def test_filtros_omitem_ramo_e_listas_vazias() -> None:
    fatia = to_slice(derive_context("", DIA))
    assert fatia["context"]["branch"] is None  # type: ignore[index]
    assert fatia["retrieval_filters"] == {
        "effective_from": "2026-10-07",
        "effective_until": "2026-10-07",
        "include_superseded": False,
        "previous_version": False,
    }


def test_prepare_context_publica_a_fatia() -> None:
    pergunta = "Quais competências do Pioneiro estão relacionadas à liderança?"
    view = turn_view(question=pergunta, reference_time=datetime(2026, 10, 8, 1, tzinfo=UTC))  # 22h do dia 7 em Brasília
    fatia = asyncio.run(ScoutingPack().prepare_context(view, pack_context("scouting")))
    assert fatia == to_slice(derive_context(pergunta, DIA))


# --- US3: consulta histórica ---------------------------------------------------------------------------------------


def test_historico_us3_cenario_1() -> None:
    contexto = derive_context("Qual era a regra em 2023?", DIA)
    assert contexto.validity == Validity(start=date(2023, 1, 1), end=date(2023, 12, 31))
    assert contexto.include_superseded is True
    assert contexto.previous_version is False
    assert contexto.document_types == ()
    assert contexto.suggested_types == ("por", "resolucao", "regulamento")


def test_historico_us3_cenario_2() -> None:
    contexto = derive_context("O que dizia o manual anterior?", DIA)
    assert contexto.validity == HOJE
    assert contexto.include_superseded is True
    assert contexto.previous_version is True
    assert contexto.document_types == ("manual",)


def test_historico_antigamente_mira_a_versao_anterior() -> None:
    contexto = derive_context("Como era a promessa antigamente?", DIA)
    assert contexto.previous_version is True
    assert contexto.include_superseded is True


def test_historico_filtros_com_o_periodo() -> None:
    filtros = to_slice(derive_context("Qual era a regra em 2023?", DIA))["retrieval_filters"]
    assert filtros == {
        "suggested_types": ["por", "resolucao", "regulamento"],
        "effective_from": "2023-01-01",
        "effective_until": "2023-12-31",
        "include_superseded": True,
        "previous_version": False,
    }


# --- facetas e o Ramo Filhotes (ADR-045) ---------------------------------------------------------------------------

TODOS = ("filhotes", "lobinho", "escoteiro", "senior", "pioneiro")


@pytest.mark.parametrize(
    ("pergunta", "branch", "branches"),
    [
        ("Qual o marco simbólico dos 5 ramos?", None, TODOS),
        ("Qual o lema de cada ramo?", None, TODOS),
        ("Quais os marcos simbólicos dos cinco ramos?", None, TODOS),
        ("Qual a frase do marco simbólico dos ramos Filhotes, Lobinho, Escoteiro, Sênior e Pioneiro?", None, TODOS),
        ("Qual a diferença entre o marco simbólico do Ramo Lobinho e o do Ramo Sênior?", None, ("lobinho", "senior")),
        ("Qual o marco simbólico do Ramo Filhotes?", "filhotes", ()),
        ("Quantas crianças cabem numa ninhada?", "filhotes", ()),
        ("O que é a promessa escoteira?", None, ()),
    ],
)
def test_facetas(pergunta: str, branch: str | None, branches: tuple[str, ...]) -> None:
    contexto = derive_context(pergunta, DIA)
    assert (contexto.branch, contexto.branches) == (branch, branches)


def test_fatia_com_facetas_nao_tem_branch() -> None:
    filtros = to_slice(derive_context("Qual o lema de cada ramo?", DIA))["retrieval_filters"]
    assert isinstance(filtros, dict) and "branch" not in filtros and filtros["branches"] == list(TODOS)


# --- tipo citado filtra, sugerido ordena; tempo relativo (ADR-046) -------------------------------------------------


@pytest.mark.parametrize(
    ("pergunta", "citados", "sugeridos"),
    [
        ("Quais são as especialidades para o Ramo Sênior?", (), ("manual",)),
        ("Procure as especialidades no guia de especialidades do ramo sênior", ("manual",), ()),  # guia é Manual
        ("O que a Política Nacional de Programa Educativo define?", ("politica",), ()),  # tipo da taxonomia
        ("Quais foram as chapas concorrentes à diretoria regional?", (), ("por", "resolucao", "estatuto")),
    ],
)
def test_so_o_tipo_citado_filtra(pergunta: str, citados: tuple[str, ...], sugeridos: tuple[str, ...]) -> None:
    contexto = derive_context(pergunta, DIA)
    assert (contexto.document_types, contexto.suggested_types) == (citados, sugeridos)
    filtros = to_slice(contexto)["retrieval_filters"]
    assert isinstance(filtros, dict) and filtros.get("document_types", []) == list(citados)


@pytest.mark.parametrize(
    ("pergunta", "ano"),
    [
        ("Qual o parecer da comissão fiscal sobre as contas do último ano?", 2025),
        ("E no ano passado?", 2025),
        ("O que mudou no ano anterior?", 2025),
        ("Quais resoluções saíram este ano?", 2026),
        ("O que foi aprovado no ano atual?", 2026),
    ],
)
def test_tempo_relativo(pergunta: str, ano: int) -> None:
    contexto = derive_context(pergunta, date(2026, 10, 27))
    assert contexto.validity == Validity(start=date(ano, 1, 1), end=date(ano, 12, 31))
    assert contexto.include_superseded is True


# --- pedido de lista de documentos (ADR-047) ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("pergunta", "esperado"),
    [
        ("Quais são as políticas nacionais que existem?", ("politica", ("nacionais",))),
        ("Liste as resoluções sobre uniforme", ("resolucao", ("uniforme",))),
        ("Quais manuais existem para o Ramo Lobinho?", ("manual", ("ramo", "lobinho"))),
        ("Que documentos existem sobre especialidades?", ("documento", ("especialidades",))),
        ("Quais são os guias disponíveis?", ("manual", ())),
        ("Quais atas da assembleia nacional de 2023?", ("ata", ("assembleia", "nacional", "2023"))),
    ],
)
def test_pedido_de_lista(pergunta: str, esperado: tuple[str, tuple[str, ...]]) -> None:
    contexto = derive_context(pergunta, DIA)
    assert contexto.catalog is not None
    assert (contexto.catalog.kind, contexto.catalog.terms) == esperado
    kind, terms = esperado
    assert to_slice(contexto)["retrieval_filters"]["catalog"] == {"kind": kind, "terms": list(terms)}  # type: ignore[index]


@pytest.mark.parametrize(
    "pergunta",
    [
        "Quais são as etapas do Ramo Escoteiro?",  # enumeração, mas não de documentos
        "Quais competências do Pioneiro estão relacionadas à liderança?",
        "O que diz a política nacional de adultos?",  # cita a espécie, não pede a lista
        "Quais documentos existem?",  # sem qualificador seria o acervo inteiro
    ],
)
def test_sem_pedido_de_lista(pergunta: str) -> None:
    contexto = derive_context(pergunta, DIA)
    assert contexto.catalog is None
    assert "catalog" not in to_slice(contexto)["retrieval_filters"]  # type: ignore[operator]


# --- ADR-048: o pedido de enumeração -------------------------------------------------------------------------------

COM_TIPOS = dataclasses.replace(
    LEXICON,
    unit_kinds=MappingProxyType({"especialidade": ("especialidades",), "insignia": ("insignias", "distintivos")}),
)


@pytest.mark.parametrize(
    ("pergunta", "esperado"),
    [
        ("Quais são as especialidades do ramo pioneiro?", EnumerationRequest(unit_kind="especialidade", terms=())),
        ("Liste as insígnias do Ramo Sênior", EnumerationRequest(unit_kind="insignia", terms=())),
        ("Quais são os órgãos nacionais da UEB?", EnumerationRequest(unit_kind=None, terms=("orgaos", "nacionais"))),
        ("Quais são as políticas nacionais?", None),  # espécie de documento: é o `catalog` (ADR-047)
        ("O que é a especialidade de gastronomia?", None),  # sem marcador de enumeração
    ],
)
def test_enumeracao(pergunta: str, esperado: EnumerationRequest | None) -> None:
    assert derive_context(pergunta, DIA, COM_TIPOS).enumeration == esperado


def test_sem_tipos_no_lexico_a_enumeracao_sai_sem_tipo() -> None:
    contexto = derive_context("Quais são as especialidades do ramo pioneiro?", DIA)
    assert contexto.enumeration == EnumerationRequest(unit_kind=None, terms=("especialidades",))


@pytest.mark.parametrize(
    ("pergunta", "tipos", "orgaos", "abrangencias", "sugeridos"),
    [
        ("Quais são as atas da assembleia regional de SP?", ("ata",), ("assembleias",), ("regional",), ()),
        ("Segundo o regulamento da RNJL, quem vota?", ("regulamento",), ("rede_de_jovens_lideres",), (), ()),
        ("Como funciona o CAN?", (), (), (), ("conselho_de_administracao_nacional",)),
        ("Como funciona a assembleia de grupo?", (), (), (), ()),
    ],
)
def test_orgao_e_abrangencia_filtram_so_com_tipo_ou_lista(
    pergunta: str,
    tipos: tuple[str, ...],
    orgaos: tuple[str, ...],
    abrangencias: tuple[str, ...],
    sugeridos: tuple[str, ...],
) -> None:
    """ADR-049: o órgão e a abrangência citados filtram junto com um tipo citado ou um pedido de lista; sozinhos, o
    órgão só desempata, e "assembleia" sem nacional ou regional não é órgão."""
    contexto = derive_context(pergunta, DIA)
    assert (contexto.document_types, contexto.bodies, contexto.scopes) == (tipos, orgaos, abrangencias)
    assert contexto.suggested_bodies == sugeridos


# --- ADR-061: o órgão emissor filtra; o órgão assunto só desempata --------------------------------------------------


@pytest.mark.parametrize(
    ("pergunta", "tipos", "orgaos", "abrangencias", "sugeridos"),
    [
        # Assunto: o tipo citado filtra, o órgão vira desempate, e o nome dele não é abrangência.
        (
            "O que diz o regulamento regional sobre a diretoria regional?",
            ("regulamento",),
            (),
            ("regional",),
            ("diretoria_regional",),
        ),
        ("Segundo o estatuto, qual o mandato da diretoria regional?", ("estatuto",), (), (), ("diretoria_regional",)),
        ("O que o estatuto diz sobre o CAN?", ("estatuto",), (), (), ("conselho_de_administracao_nacional",)),
        # Emissor: o órgão logo depois do tipo, ligado por da/do/das/dos/de, filtra.
        ("Quais as resoluções da diretoria regional?", ("resolucao",), ("diretoria_regional",), ("regional",), ()),
        ("Quais são as atas da assembleia regional?", ("ata",), ("assembleias",), ("regional",), ()),
        ("Regimento interno do conselho jovem: quem vota?", ("regulamento",), ("conselho_jovem",), (), ()),
        ("O parecer da comissão fiscal aprovou as contas?", ("parecer",), ("comissao_fiscal",), (), ()),
    ],
)
def test_orgao_emissor_filtra_e_assunto_desempata(
    pergunta: str,
    tipos: tuple[str, ...],
    orgaos: tuple[str, ...],
    abrangencias: tuple[str, ...],
    sugeridos: tuple[str, ...],
) -> None:
    contexto = derive_context(pergunta, DIA)
    assert (contexto.document_types, contexto.bodies, contexto.scopes) == (tipos, orgaos, abrangencias)
    assert contexto.suggested_bodies == sugeridos


# --- ADR-061: a pergunta de seguimento herda o assunto da anterior --------------------------------------------------

REELEICAO = (
    "para a diretoria regional da região de são paulo, uma pessoa que foi diretor vice presidente em duas gestões "
    "seguidas, pode se candidatar a diretor presidente na gestão seguinte?"
)


@pytest.mark.parametrize(
    ("pergunta", "anteriores", "ancora"),
    [
        ("olhe no estatuto nacional", (REELEICAO,), REELEICAO),
        ("e no estatuto nacional?", (REELEICAO,), REELEICAO),
        ("Qual o prazo de gestão?", ("Como funciona a diretoria regional?",), "Como funciona a diretoria regional?"),
        (
            "e quem pode ser reeleito para a mesma função?",
            ("Como funciona a diretoria regional?",),
            "Como funciona a diretoria regional?",
        ),
        # A âncora encadeada: a anterior também era de seguimento; recua até a que tem assunto.
        (
            "e no regulamento?",
            ("Como funciona a diretoria regional?", "e no estatuto?"),
            "Como funciona a diretoria regional?",
        ),
        # Com assunto próprio, não é seguimento.
        ("Qual o mandato do CAN?", ("Como funciona a diretoria regional?",), None),
        # Longa e sem marcador de retomada: pergunta nova.
        ("O que é a flor de lis no escotismo mundial?", ("Como funciona a diretoria regional?",), None),
        # Sem pergunta anterior, não há âncora.
        ("Qual o prazo de gestão?", (), None),
    ],
)
def test_seguimento_e_a_ancora(pergunta: str, anteriores: tuple[str, ...], ancora: str | None) -> None:
    assert derive_context(pergunta, DIA, previous=anteriores).anchor == ancora


def test_seguimento_herda_o_assunto_e_mantem_o_tipo_da_pergunta_atual() -> None:
    contexto = derive_context("e no estatuto nacional?", DIA, previous=(REELEICAO,))
    assert contexto.document_types == ("estatuto",)  # o tipo é o da pergunta atual
    assert contexto.scopes == ("nacional",)
    assert contexto.suggested_bodies == ("diretoria_regional",)  # o assunto veio da âncora
    assert "gestao_do_grupo" in contexto.topics
    filtros = to_slice(contexto)["retrieval_filters"]
    assert isinstance(filtros, dict) and filtros["anchor"] == REELEICAO


def test_seguimento_herda_o_ramo_so_quando_nao_cita_outro() -> None:
    anterior = ("Quais são as especialidades do ramo lobinho?",)
    assert derive_context("e quantas são?", DIA, previous=anterior).branch == "lobinho"
    assert derive_context("e para o ramo pioneiro?", DIA, previous=anterior).branch == "pioneiro"
    # Tema próprio ("uniforme") é assunto: não é seguimento, e nada se herda.
    assert derive_context("e o uniforme?", DIA, previous=anterior).anchor is None


def test_ancora_e_cortada_no_teto() -> None:
    longa = "Como funciona a diretoria regional? " + "detalhe " * 100
    ancora = derive_context("e no estatuto?", DIA, previous=(longa,)).anchor
    assert ancora is not None and len(ancora) == 400


def test_perguntas_anteriores_da_turn_view() -> None:
    from axon_scouting.context.derivation import earlier_questions

    view = turn_view(
        question="e no estatuto nacional?",
        messages=[
            {"role": "user", "content": REELEICAO},
            {"role": "assistant", "content": "Não encontrei."},
            {"role": "user", "content": "e no estatuto nacional?"},
        ],
    )
    assert earlier_questions(view) == (REELEICAO,)


def test_prepare_context_publica_a_ancora() -> None:
    view = turn_view(
        question="e no estatuto nacional?",
        messages=[
            {"role": "user", "content": REELEICAO},
            {"role": "assistant", "content": "Não encontrei."},
            {"role": "user", "content": "e no estatuto nacional?"},
        ],
    )
    fatia = asyncio.run(ScoutingPack().prepare_context(view, pack_context("scouting")))
    filtros = fatia["retrieval_filters"]
    assert isinstance(filtros, dict) and filtros["anchor"] == REELEICAO
    assert filtros["document_types"] == ["estatuto"]


# --- ADR-062: o pedido de síntese ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("pergunta", "anos"),
    [
        (
            "Quem foram os diretores-presidentes eleitos para a Diretoria Regional de São Paulo desde 2016?",
            (2016, 2026),
        ),
        ("Onde se reuniu o Conselho Consultivo Regional entre 2017 e 2019?", (2017, 2019)),
        ("Que chapas disputaram a Diretoria Regional na assembleia de 2016?", (2016, 2016)),
        ("Em que assembleia foi aprovado um novo Regulamento Regional nos anos 2000?", (2000, 2009)),
        ("O que a Assembleia Regional decidiu sobre o quórum?", (None, None)),
    ],
)
def test_pedido_de_sintese(pergunta: str, anos: tuple[int | None, int | None]) -> None:
    sintese = derive_context(pergunta, date(2026, 11, 9)).synthesis
    assert sintese is not None and (sintese.year_from, sintese.year_until) == anos
    assert "regional" in sintese.scopes or "regional" not in pergunta.lower()


@pytest.mark.parametrize(
    "pergunta",
    [
        "Qual o mandato da diretoria regional?",
        "Quem pode ser eleito para a diretoria regional?",
        "Um conselheiro nacional pode ser reeleito?",
        "Em que cidade fica a sede nacional dos Escoteiros do Brasil?",
        "O que diz a resolução 08/2021?",
        REELEICAO,
    ],
)
def test_pergunta_pontual_nao_pede_sintese(pergunta: str) -> None:
    assert derive_context(pergunta, date(2026, 11, 9)).synthesis is None


def test_sintese_na_fatia() -> None:
    fatia = to_slice(derive_context("Onde se reuniu o Conselho Consultivo Regional entre 2017 e 2019?", DIA))
    filtros = fatia["retrieval_filters"]
    assert isinstance(filtros, dict)
    assert filtros["synthesis"] == {
        "year_from": 2017,
        "year_until": 2019,
        "scopes": ["regional"],
        "bodies": ["conselho_consultivo"],
    }


# --- ADR-063, ADR-064: as entidades nomeadas, pelo grafo de domínio, e o plano -------------------------------------

COM_ORGAOS = dataclasses.replace(
    LEXICON,
    entity_terms=MappingProxyType(
        {
            "tipo_orgao:diretoria_regional": ("diretoria regional",),
            "tipo_orgao:conselho_consultivo_nacional": ("conselho consultivo",),
            "tipo_orgao:conselho_consultivo_regional": ("conselho consultivo regional",),
        }
    ),
)


@pytest.mark.parametrize(
    ("pergunta", "orgaos"),
    [
        (REELEICAO, ("tipo_orgao:diretoria_regional",)),
        ("Qual o papel do Conselho Consultivo Regional?", ("tipo_orgao:conselho_consultivo_regional",)),
        ("Como funciona o conselho consultivo?", ("tipo_orgao:conselho_consultivo_nacional",)),
        ("Qual o marco simbólico do ramo lobinho?", ()),
        # Síntese, lista e pergunta histórica não ancoram no grafo.
        ("Quem foram os presidentes eleitos para a diretoria regional desde 2016?", ()),
        ("Quais são as resoluções da diretoria regional?", ()),
        ("Qual era o mandato da diretoria regional em 2010?", ()),
    ],
)
def test_orgaos_pelo_grafo(pergunta: str, orgaos: tuple[str, ...]) -> None:
    contexto = derive_context(pergunta, DIA, COM_ORGAOS)
    assert contexto.entities == orgaos
    # G1 (101 FR-037): um sobre com todas as entidades, nas normas; sem entidade, plano vazio.
    if orgaos:
        assert contexto.plan.to_json() == {
            "version": 1,
            "forma": "simples",
            "consultas": [{"q": "sobre", "args": {"entidades": list(orgaos), "papeis": ["norma"]}}],
        }
    else:
        assert contexto.plan.empty


def test_o_plano_vai_aos_filtros_e_e_valido_pelo_esquema() -> None:
    from axon_scouting.rag.graph.api import QueryPlan

    filtros_ = to_slice(derive_context(REELEICAO, DIA, COM_ORGAOS))["retrieval_filters"]
    assert filtros_["entities"] == ["tipo_orgao:diretoria_regional"]
    assert QueryPlan.from_json(filtros_["plan"]).consultas[0].q == "sobre"  # type: ignore[arg-type]
    assert "plan" not in to_slice(derive_context("Qual o lema do lobinho?", DIA, COM_ORGAOS))["retrieval_filters"]
