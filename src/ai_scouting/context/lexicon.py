"""Léxico de domínio da derivação de contexto (spec 101 FR-007, research R7).

Dado em código, versionado com o pack e curado pelo mantenedor. Todos os termos estão **normalizados** (minúsculas, sem
acento, palavras separadas por um espaço — `context.text.normalize`), e os ids são ASCII com `_`, na forma dos
metadados que o índice (102) vai gravar. A ordem das tuplas é a ordem de saída da derivação.

Plural, feminino e sinônimos são termos explícitos: não há stemmer. Termos que nomeiam o movimento e um ramo ao mesmo
tempo ("escoteiro") ficam em `ambiguous` e nunca em um ramo — na dúvida, a derivação não filtra por ramo.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final


@dataclass(frozen=True, slots=True, kw_only=True)
class Topic:
    """Tema da pergunta: os termos que o reconhecem e os tipos de documento em que costuma estar."""

    id: str
    terms: tuple[str, ...]
    document_types: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class Lexicon:
    branches: Mapping[str, tuple[str, ...]]
    """Ramo → termos não ambíguos."""
    ambiguous: tuple[str, ...]
    """Termos que lembram um ramo mas também nomeiam o movimento ou outra coisa: nunca filtram."""
    topics: tuple[Topic, ...]
    document_types: Mapping[str, tuple[str, ...]]
    """Tipo de documento → termos que o citam explicitamente."""
    type_signals: Mapping[str, tuple[str, ...]]
    """Termo que aponta tipos de documento sem nomeá-los → esses tipos."""
    previous_markers: tuple[str, ...]
    """Consulta à versão anterior sem data."""


LEXICON: Final = Lexicon(
    branches=MappingProxyType(
        {
            "lobinho": (
                "lobinho",
                "lobinhos",
                "lobinha",
                "lobinhas",
                "lobito",
                "lobitos",
                "ramo lobinho",
                "alcateia",
                "alcateias",
                "akela",
            ),
            "escoteiro": ("ramo escoteiro", "tropa escoteira", "tropas escoteiras"),
            "senior": (
                "senior",
                "seniores",
                "ramo senior",
                "tropa senior",
                "tropas seniores",
            ),
            "pioneiro": (
                "pioneiro",
                "pioneiros",
                "pioneira",
                "pioneiras",
                "ramo pioneiro",
                "cla",
                "clas",
                "cla pioneiro",
                "rover",
                "rovers",
            ),
        }
    ),
    ambiguous=(
        "escoteiro",
        "escoteiros",
        "escoteira",
        "escoteiras",
        "tropa",
        "tropas",
        "patrulha",
        "patrulhas",
        "guias",
    ),
    topics=(
        Topic(id="lideranca", terms=("lideranca", "lider", "lideres", "liderar"), document_types=("manual",)),
        Topic(
            id="progressao",
            terms=("progressao", "progressao pessoal"),
            document_types=("manual", "programa_educativo"),
        ),
        Topic(id="especialidades", terms=("especialidade", "especialidades"), document_types=("programa_educativo",)),
        Topic(
            id="insignias",
            terms=("insignia", "insignias", "distintivo", "distintivos"),
            document_types=("programa_educativo",),
        ),
        Topic(
            id="promessa_e_lei",
            terms=("promessa", "promessa escoteira", "lei escoteira", "lei do escoteiro"),
            document_types=("manual", "por"),
        ),
        Topic(id="metodo_escoteiro", terms=("metodo escoteiro",), document_types=("manual", "programa_educativo")),
        Topic(
            id="sistema_de_patrulhas", terms=("sistema de patrulhas", "sistema de equipes"), document_types=("manual",)
        ),
        Topic(id="uniforme", terms=("uniforme", "uniformes", "vestuario", "lenco", "lencos"), document_types=("por",)),
        Topic(
            id="cerimonias",
            terms=("cerimonia", "cerimonias", "investidura", "cerimonia de passagem"),
            document_types=("manual",),
        ),
        Topic(
            id="acampamento",
            terms=("acampamento", "acampamentos", "acampar", "atividade ao ar livre", "atividades ao ar livre"),
            document_types=("manual",),
        ),
        Topic(
            id="seguranca",
            terms=("seguranca", "primeiros socorros", "risco", "riscos"),
            document_types=("manual", "por"),
        ),
        Topic(id="pioneirismo", terms=("pioneirismo", "amarra", "amarras"), document_types=("manual",)),
        Topic(
            id="servico_comunitario",
            terms=("servico comunitario", "acao comunitaria", "boa acao"),
            document_types=("manual", "programa_educativo"),
        ),
        Topic(id="meio_ambiente", terms=("meio ambiente", "natureza", "ambiental"), document_types=("manual",)),
        Topic(
            id="adultos",
            terms=("escotista", "escotistas", "dirigente", "dirigentes", "formacao de adultos"),
            document_types=("por", "manual"),
        ),
        Topic(
            id="gestao_do_grupo",
            terms=("grupo escoteiro", "diretoria", "assembleia", "registro", "registros", "anuidade"),
            document_types=("por", "estatuto", "resolucao"),
        ),
    ),
    document_types=MappingProxyType(
        {
            "manual": ("manual", "manuais"),
            "programa_educativo": ("programa educativo", "programa de jovens"),
            # A sigla "POR" normalizada é a preposição "por": só o nome por extenso e a sigla com pontos casam.
            "por": ("principios organizacao e regras", "p o r"),
            "resolucao": ("resolucao", "resolucoes"),
            "estatuto": ("estatuto", "estatutos"),
            # "guia" sozinho também nomeia a escoteira do Ramo Sênior: só a forma de título casa.
            "guia": ("guia do", "guia da", "guia para"),
        }
    ),
    type_signals=MappingProxyType(
        {
            "competencia": ("programa_educativo", "manual"),
            "competencias": ("programa_educativo", "manual"),
            "regra": ("por", "resolucao"),
            "regras": ("por", "resolucao"),
            "norma": ("por", "resolucao"),
            "normas": ("por", "resolucao"),
            "regulamento": ("por", "resolucao"),
        }
    ),
    previous_markers=(
        "manual anterior",
        "versao anterior",
        "edicao anterior",
        "regra anterior",
        "antigamente",
        "antes da mudanca",
    ),
)
"""Semente proposta em 2026-10-07 para revisão do mantenedor (clarificação 5)."""
