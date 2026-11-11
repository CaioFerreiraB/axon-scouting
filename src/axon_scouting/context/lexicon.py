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
    """Tema da pergunta: os termos que o reconhecem e os tipos de documento em que costuma estar — uma preferência de
    ordem, nunca um filtro (ADR-046)."""

    id: str
    terms: tuple[str, ...]
    document_types: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class Lexicon:
    branches: Mapping[str, tuple[str, ...]]
    """Ramo → termos não ambíguos."""
    all_branches: tuple[str, ...]
    """Pergunta sobre todos os ramos ("cada ramo", "os 5 ramos"): as facetas são os cinco (ADR-045)."""
    ambiguous: tuple[str, ...]
    """Termos que lembram um ramo mas também nomeiam o movimento ou outra coisa: nunca filtram."""
    topics: tuple[Topic, ...]
    document_types: Mapping[str, tuple[str, ...]]
    """Tipo de documento (id da taxonomia, ADR-049) → termos que o citam explicitamente. Só o tipo citado filtra
    (ADR-046)."""
    type_signals: Mapping[str, tuple[str, ...]]
    """Termo que aponta tipos de documento sem nomeá-los → esses tipos, como preferência de ordem (ADR-046)."""
    previous_markers: tuple[str, ...]
    """Consulta à versão anterior sem data."""
    list_markers: tuple[str, ...]
    """Pedido de enumeração ("quais", "liste"): com um plural de `catalog_kinds`, a pergunta pede a lista de documentos
    (ADR-047)."""
    catalog_kinds: Mapping[str, tuple[str, ...]]
    """Espécie de documento — o id do tipo na taxonomia (ADR-049) — → os plurais que a pedem. `documento` vale qualquer
    tipo."""
    catalog_stopwords: frozenset[str]
    """Palavras que não qualificam a lista: as de ligação, os verbos da pergunta e o que todo título repete."""
    bodies: Mapping[str, tuple[str, ...]] = MappingProxyType({})
    """Órgão (id da categoria ou subcategoria da taxonomia) → termos. Filtra junto com um tipo citado ou um pedido de
    lista; sozinho, só desempata (ADR-049)."""
    scopes: Mapping[str, tuple[str, ...]] = MappingProxyType({})
    """Abrangência (`nacional`, `regional`…) → termos. Filtra como o órgão: só com tipo citado ou pedido de lista."""
    unit_kinds: Mapping[str, tuple[str, ...]] = MappingProxyType({})
    """Tipo de unidade da árvore dos documentos → os plurais que o pedem ("especialidade" → "especialidades"),
    normalizados. Vêm do índice (os tipos do `estrutura.yaml` na ingestão), postos aqui pela composição do pack
    (ADR-048); o léxico em código não os tem."""
    issuer_links: tuple[str, ...] = ()
    """Palavras que ligam o tipo citado ao órgão que o emitiu ("resoluções **da** diretoria regional"): só com uma
    delas logo depois do tipo o órgão é emissor e filtra (ADR-061)."""
    issuer_fillers: tuple[str, ...] = ()
    """Palavras que podem ficar entre o tipo e a ligação sem desfazer o emissor ("regimento **interno** do conselho");
    a abrangência ("regulamento **regional** da …") também pode."""
    follow_up_markers: tuple[str, ...] = ()
    """Retomada da pergunta anterior em qualquer ponto ("isso", "disso", "nele"): com eles, a pergunta sem assunto
    próprio é de seguimento (ADR-061)."""
    follow_up_openers: tuple[str, ...] = ()
    """Começos que retomam a pergunta anterior ("e no", "e quanto"): só valem no início da pergunta."""
    follow_up_max_words: int = 6
    """Pergunta sem assunto próprio com até esse número de palavras é de seguimento, mesmo sem marcador."""
    synthesis_markers: tuple[str, ...] = ()
    """Pedido de síntese ao longo do acervo ("ao longo", "desde", "quem foram", "onde se reuniu"): a resposta está
    espalhada pelas atas, e a busca de resumos responde (ADR-062)."""
    entity_terms: Mapping[str, tuple[str, ...]] = MappingProxyType({})
    """Entidade do grafo de domínio (`tipo_orgao:…`, `tipo_inst:…`) → os nomes que a reconhecem, normalizados, na ordem
    do catálogo. Vêm do grafo (`DomainGraphSource.aliases`), postos aqui pela composição do pack, como `unit_kinds`
    (ADR-063, ADR-064)."""
    meeting_terms: tuple[str, ...] = ()
    """Palavras de reunião ("assembleia", "reunião", "atas"): com um ano citado, a pergunta é sobre o que as atas
    registram, e também é de síntese (ADR-062)."""
    election_terms: tuple[str, ...] = ()
    """Palavras de eleição ("eleitos", "mandato", "chapa"): na síntese, a linha do tempo fica com os eventos de eleição
    (106 G4)."""
    change_terms: tuple[str, ...] = ()
    """Palavras de mudança ("o que mudou", "que resolução alterou", "foi revogada"): a pergunta quer as resoluções que
    alteram ou revogam o que ela cita (106 G5, US5)."""
    publication_terms: tuple[str, ...] = ()
    """Palavras de publicação ("publicou", "estabeleceu", "normatizou"): com um órgão e um assunto, a pergunta quer os
    documentos do órgão sobre o assunto (106 G5, US5)."""


LEXICON: Final = Lexicon(
    branches=MappingProxyType(
        {
            "filhotes": ("filhote", "filhotes", "ramo filhotes", "ninhada", "ninhadas"),
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
    all_branches=(
        "cada ramo",
        "todos os ramos",
        "os ramos",
        "dos ramos",
        "nos ramos",
        "aos ramos",
        "pelos ramos",
        "entre os ramos",
        "5 ramos",
        "cinco ramos",
        "por ramo",
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
            document_types=("manual",),
        ),
        Topic(id="especialidades", terms=("especialidade", "especialidades"), document_types=("manual",)),
        Topic(
            id="insignias",
            terms=("insignia", "insignias", "distintivo", "distintivos"),
            document_types=("manual", "por"),
        ),
        Topic(
            id="promessa_e_lei",
            terms=("promessa", "promessa escoteira", "lei escoteira", "lei do escoteiro"),
            document_types=("manual", "por"),
        ),
        Topic(id="metodo_escoteiro", terms=("metodo escoteiro",), document_types=("manual",)),
        Topic(
            id="sistema_de_patrulhas", terms=("sistema de patrulhas", "sistema de equipes"), document_types=("manual",)
        ),
        Topic(
            id="uniforme",
            terms=("uniforme", "uniformes", "vestuario", "lenco", "lencos"),
            document_types=("por", "resolucao"),
        ),
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
            document_types=("manual",),
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
    # Os tipos são os da taxonomia do acervo, como id (ADR-049); só os que uma pergunta nomeia sem ambiguidade têm
    # termo — "recomendação", "apresentação", "posicionamento" e "manifestação" são palavras comuns demais.
    document_types=MappingProxyType(
        {
            # "guia" sozinho também nomeia a escoteira do Ramo Sênior: só a forma de título casa; os guias são Manual na
            # taxonomia.
            "manual": ("manual", "manuais", "guia do", "guia da", "guia de", "guia para"),
            # A sigla "POR" normalizada é a preposição "por": só o nome por extenso e a sigla com pontos casam.
            "por": ("principios organizacao e regras", "p o r"),
            "resolucao": ("resolucao", "resolucoes", "instrucao normativa", "instrucoes normativas"),
            "estatuto": ("estatuto", "estatutos"),
            "regulamento": ("regulamento", "regulamentos", "regimento", "regimentos"),
            "ata": ("ata", "atas"),
            "politica": ("politica", "politicas"),
            "relatorio": ("relatorio", "relatorios"),
            "planejamento_estrategico": ("planejamento estrategico",),
            "parecer": ("parecer da", "parecer do", "pareceres"),
            "oficio": ("oficio", "oficios"),
            "edital": ("edital", "editais"),
            "boletim": ("boletim", "boletins"),
        }
    ),
    type_signals=MappingProxyType(
        {
            "competencia": ("manual",),
            "competencias": ("manual",),
            "regra": ("por", "resolucao", "regulamento"),
            "regras": ("por", "resolucao", "regulamento"),
            "norma": ("por", "resolucao", "regulamento"),
            "normas": ("por", "resolucao", "regulamento"),
        }
    ),
    # Órgãos: a categoria (ou a subcategoria) da taxonomia, como id (ADR-049). "Assembleia" sozinha não entra: a
    # assembleia de grupo é assunto do POR, não um órgão com atas no acervo.
    bodies=MappingProxyType(
        {
            "assembleias": (
                "assembleia nacional",
                "assembleias nacionais",
                "assembleia regional",
                "assembleias regionais",
            ),
            "conselho_de_administracao_nacional": (
                "can",
                "conselho de administracao",
                "conselho de administracao nacional",
            ),
            "diretoria_executiva_nacional": ("den", "diretoria executiva", "diretoria executiva nacional"),
            "diretoria_regional": ("diretoria regional", "diretorias regionais"),
            "conselho_consultivo": ("conselho consultivo",),
            "comissao_fiscal": ("comissao fiscal", "comissoes fiscais"),
            "comissao_de_etica": ("comissao de etica", "cedn", "cedr"),
            "conselho_jovem": ("conselho jovem",),
            "conselho_de_velhos_lobos": ("velhos lobos",),
            "rede_de_jovens_lideres": (
                "rede de jovens lideres",
                "rede nacional de jovens lideres",
                "rnjl",
                "forum de jovens lideres",
                "forum nacional de jovens lideres",
                "forum regional de jovens lideres",
                "fnjl",
                "frjl",
            ),
            "comissao_regional_pioneira": (
                "forum pioneiro",
                "forum regional pioneiro",
                "comissao regional pioneira",
                "comissao pioneira",
                "assembleia regional pioneira",
                "arpi",
                "frpio",
            ),
        }
    ),
    # Abrangência (ADR-049): o nível da taxonomia; a Região de São Paulo é o único regional do acervo.
    scopes=MappingProxyType(
        {
            "nacional": ("nacional", "nacionais"),
            "regional": ("regional", "regionais", "sao paulo", "sp", "paulista", "paulistas"),
            "local": ("local", "locais"),
            "mundial": ("mundial", "omme", "wosm"),
            "interamericano": ("interamericano", "interamericana"),
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
    list_markers=(
        "quais",
        "liste",
        "listar",
        "lista",
        "relacione",
        "relacao",
        "enumere",
        "existem",
        "todas as",
        "todos os",
    ),
    issuer_links=("da", "do", "das", "dos", "de"),
    issuer_fillers=("interno", "interna", "geral"),
    follow_up_markers=(
        "isso",
        "disso",
        "nisso",
        "esse",
        "essa",
        "desse",
        "dessa",
        "nesse",
        "nessa",
        "deste",
        "desta",
        "neste",
        "nesta",
        "nele",
        "nela",
        "dele",
        "dela",
        "o mesmo",
        "a mesma",
        "tambem",
    ),
    follow_up_openers=(
        "e no",
        "e na",
        "e nos",
        "e nas",
        "e o",
        "e a",
        "e os",
        "e as",
        "e se",
        "e quanto",
        "e para",
        "e em",
        "e qual",
        "e quais",
        "e quem",
        "e como",
        "e quando",
        "mas e",
        "olhe no",
        "olhe na",
        "veja no",
        "veja na",
        "procure no",
        "procure na",
    ),
    synthesis_markers=(
        "ao longo",
        "desde",
        "evolucao",
        "evoluiu",
        "historico",
        "historia",
        "quem foram",
        "quais foram",
        "o que foi decidido",
        "o que foi discutido",
        "o que foi aprovado",
        "decidiu",
        "decidiram",
        "deliberou",
        "deliberaram",
        "decisoes",
        "deliberacoes",
        "onde se reuniu",
        "onde se reuniram",
        "onde aconteceram",
        "onde aconteceu",
        "onde foram realizadas",
        "onde foi realizada",
        "em que assembleia",
        "em que reuniao",
        "nas atas",
        "quantos delegados",
        "quantos votos",
    ),
    meeting_terms=(
        "assembleia",
        "assembleias",
        "reuniao",
        "reunioes",
        "conselho consultivo",
        "ata",
        "atas",
        "gestao",
        "eleito",
        "eleita",
        "eleitos",
        "eleicao",
        "eleicoes",
        "forum",
        "foruns",
    ),
    election_terms=(
        "eleito",
        "eleita",
        "eleitos",
        "eleitas",
        "eleicao",
        "eleicoes",
        "eleger",
        "elegeu",
        "elegeram",
        "mandato",
        "mandatos",
        "chapa",
        "chapas",
    ),
    change_terms=(
        "mudou",
        "mudaram",
        "o que mudou",
        "alterou",
        "alteraram",
        "foi alterada",
        "foi alterado",
        "foram alteradas",
        "foram alterados",
        "revogou",
        "revogaram",
        "foi revogada",
        "foi revogado",
        "que resolucao",
        "qual resolucao",
        "quais resolucoes",
    ),
    publication_terms=(
        "publicou",
        "publicaram",
        "estabeleceu",
        "estabeleceram",
        "emitiu",
        "editou",
        "normatizou",
        "regulamentou",
        "determinou",
        "descreveu",
    ),
    catalog_kinds=MappingProxyType(
        {
            "politica": ("politicas",),
            "manual": ("manuais", "guias"),
            "resolucao": ("resolucoes", "instrucoes normativas"),
            "ata": ("atas",),
            "regulamento": ("regulamentos", "regimentos"),
            "relatorio": ("relatorios",),
            "edital": ("editais",),
            "boletim": ("boletins",),
            "parecer": ("pareceres",),
            "oficio": ("oficios",),
            "documento": ("documentos", "publicacoes"),
        }
    ),
    catalog_stopwords=frozenset(
        [
            "a",
            "o",
            "as",
            "os",
            "um",
            "uma",
            "uns",
            "umas",
            "de",
            "da",
            "do",
            "das",
            "dos",
            "em",
            "na",
            "no",
            "nas",
            "nos",
            "para",
            "por",
            "pelo",
            "pela",
            "pelos",
            "pelas",
            "com",
            "sem",
            "sobre",
            "e",
            "ou",
            "que",
            "qual",
            "quais",
            "quem",
            "quando",
            "onde",
            "como",
            "sao",
            "ser",
            "existem",
            "existe",
            "ha",
            "tem",
            "temos",
            "tenho",
            "ter",
            "falam",
            "fala",
            "tratam",
            "trata",
            "abordam",
            "aborda",
            "dizem",
            "diz",
            "mencionam",
            "menciona",
            "estao",
            "esta",
            "atuais",
            "atual",
            "vigentes",
            "vigente",
            "disponiveis",
            "disponivel",
            "me",
            "voce",
            "pode",
            "podem",
            "liste",
            "listar",
            "lista",
            "relacione",
            "relacao",
            "enumere",
            "todos",
            "todas",
            "todo",
            "toda",
            "cada",
            "seu",
            "sua",
            "seus",
            "suas",
            "escoteiros",
            "escoteiro",
            "escoteira",
            "escoteiras",
            "brasil",
            "ueb",
            "uniao",
            "movimento",
            "oficiais",
            "oficial",
        ]
    ),
)
"""Semente proposta em 2026-10-07 para revisão do mantenedor (clarificação 5)."""
