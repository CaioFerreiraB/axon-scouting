"""A árvore de unidades de um documento (spec 102 FR-025, FR-027, FR-028; ADR-048)."""

from __future__ import annotations

from typing import Any

import pytest

from axon_scouting.rag.ingestion.domain.chunking import ChunkDraft
from axon_scouting.rag.ingestion.domain.corpus_structure import Profile, Resolution, parse_structure
from axon_scouting.rag.ingestion.domain.sources import Bookmark, TocEntry
from axon_scouting.rag.ingestion.domain.tree import TreeProblem, build_tree, shape_of

TITULO = "Guia de Especialidades"


def perfil(nome: str) -> Resolution:
    """Um perfil sem taxonomia nem exceção, com as fontes de sempre (os testes da árvore não dependem da pasta)."""
    fontes = {"artigos": ("marcadores",), "regras": ("marcadores",)}.get(nome, ("títulos",))
    return Resolution(profile=Profile(name=nome, sources=fontes, units=()))


def rascunho(page: int, body: str = "Uma frase qualquer do trecho.", **labels: Any) -> ChunkDraft:
    path = [labels[k] for k in ("chapter", "section", "subsection") if labels.get(k)]
    return ChunkDraft(
        section_key=f"s{page}",
        prefix="[" + " · ".join([TITULO, *path]) + "]",
        body=body,
        page_start=page,
        page_end=page,
        **labels,
    )


def resolucao(
    perfil: str, *, unidades: list[dict[str, Any]] | None = None, branches: list[str] | None = None
) -> Resolution:
    estrutura = {
        "version": 1,
        "unit_kinds": {
            k: {"label": k.capitalize().replace("_", " ")}
            for k in (
                "capitulo",
                "secao",
                "eixo",
                "bloco",
                "especialidade",
                "grupo_de_insignias",
                "insignia",
                "reuniao",
                "item_de_pauta",
            )
        }
        | {"especialidade": {"label": "Especialidade"}, "eixo": {"label": "Eixo"}, "bloco": {"label": "Bloco"}},
        "vocabulary": {
            "eixo": {"Habilidades para a Vida": ["Autonomia e Liderança"], "Meio Ambiente": ["Mudanças Climáticas"]}
        },
        "profiles": {perfil: {"sources": ["sumário", "rodapé", "bookmarks", "títulos"], "units": ["capitulo"]}},
        "profile_by_type": {"Manual": perfil},
        "documents": [{"match": ["doc"], "profile": perfil, "branches": branches or [], "units": unidades or []}],
    }
    taxonomia = {"version": 1, "entries": [{"categoria": "PE", "tipo": "Manual", "pasta": "pe"}]}
    structure, problems = parse_structure(taxonomia, estrutura)
    assert structure is not None, problems
    entry = structure.classify("pe/doc.pdf")
    assert entry is not None
    return structure.resolve("doc", entry)


def test_especialidades_pelo_sumario_com_eixo_do_rodape_e_grupos_com_ramo() -> None:
    resolution = resolucao(
        "especialidades",
        branches=["senior", "pioneiro"],
        unidades=[
            {"kind": "especialidade", "pages": "10-19", "source": "sumário", "parent": "eixo"},
            {
                "kind": "grupo_de_insignias",
                "pages": "20-29",
                "name": "Insígnias — Ramo Pioneiro",
                "branches": ["pioneiro"],
            },
            {"kind": "insignia", "pages": "21-29", "source": "sumário", "parent": "grupo_de_insignias"},
        ],
    )
    toc = [
        TocEntry(title="As Especialidades", page=3),
        TocEntry(title="Gastronomia", page=10),
        TocEntry(title="Viagens", page=14),
        TocEntry(title="Insígnia do Aprender", page=21),
    ]
    running = [()] * 30
    running[9] = running[10] = ("Habilidades para a Vida",)
    running[13] = ("Meio Ambiente",)
    drafts = [
        rascunho(4, chapter="AS ESPECIALIDADES"),
        rascunho(10, chapter="GASTRONOMIA"),
        rascunho(11, chapter="GASTRONOMIA", section="SUGESTÃO DE TEMAS"),
        rascunho(14, chapter="VIAGENS"),
        rascunho(22, chapter="INSÍGNIA DO APRENDER", section="APRENDENDO A APRENDER"),
    ]
    tree = build_tree(drafts, resolution, doc_key="guia-1", title=TITULO, branch=None, toc=toc, running=running)
    caminhos = [[(u.kind, u.title) for u in placed.path] for placed in tree.chunks]
    assert caminhos[0] == [("capitulo", "As Especialidades")]
    assert caminhos[1] == [("eixo", "Habilidades para a Vida"), ("especialidade", "Gastronomia")]
    assert caminhos[2] == [
        ("eixo", "Habilidades para a Vida"),
        ("especialidade", "Gastronomia"),
        ("secao", "SUGESTÃO DE TEMAS"),
    ]
    assert caminhos[3] == [("eixo", "Meio Ambiente"), ("especialidade", "Viagens")]
    assert caminhos[4] == [
        ("grupo_de_insignias", "Insígnias — Ramo Pioneiro"),
        ("insignia", "Insígnia do Aprender"),
        ("secao", "APRENDENDO A APRENDER"),
    ]
    rotulo = tree.chunks[2].draft.prefix
    assert rotulo == f"[{TITULO} · Eixo: Habilidades para a Vida · Especialidade: Gastronomia · SUGESTÃO DE TEMAS]"
    assert tree.chunks[1].unit_kind == "especialidade"
    assert tree.chunks[1].branches == ("senior", "pioneiro")  # o do documento declarado
    assert tree.chunks[4].branches == ("pioneiro",)  # o do grupo vence
    especialidades = [u for u in tree.units if u.kind == "especialidade"]
    assert [(u.title, u.page_start, u.page_end) for u in especialidades] == [
        ("Gastronomia", 10, 11),
        ("Viagens", 14, 14),
    ]
    gastronomia = especialidades[0]
    assert gastronomia.unit_id == "guia-1#u/eixo-habilidades-para-a-vida/especialidade-gastronomia"
    assert next(u for u in tree.units if u.kind == "insignia").branches == ("pioneiro",)


def test_faixa_sem_entrada_no_sumario_recusa() -> None:
    resolution = resolucao(
        "especialidades", unidades=[{"kind": "especialidade", "pages": "50-60", "source": "sumário"}]
    )
    with pytest.raises(TreeProblem, match="50-60"):
        build_tree(
            [rascunho(1)], resolution, doc_key="g-1", title=TITULO, branch=None, toc=[TocEntry(title="A", page=3)]
        )


def test_progressao_eixo_pelo_rodape_e_bloco_pelo_vocabulario() -> None:
    resolution = resolucao("progressao")
    bookmarks = [Bookmark(level=0, title="Progressão Pessoal", page=1), Bookmark(level=1, title="Sub", page=2)]
    running = [("Habilidades para a Vida",), ("Meio Ambiente",), ()]
    drafts = [
        rascunho(1, chapter="AUTONOMIA E LIDERANÇA"),
        rascunho(2, chapter="MUDANÇAS CLIMÁTICAS MEIO AMBIENTE", section="AÇÕES FIXAS"),
        rascunho(3, chapter="VALORES"),  # fora da Progressão: "Valores" não vira bloco
    ]
    tree = build_tree(
        drafts, resolution, doc_key="g-1", title=TITULO, branch="pioneiro", bookmarks=bookmarks, running=running
    )
    caminhos = [[(u.kind, u.title) for u in placed.path] for placed in tree.chunks]
    assert caminhos[0] == [
        ("capitulo", "Progressão Pessoal"),
        ("eixo", "Habilidades para a Vida"),
        ("bloco", "Autonomia e Liderança"),
    ]
    assert caminhos[1] == [
        ("capitulo", "Progressão Pessoal"),
        ("eixo", "Meio Ambiente"),
        ("bloco", "Mudanças Climáticas"),
        ("secao", "AÇÕES FIXAS"),
    ]
    assert caminhos[2] == [("capitulo", "Progressão Pessoal"), ("secao", "VALORES")]
    assert tree.chunks[0].branches == ("pioneiro",)  # o do catálogo


def test_padrao_declarado_tipa_o_titulo() -> None:
    resolution = resolucao("capitulos", unidades=[{"kind": "reuniao", "pattern": r"^\d+\s*[ªa]\s*reuni[ãa]o"}])
    tree = build_tree(
        [rascunho(5, chapter="1ª REUNIÃO DE ALCATEIA", section="JOGO 1")],
        resolution,
        doc_key="pm-1",
        title=TITULO,
        branch=None,
    )
    assert [(u.kind, u.title) for u in tree.chunks[0].path] == [
        ("reuniao", "1ª REUNIÃO DE ALCATEIA"),
        ("secao", "JOGO 1"),
    ]
    assert tree.chunks[0].draft.prefix == f"[{TITULO} · Reuniao: 1ª REUNIÃO DE ALCATEIA · JOGO 1]"


def test_norma_pelos_marcadores_sem_mudar_o_rotulo() -> None:
    draft = ChunkDraft(
        section_key="art12",
        prefix="[Estatuto · Capítulo 3 – Do Nível Nacional · Seção I – Da Composição]",
        body="Art. 12 - São órgãos Nacionais:\nI - a Assembleia Nacional;",
        page_start=9,
        page_end=9,
        chapter="Capítulo 3 – Do Nível Nacional",
        section="Seção I – Da Composição",
        article="Art. 12",
        normative=True,
    )
    tree = build_tree([draft], perfil("artigos"), doc_key="estatuto-2024", title="Estatuto", branch=None)
    placed = tree.chunks[0]
    assert [(u.kind, u.title) for u in placed.path] == [
        ("capitulo", "Capítulo 3 – Do Nível Nacional"),
        ("secao", "Seção I – Da Composição"),
        ("artigo", "Art. 12"),
    ]
    assert placed.draft.prefix == draft.prefix and placed.unit_kind == "artigo"


def test_sem_estrutura_o_rotulo_nao_muda() -> None:
    draft = rascunho(2, chapter="4 Programa", section="4.2 Acampamento")
    tree = build_tree([draft], perfil("capitulos"), doc_key="m-1", title=TITULO, branch=None)
    assert tree.chunks[0].draft.prefix == draft.prefix
    assert [u.kind for u in tree.units] == ["capitulo", "secao"]


def test_comunicado_e_apresentacao() -> None:
    comunicado = build_tree([rascunho(1)], resolucao("comunicado"), doc_key="of-1", title="Ofício", branch=None)
    assert [(u.kind, u.title) for u in comunicado.chunks[0].path] == [("documento", "Ofício")]
    slides = build_tree(
        [rascunho(3), rascunho(4)], resolucao("apresentacao"), doc_key="s-1", title="Slides", branch=None
    )
    assert [u.title for u in slides.units] == ["Slide 3", "Slide 4"]


@pytest.mark.parametrize(
    ("corpo", "forma"),
    [
        ("Os eixos: • Comunicações • Educação • Empreendedorismo e Negócios • Finanças e Economia • Viagens", "list"),
        (
            "Art. 12 - São órgãos Nacionais:\nI - a Assembleia;\nII - o Conselho;\nIII - a Diretoria;\n"
            "IV - a Comissão.",
            "list",
        ),
        ("a) primeiro; b) segundo; c) terceiro; d) quarto", "list"),
        ("O Movimento Escoteiro apoia cada jovem. Ele cresce em grupo, com projetos e responsabilidade.", "prose"),
        (
            "1. Explicar o princípio de sustentação do voo, demonstrando o efeito do ar sobre as asas com um "
            "experimento simples. 2. Identificar as principais partes de uma aeronave em um modelo físico ou "
            "digital, explicando "
            "a função de cada uma. 3. Em um aeródromo identificar os principais tipos de aeronaves e helicópteros. "
            "4. Explicar como funciona o motor a pistão e o motor a jato, mostrando as diferenças de cada um.",
            "prose",
        ),
    ],
)
def test_forma_do_trecho(corpo: str, forma: str) -> None:
    assert shape_of(corpo) == forma


def test_entradas_de_mesmo_titulo_sao_unidades_distintas_e_herdam_os_ramos() -> None:
    resolution = resolucao("capitulos", branches=["lobinho"])
    toc = [TocEntry(title="Reunião de Alcateia", page=3), TocEntry(title="Reunião de Alcateia", page=5)]
    tree = build_tree(
        [rascunho(3), rascunho(4), rascunho(5)], resolution, doc_key="pm-1", title=TITULO, branch=None, toc=toc
    )
    capitulos = [u for u in tree.units if u.kind == "capitulo"]
    assert [(u.page_start, u.page_end) for u in capitulos] == [(3, 4), (5, 5)]
    assert all(u.branches == ("lobinho",) for u in tree.units)  # sem ramo próprio, o do documento declarado


def test_o_eixo_do_bloco_vem_do_vocabulario_e_nao_do_rodape_da_transicao() -> None:
    resolution = resolucao("progressao")
    tree = build_tree(
        [rascunho(1, chapter="AUTONOMIA E LIDERANÇA")],
        resolution,
        doc_key="g-1",
        title=TITULO,
        branch=None,
        running=[("Meio Ambiente",)],  # a página de transição já traz o eixo seguinte
    )
    assert [(u.kind, u.title) for u in tree.chunks[0].path] == [
        ("eixo", "Habilidades para a Vida"),
        ("bloco", "Autonomia e Liderança"),
    ]


def test_item_de_pauta_e_sumario_da_arvore() -> None:
    resolution = resolucao("ata")
    drafts = [rascunho(n, chapter=f"0{n}) ASSUNTO {n}") for n in range(1, 6)]
    tree = build_tree(drafts, resolution, doc_key="ata-1", title="Ata", branch=None)
    assert [u.kind for u in tree.units] == ["item_de_pauta"] * 5
    (sumario,) = [p for p in tree.chunks if p.shape == "outline"]
    assert sumario.draft.section_key == "sumario" and sumario.draft.prefix == "[Ata · Sumário]"
    assert sumario.draft.body.splitlines()[0] == "- Item de pauta: 01) ASSUNTO 1 (p. 1)"


def test_documento_pequeno_sem_sumario() -> None:
    tree = build_tree([rascunho(1, chapter="ÚNICO")], resolucao("capitulos"), doc_key="d-1", title="D", branch=None)
    assert all(p.shape != "outline" for p in tree.chunks)


def test_a_janela_final_curta_do_sumario_junta_se_a_anterior() -> None:
    drafts = [rascunho(n, chapter=f"{n:02d}) ASSUNTO COM UM TÍTULO BEM LONGO NÚMERO {n}") for n in range(1, 26)]
    tree = build_tree(drafts, resolucao("ata"), doc_key="ata-1", title="Ata", branch=None)
    sumarios = [p.draft.body for p in tree.chunks if p.shape == "outline"]
    assert sumarios and all(len(body.splitlines()) >= 5 for body in sumarios)


def test_norma_so_com_artigos_fica_sem_sumario() -> None:
    drafts = [
        ChunkDraft(
            section_key=f"art{n}",
            prefix="[Res]",
            body=f"Art. {n} - Texto.",
            page_start=1,
            page_end=1,
            article=f"Art. {n}",
            normative=True,
        )
        for n in range(1, 8)
    ]
    tree = build_tree(drafts, perfil("artigos"), doc_key="res-1", title="Res", branch=None)
    assert all(p.shape != "outline" for p in tree.chunks)
