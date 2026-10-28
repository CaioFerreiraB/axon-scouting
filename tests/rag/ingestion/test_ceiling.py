"""O teto do trecho no embedder (spec 102 FR-032; ADR-050)."""

from __future__ import annotations

from axon_scouting.rag.ingestion.domain.ceiling import fit
from axon_scouting.rag.ingestion.domain.chunking import ChunkDraft


def estimativa(texto: str) -> int:
    return -(-len(texto) // 4)


def test_artigo_longo_vira_partes_com_o_caput_e_as_paginas_de_cada_uma() -> None:
    incisos = [
        f"{n}) O associado zela pelo patrimônio, pela imagem e pela segurança de todos os jovens." for n in range(60)
    ]
    corpo = "\n".join(["Art. 14 - São deveres do associado:", *incisos])
    paginas = (10, *(10 + n // 20 for n in range(60)))
    rascunho = ChunkDraft(
        section_key="art14",
        prefix="[Estatuto · Capítulo 3]",
        body=corpo,
        page_start=10,
        page_end=12,
        article="Art. 14",
        normative=True,
        line_pages=paginas,
    )
    partes = fit(rascunho, estimativa)
    assert len(partes) > 1 and all(estimativa(p.text) <= 512 for p in partes)
    assert partes[1].prefix == f"[Estatuto · Capítulo 3 · Art. 14 (parte 2 de {len(partes)})]"
    assert all(p.body.startswith("Art. 14 - São deveres do associado:") for p in partes)
    assert partes[0].page_start == 10 and partes[-1].page_end == 12
    assert [p.page_start for p in partes] == sorted(p.page_start for p in partes)


def test_trecho_que_cabe_fica_como_esta() -> None:
    rascunho = ChunkDraft(section_key="s1", prefix="[Manual]", body="Texto curto.", page_start=1, page_end=1)
    assert fit(rascunho, estimativa) == [rascunho]
