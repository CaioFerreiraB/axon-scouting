"""Taxonomia e estrutura mínimas das suítes da ingestão (ADR-049: o tipo e o perfil vêm da taxonomia, pela pasta).

O arquivo na raiz do corpus é um Manual; numa pasta com o nome do tipo antigo (`por/`, `estatuto/`, `resolucao/`), o
tipo correspondente da taxonomia. `pasta(tipo, arquivo)` monta o caminho.
"""

from __future__ import annotations

from typing import Any

TIPOS = {"manual": "Manual", "por": "P.O.R.", "estatuto": "Estatuto", "resolucao": "Resolução", "ata": "Ata"}

TAXONOMIA: dict[str, Any] = {
    "version": 1,
    "entries": [
        {"categoria": "Testes", "tipo": "Manual", "pasta": "."},
        {"categoria": "Testes", "subcategoria": "Manuais", "tipo": "Manual", "pasta": "manuais"},
        {"categoria": "Testes", "subcategoria": "Normas", "nivel": "Nacional", "tipo": "P.O.R.", "pasta": "por"},
        {"categoria": "Testes", "nivel": "Nacional", "tipo": "Estatuto", "pasta": "estatuto"},
        {"categoria": "Testes", "nivel": "Nacional", "tipo": "Resolução", "pasta": "resolucao"},
        {"categoria": "Assembleias", "nivel": "Regional", "tipo": "Ata", "pasta": "ata"},
    ],
}

ESTRUTURA: dict[str, Any] = {
    "version": 1,
    "unit_kinds": {
        "documento": {"plural": "documentos"},
        "capitulo": {"label": "Capítulo", "plural": "capítulos"},
        "secao": {"label": "Seção", "plural": "seções"},
        "regra": {"label": "Regra", "plural": "regras"},
        "artigo": {"label": "Artigo", "plural": "artigos"},
        "item_de_pauta": {"label": "Item de pauta", "plural": "itens de pauta"},
    },
    "profiles": {
        "regras": {"sources": ["marcadores"], "units": ["capitulo", "regra"]},
        "artigos": {"sources": ["marcadores"], "units": ["capitulo", "secao", "artigo"]},
        "capitulos": {"sources": ["títulos"], "units": ["capitulo", "secao"]},
        "ata": {"sources": ["marcadores", "títulos"], "units": ["item_de_pauta"]},
    },
    "profile_by_type": {
        "Manual": "capitulos",
        "P.O.R.": "regras",
        "Estatuto": "artigos",
        "Resolução": "artigos",
        "Ata": "ata",
    },
}

STRUCTURE: tuple[object, object] = (TAXONOMIA, ESTRUTURA)


def pasta(tipo: str, arquivo: str) -> str:
    """O caminho do arquivo do tipo (`manual` fica na raiz)."""
    return arquivo if tipo == "manual" else f"{tipo}/{arquivo}"
