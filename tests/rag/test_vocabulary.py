"""Os ids que a 101 publica nos filtros são os valores que o índice da 102 filtra (spec 102 research R3; T049).

Nota para a US5 da 101: se o léxico ganhar ramo ou tipo novo, o modelo do índice precisa acompanhar.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from axon_scouting.context.lexicon import LEXICON
from axon_scouting.rag.index.api import BRANCHES, SCOPES, taxonomy_id

TAXONOMIA = yaml.safe_load((Path(__file__).resolve().parents[2] / "corpus" / "taxonomia.yaml").read_text("utf-8"))


def test_ramos_do_lexico_sao_os_do_indice() -> None:
    assert set(LEXICON.branches) == set(BRANCHES)


def test_tipos_do_lexico_sao_os_do_indice() -> None:
    """Todo tipo que a derivação publica é um tipo da taxonomia (ADR-049); e o sugerido por tema ou sinal está entre os
    citáveis — a derivação só publica tipos do léxico (ADR-046)."""
    tipos = {taxonomy_id(entrada["tipo"]) for entrada in TAXONOMIA["entries"]}
    citaveis = set(LEXICON.document_types)
    assert citaveis <= tipos
    assert {tipo for topico in LEXICON.topics for tipo in topico.document_types} <= citaveis
    assert {tipo for sinais in LEXICON.type_signals.values() for tipo in sinais} <= citaveis


def test_orgaos_e_abrangencias_do_lexico_existem_na_taxonomia() -> None:
    """ADR-049: um órgão do léxico é a categoria ou a subcategoria de alguma pasta; a abrangência, um dos níveis."""
    orgaos = {
        taxonomy_id(entrada[campo])
        for entrada in TAXONOMIA["entries"]
        for campo in ("categoria", "subcategoria")
        if entrada.get(campo)
    }
    assert set(LEXICON.bodies) <= orgaos
    assert set(LEXICON.scopes) <= set(SCOPES)
    assert set(LEXICON.catalog_kinds) - {"documento"} <= set(LEXICON.document_types)
