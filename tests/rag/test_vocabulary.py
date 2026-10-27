"""Os ids que a 101 publica nos filtros são os valores que o índice da 102 filtra (spec 102 research R3; T049).

Nota para a US5 da 101: se o léxico ganhar ramo ou tipo novo, o modelo do índice precisa acompanhar.
"""

from __future__ import annotations

from axon_scouting.context.lexicon import LEXICON
from axon_scouting.rag.index.api import BRANCHES, DocumentType


def test_ramos_do_lexico_sao_os_do_indice() -> None:
    assert set(LEXICON.branches) == set(BRANCHES)


def test_tipos_do_lexico_sao_os_do_indice() -> None:
    """Todo tipo que a derivação publica existe no índice; e o sugerido por tema ou sinal está entre os citáveis — a
    derivação só publica tipos do léxico (ADR-046: `programa_educativo` saiu do léxico, porque nenhum documento o
    tem)."""
    tipos = {t.value for t in DocumentType}
    citaveis = set(LEXICON.document_types)
    assert citaveis <= tipos
    assert {tipo for topico in LEXICON.topics for tipo in topico.document_types} <= citaveis
    assert {tipo for sinais in LEXICON.type_signals.values() for tipo in sinais} <= citaveis
