"""Invariantes do léxico (spec 101 FR-007; data-model "Lexicon"; T007)."""

from __future__ import annotations

import re
from collections import Counter

from axon_scouting.context.lexicon import LEXICON
from axon_scouting.context.text import normalize

ID = re.compile(r"^[a-z][a-z0-9_]*$")


def _all_terms() -> list[str]:
    terms = [term for terms in LEXICON.branches.values() for term in terms]
    terms += list(LEXICON.ambiguous)
    terms += [term for topic in LEXICON.topics for term in topic.terms]
    terms += [term for terms in LEXICON.document_types.values() for term in terms]
    terms += list(LEXICON.type_signals)
    terms += list(LEXICON.previous_markers)
    return terms


def test_todo_termo_ja_esta_normalizado() -> None:
    for term in _all_terms():
        assert term and normalize(term) == term, term


def test_nenhum_termo_em_dois_ramos() -> None:
    counts = Counter(term for terms in LEXICON.branches.values() for term in terms)
    assert [term for term, count in counts.items() if count > 1] == []


def test_termo_ambiguo_nunca_filtra_ramo() -> None:
    branch_terms = {term for terms in LEXICON.branches.values() for term in terms}
    assert branch_terms.isdisjoint(LEXICON.ambiguous)


def test_tipos_citados_existem() -> None:
    known = set(LEXICON.document_types)
    cited = {kind for topic in LEXICON.topics for kind in topic.document_types}
    cited |= {kind for kinds in LEXICON.type_signals.values() for kind in kinds}
    assert cited <= known, cited - known


def test_ids_unicos_e_ascii() -> None:
    ids = [*LEXICON.branches, *(topic.id for topic in LEXICON.topics), *LEXICON.document_types]
    assert len(ids) == len(set(ids))
    assert all(ID.match(identifier) for identifier in ids), ids


def test_ramos_e_tipos_da_spec() -> None:
    assert tuple(LEXICON.branches) == ("filhotes", "lobinho", "escoteiro", "senior", "pioneiro")
    assert tuple(LEXICON.document_types) == ("manual", "por", "resolucao", "estatuto", "guia")
