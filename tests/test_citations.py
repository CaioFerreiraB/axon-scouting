"""Sintaxe `[[id]]`: extração, chave de comparação e reescrita do texto (spec 104 FR-001, FR-009; R3, R5; T004)."""

from __future__ import annotations

import pytest

from axon_scouting.validators.citations import extract, key, rewrite


def test_extract_na_ordem_e_como_escrito() -> None:
    texto = "Primeiro [[a]], depois [[ a b ]] e [[MAN#1@p2]]; de novo [[a]]."
    assert extract(texto) == ["a", "a b", "MAN#1@p2", "a"]


def test_extract_sem_marcador() -> None:
    assert extract("Nenhuma citação aqui [a] [[]].") == []


@pytest.mark.parametrize(
    ("bruto", "chave"), [(" MAN #1 ", "man#1"), ("man-pio-2026#4.2.1@p87", "man-pio-2026#4.2.1@p87")]
)
def test_key(bruto: str, chave: str) -> None:
    assert key(bruto) == chave


@pytest.mark.parametrize(
    ("texto", "numeros", "esperado"),
    [
        ("Conforme o manual [[a]].", {"a": 1}, "Conforme o manual [1]."),
        ("Conforme o manual [[x]].", {}, "Conforme o manual."),
        ("Um [[a]], outro [[ B ]], de novo [[A]].", {"a": 1, "b": 2}, "Um [1], outro [2], de novo [1]."),
        ("Válida [[a]] e inventada [[x]], fim.", {"a": 1}, "Válida [1] e inventada, fim."),
        ("Sem marcador.", {"a": 1}, "Sem marcador."),
    ],
)
def test_rewrite(texto: str, numeros: dict[str, int], esperado: str) -> None:
    assert rewrite(texto, numeros) == esperado
