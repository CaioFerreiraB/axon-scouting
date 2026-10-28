"""Títulos pela tipografia (spec 102 FR-031; ADR-050)."""

from __future__ import annotations

from axon_scouting.rag.ingestion.domain.layout import TextLine
from axon_scouting.rag.ingestion.domain.typography import headings


def linha(texto: str, tamanho: float, fonte: str = "Avenir-Book", negrito: bool = False) -> TextLine:
    return TextLine(text=texto, left=40, bottom=0, right=300, top=10, size=tamanho, font=fonte, bold=negrito)


def corpo(n: int) -> list[TextLine]:
    return [
        linha(f"Texto corrido do relatório, linha {i}, com bastante conteúdo para ser o corpo.", 10) for i in range(n)
    ]


def pagina(*titulos: TextLine) -> list[TextLine]:
    return [*titulos, *corpo(8)]


def test_o_nome_em_maiusculas_da_fonte_do_corpo_nao_e_titulo() -> None:
    premio = linha("Prêmio Walter Dohme", 14, fonte="GotchaGothic")
    ramo = linha("Ramo Lobinho", 12, fonte="GotchaGothic")
    nome = linha("CAIO FERREIRA BERNARDO", 10)
    paginas = [pagina(premio, ramo, nome), pagina(premio, ramo), pagina(premio, ramo)]
    achados = {(h.text, h.level) for h in headings(paginas)}
    assert ("Prêmio Walter Dohme", 1) in achados and ("Ramo Lobinho", 2) in achados
    assert all(texto != "CAIO FERREIRA BERNARDO" for texto, _ in achados)


def test_sem_estilo_de_titulo_nao_ha_titulo_tipografico() -> None:
    assert headings([corpo(10), corpo(10)]) == ()


def test_destaque_em_varias_linhas_e_frase_no_estilo_nao_sao_titulo() -> None:
    destaque = [linha(f"Os Objetivos de Desenvolvimento, parte {i}", 14, fonte="GotchaGothic") for i in range(5)]
    frase = linha("adequado das tarefas desenvolvidas.", 14, fonte="GotchaGothic")
    titulo = linha("APRESENTAÇÃO", 14, fonte="GotchaGothic")
    paginas = [[*destaque, *corpo(8)], pagina(frase), pagina(titulo), pagina(titulo)]
    assert {h.text for h in headings(paginas)} == {"APRESENTAÇÃO"}
