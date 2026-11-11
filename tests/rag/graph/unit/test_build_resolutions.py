"""As resoluções no build do grafo (spec 106 G5, US5; research R22): ementa, número, `ALTERA`, `REVOGA`, `CITA` e
`SUBSTITUI`, por regra."""

from __future__ import annotations

from typing import Any

from axon_scouting.rag.graph.domain.build.resolutions import (
    ementa,
    lineage_edges,
    norm_units,
    number_aliases,
    number_of,
    resolution_links,
    resolutions_of,
)

POR = "por-2025"
ESTATUTO = "estatuto-2024"


def _doc(
    doc_key: str, title: str, effective: str, body: str, page: str = "", status: str = "vigente"
) -> dict[str, Any]:
    return {
        "doc_key": doc_key,
        "fields": {
            "document_id": doc_key.rsplit("-", 1)[0],
            "document_type": "resolucao",
            "document_title": title,
            "effective_date": effective,
            "status": status,
            "bodies": [body],
        },
        "pages": [page],
        "chunk_ids": [f"{doc_key}#preambulo@p1"],
    }


def _chunk(chunk_id: str, text: str, path: list[tuple[str, str, str]] | None = None, **extra: Any) -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "doc_key": chunk_id.split("#")[0],
        "text": text,
        "path": [{"unit_id": u, "kind": k, "title": t} for u, k, t in path or []],
        **extra,
    }


def _rule(number: int, title: str) -> dict[str, Any]:
    unit = f"{POR}#u/regra-{number:03d}"
    return _chunk(
        f"{POR}#regra{number}@p1",
        f"[POR · Regra {number:03d}]\nREGRA {number:03d} {title}\nTexto da regra.",
        [(unit, "regra", f"Regra {number:03d}")],
        effective_date="2025-04-01",
    )


NORM_CHUNKS = [
    _rule(57, "IMPLANTAÇÃO DA SEÇÃO DO RAMO FILHOTES"),
    _rule(178, "RECONHECIMENTOS DE RAMOS"),
    _rule(183, "DISTINTIVOS DE IDIOMAS"),
    _rule(134, "SÍMBOLOS, CERTIFICAÇÕES E REVALIDAÇÃO DA FORMAÇÃO DE ADULTOS"),
    _chunk(
        f"{ESTATUTO}#art27@p19",
        "Art. 27 – Compete à Diretoria Regional…",
        [(f"{ESTATUTO}#u/art-27", "artigo", "Art. 27")],
        effective_date="2024-01-01",
    ),
]
RULES = norm_units(NORM_CHUNKS, POR, "regra")
ARTICLES = norm_units(NORM_CHUNKS, ESTATUTO, "artigo")


def test_ementa_pela_regra() -> None:
    assert (
        ementa(
            "RESOLUÇÃO CAN 03/2021 \r\nImplementa a iniciativa “Tribo da Terra” \r\nnos Escoteiros do Brasil\r\n"
            "Considerando: \r\na) A importância"
        )
        == "Implementa a iniciativa “Tribo da Terra” nos Escoteiros do Brasil"
    )
    assert ementa("– 1/5 –\nRESOLUÇÃO Nº 001/97\nREGULAMENTA A ORGANIZAÇÃO \nDE DELEGAÇÕES\nCONSIDERANDO:\na) que") == (
        "REGULAMENTA A ORGANIZAÇÃO DE DELEGAÇÕES"
    )
    # a instrução com a data no cabeçalho, e a fórmula de quem resolve no lugar do "Considerando"
    assert (
        ementa(
            "Escritório Regional\nINSTRUÇÃO NORMATIVA 2024.02\nGESTÃO DE ADULTOS\nEstabelece normas complementares "
            "para as Iniciativas de Formação.\nA DIRETORIA REGIONAL, no uso das atribuições"
        )
        == "GESTÃO DE ADULTOS Estabelece normas complementares para as Iniciativas de Formação."
    )
    assert ementa("Resolução 05/2020 Altera a Regra 170 do POR\nConsiderando") == "Altera a Regra 170 do POR"
    assert ementa("Um ofício qualquer, sem cabeçalho de resolução.") is None


def test_numero_e_apelidos() -> None:
    assert number_of("Resolução CAN 15/2025", "2025-11-29") == ("can", 2025, 15)
    assert number_of("Resolução Regional SP 03/2025 — Prorroga ATV Distrital", None) == ("regional", 2025, 3)
    assert number_of("Instrução Normativa 2024.02 – Estabelece normas", None) == ("in", 2024, 2)
    assert number_of("Instrução Normativa 002 – concessão da insígnia", "2017-05-29") == ("in", 2017, 2)
    aliases = number_aliases(("can", 2025, 15))
    assert "resolucao can 15 2025" in aliases and "resolucao can no 15 2025" in aliases
    assert {"resolucao can 3 2021", "resolucao can 03 2021", "resolucao can 003 2021"} <= set(
        number_aliases(("can", 2021, 3))
    )
    assert "resolucao regional 2025 03" in number_aliases(("regional", 2025, 3))


def test_altera_a_regra_pelo_titulo_mesmo_renumerada() -> None:
    """CAN 11/2021 "altera a Regra 178" reproduzindo o título: no POR vigente, é a Regra 183 (Distintivos de Idiomas),
    e a 178 é outra regra."""
    docs = [_doc("can-2021-11-1", "Resolução CAN 11/2021", "2021-12-17", "conselho_de_administracao_nacional")]
    chunks = [
        _chunk(
            "can-2021-11-1#art1@p1",
            "[Resolução CAN 11/2021]\nArt. 1º Alterar a Regra 178 do P.O.R., que passa a vigorar com a seguinte "
            "redação:\nREGRA 178 – DISTINTIVO ADICIONAL DE IDIOMAS Ao jovem que conquistar a Especialidade de LIBRAS",
            [("can-2021-11-1#u/art-1", "artigo", "Art. 1")],
        )
    ]
    links = resolution_links(resolutions_of(docs), chunks, RULES, ARTICLES)
    (edge,) = [e for e in links.edges if e["type"] == "ALTERA"]
    assert (edge["from"], edge["to"], edge["source"]) == (
        "can-2021-11-1",
        f"{POR}#u/regra-183",
        "can-2021-11-1#art1@p1",
    )
    assert edge["attrs"] == {"regra": "178", "casamento": "titulo"}
    assert not [e for e in links.edges if e["type"] == "CITA"]  # a regra alterada não vira citação


def test_o_titulo_acaba_no_primeiro_item_e_nao_no_e() -> None:
    docs = [_doc("can-2021-05-1", "Resolução CAN 05/2021", "2021-06-26", "conselho_de_administracao_nacional")]
    chunks = [
        _chunk(
            "can-2021-05-1#art2@p3",
            "Alterar a Regra 131 do POR:\nREGRA 131 SÍMBOLOS DA FORMAÇÃO DE ADULTOS I – O Anel de Gilwell",
        )
    ]
    (edge,) = resolution_links(resolutions_of(docs), chunks, RULES, ARTICLES).edges
    assert edge["to"] == f"{POR}#u/regra-134"


def test_sem_titulo_so_pelo_numero_quando_posterior_ao_por() -> None:
    docs = [
        _doc("can-2025-15-1", "Resolução CAN 15/2025", "2025-11-29", "conselho_de_administracao_nacional"),
        _doc("can-2019-10-1", "Resolução CAN 10/2019", "2019-10-24", "conselho_de_administracao_nacional"),
    ]
    chunks = [
        _chunk("can-2025-15-1#art1@p1", "Art. 1º Alterar a redação da Regra 057 do POR."),
        _chunk("can-2019-10-1#s1@p1", "RESOLVE alterar as regras 130, 131 e 132 do POR."),
    ]
    links = resolution_links(resolutions_of(docs), chunks, RULES, ARTICLES)
    assert [(e["from"], e["to"], e["attrs"]["casamento"]) for e in links.edges] == [
        ("can-2025-15-1", f"{POR}#u/regra-057", "numero")
    ]
    assert len(links.dropped) == 3 and all("antes do POR vigente" in d for d in links.dropped)


def test_revoga_e_altera_resolucoes_pelo_numero() -> None:
    docs = [
        _doc("can-2022-10-1", "Resolução CAN 10/2022", "2022-10-01", "conselho_de_administracao_nacional"),
        _doc("can-2019-09-1", "Resolução CAN 09/2019", "2019-11-13", "conselho_de_administracao_nacional"),
        _doc("sp-2025-01-1", "Resolução Regional SP 01/2025 — Revoga RR", "2025-01-10", "diretoria_regional"),
        _doc("sp-2024-11-1", "Resolução Regional SP 11/2024", "2024-11-01", "diretoria_regional"),
        _doc("can-2015-05-1", "Resolução CAN 05/2015", "2015-05-01", "conselho_de_administracao_nacional"),
        _doc("can-2019-01-1", "Resolução CAN 01/2019", "2019-03-01", "conselho_de_administracao_nacional"),
    ]
    chunks = [
        _chunk("can-2022-10-1#art5@p1", "Esta Resolução entra em vigor e revoga a Resolução CAN 009/2019."),
        _chunk("sp-2025-01-1#art1@p1", "Art. 1º Revogar a Resolução Regional 2024.11, que cria a comissão."),
        # a passiva inverte o sentido: quem altera é a 01/2019
        _chunk("can-2015-05-1#art12@p4", "Alterado conforme Resolução Nº 01/2019 § 2º - A COMENDA"),
        # a mais antiga não altera a mais nova
        _chunk("can-2015-05-1#art20@p5", "Fica alterada a Resolução CAN 01/2019 no que couber."),
    ]
    links = resolution_links(resolutions_of(docs), chunks, RULES, ARTICLES)
    assert [(e["type"], e["from"], e["to"]) for e in links.edges] == [
        ("REVOGA", "can-2022-10-1", "can-2019-09-1"),
        ("REVOGA", "sp-2025-01-1", "sp-2024-11-1"),
        ("ALTERA", "can-2019-01-1", "can-2015-05-1"),
    ]
    assert any("anterior à de destino" in d for d in links.dropped)


def test_cita_a_regra_e_o_artigo_vigentes() -> None:
    docs = [
        _doc("den-2025-03-1", "Resolução DEN 03/2025", "2025-06-01", "diretoria_executiva_nacional"),
        _doc("den-2020-01-1", "Resolução DEN 01/2020", "2020-01-01", "diretoria_executiva_nacional"),
    ]
    article = [("den-2025-03-1#u/art-2", "artigo", "Art. 2")]
    chunks = [
        _chunk("den-2025-03-1#art2@p1", "Conforme a Regra 057 e o art. 27 do Estatuto, a seção…", article),
        _chunk(
            "den-2020-01-1#art2@p1", "Conforme a Regra 057 do POR…", [("den-2020-01-1#u/art-2", "artigo", "Art. 2")]
        ),
    ]
    links = resolution_links(resolutions_of(docs), chunks, RULES, ARTICLES)
    assert [(e["type"], e["from"], e["to"]) for e in links.edges] == [
        ("CITA", "den-2025-03-1#u/art-2", f"{POR}#u/regra-057"),
        ("CITA", "den-2025-03-1#u/art-2", f"{ESTATUTO}#u/art-27"),
    ]  # a de 2020 é anterior ao POR vigente: a numeração pode ser outra
    assert links.units == {"den-2025-03-1#u/art-2", f"{POR}#u/regra-057", f"{ESTATUTO}#u/art-27"}


def test_substitui_pela_linhagem() -> None:
    def version(doc_key: str, document_id: str, effective: str) -> dict[str, Any]:
        return {"doc_key": doc_key, "fields": {"document_id": document_id, "effective_date": effective}}

    edges = lineage_edges(
        [version("por-2021", "por", "2021-01-01"), version("por-2025", "por", "2025-04-01"), version("x-1", "x", "")]
    )
    assert [(e["type"], e["from"], e["to"], e["source"]) for e in edges] == [
        ("SUBSTITUI", "por-2025", "por-2021", "corpus.yaml:por")
    ]


def test_ementa_com_o_numero_na_linha_de_baixo_e_lixo_antes() -> None:
    assert ementa(
        "Escritório Nacional\nRESOLUÇÃO No\n002/2014\nAltera redação da Resolução No 005/2013\nCONSIDERANDO"
    ) == ("Altera redação da Resolução No 005/2013")
    assert ementa("\u2018'RESOLUÇÃO REGIONAL 2025.15 \nDISPÕE SOBRE O CONSELHO REGIONAL JOVEM \nConsiderando: ") == (
        "DISPÕE SOBRE O CONSELHO REGIONAL JOVEM"
    )


def test_assunto_proposto_so_entra_conferido() -> None:
    from axon_scouting.rag.graph.domain.build.resolutions import checked_subjects

    chunks = [
        _chunk(
            "can-2015-02-1#preambulo@p1", "[Resolução CAN 02/2015]\nAltera o prazo de substituição do traje escoteiro"
        )
    ]
    proposals = {
        "propostas": [
            {"doc_key": "can-2015-02-1", "entidades": ["conceito:vestuario"], "trecho": "Traje Escoteiro"},
            {"doc_key": "can-2015-02-1", "entidades": ["conceito:inexistente"], "trecho": "traje escoteiro"},
            {"doc_key": "outra-1", "entidades": ["conceito:vestuario"], "trecho": "frase que não está lá"},
        ]
    }
    result = checked_subjects(proposals, chunks, {"conceito:vestuario"})
    assert result.found == {"can-2015-02-1": (("conceito:vestuario",), "can-2015-02-1#preambulo@p1")}
    assert len(result.dropped) == 2
