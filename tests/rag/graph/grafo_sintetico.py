"""Um grafo sintético pequeno, para as suítes da 106."""

from __future__ import annotations

from typing import Any


def registros(chunking: str = "teste") -> list[dict[str, Any]]:
    def entity(id_: str, terms: list[str]) -> dict[str, Any]:
        kind = id_.split(":", 1)[0]
        return {
            "kind": "entity",
            "id": id_,
            "type": kind,
            "name": id_,
            "terms": terms,
            "source": f"organizacao.yaml:{kind}/{id_}",
            "origin": "curado",
        }

    def version(doc_key: str, kind: str, role: str, scope: str) -> dict[str, Any]:
        return {
            "kind": "version",
            "doc_key": doc_key,
            "document_id": doc_key.rsplit("-", 1)[0],
            "type": kind,
            "role": role,
            "status": "vigente",
            "scope": scope,
            "date": "2024-01-01",
        }

    def unit(unit_id: str, chunks: list[str]) -> dict[str, Any]:
        return {
            "kind": "unit",
            "unit_id": unit_id,
            "doc_key": unit_id.split("#", 1)[0],
            "unit_kind": "secao",
            "title": unit_id,
            "chunk_ids": chunks,
        }

    def sobre(unit_id: str, to: str) -> dict[str, Any]:
        return {
            "kind": "edge",
            "type": "SOBRE",
            "mode": "disciplina",
            "from": unit_id,
            "to": to,
            "source": unit_id,
            "origin": "regra",
        }

    return [
        {"kind": "meta", "version": 2, "chunking": chunking},
        entity("tipo_orgao:diretoria_regional", ["diretoria regional"]),
        entity("tipo_orgao:assembleia_regional", ["assembleia regional"]),
        entity("tipo_orgao:orfao", ["orfao"]),
        version("estatuto-2024", "estatuto", "norma", "nacional"),
        version("regulamento-sp-2022", "regulamento", "norma", "regional"),
        version("local-2026", "estatuto", "norma", "local"),
        version("mundial-2020", "politica", "norma", "mundial"),
        unit("regulamento-sp-2022#u/cap3", ["regulamento-sp-2022#a26@p9", "regulamento-sp-2022#a27@p9"]),
        unit("estatuto-2024#u/s3", ["estatuto-2024#a27@p19"]),
        unit("local-2026#u/s1", ["local-2026#a10@p8"]),
        unit("mundial-2020#u/s1", ["mundial-2020#p1"]),
        unit("estatuto-2024#u/s2", ["estatuto-2024#a24@p17"]),
        sobre("regulamento-sp-2022#u/cap3", "tipo_orgao:diretoria_regional"),
        sobre("estatuto-2024#u/s3", "tipo_orgao:diretoria_regional"),
        sobre("local-2026#u/s1", "tipo_orgao:diretoria_regional"),
        sobre("mundial-2020#u/s1", "tipo_orgao:diretoria_regional"),
        sobre("estatuto-2024#u/s2", "tipo_orgao:assembleia_regional"),
        {
            "kind": "edge",
            "type": "ELEGE",
            "from": "tipo_orgao:assembleia_regional",
            "to": "tipo_orgao:diretoria_regional",
            "source": "estatuto-2024#a24@p17",
            "origin": "curado",
        },
    ]
