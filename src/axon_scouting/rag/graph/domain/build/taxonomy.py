"""`APLICA_SE_A` e `EMITIDO_POR` pela taxonomia do acervo (spec 106 G3, US3; design.md, seção 2.4; research R20).

Por regra, sem modelo, de cada documento do catálogo:
- **`APLICA_SE_A`** (versão → `inst:`): o nível do documento (`scope`) e, abaixo do nacional, a região (`jurisdiction`)
  ou a subpasta da UEL (`…/local/<uel>/`). Nos níveis de uma instituição só (mundial, interamericano, nacional), o
  nível basta;
- **`EMITIDO_POR`** (versão → `orgao:`): a pasta da taxonomia do tipo do órgão está nos `bodies` do documento, e o
  documento se aplica à instituição de que o órgão é parte (as atas regionais de SP são da Assembleia Regional de SP).
A fonte das duas é a pasta do arquivo (`pasta:<caminho>`, sem o `ocr/`). Puro.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final

from axon_scouting.rag.graph.domain.build.instances import Instance

LEVEL_DIRS: Final = ("mundial", "interamericano", "nacional", "regional", "distrital", "local")
SOURCE: Final = "pasta:"


def folder_of(source_file: str) -> str:
    """A pasta do arquivo no `corpus/`, sem o espelho `ocr/`."""
    return source_file.removeprefix("ocr/").rsplit("/", 1)[0]


def subfolder(source_file: str) -> str | None:
    """A subpasta logo abaixo do nível (`sao-paulo`, a UEL), ou nenhuma."""
    parts = folder_of(source_file).split("/")
    for index, part in enumerate(parts):
        if part in LEVEL_DIRS:
            return parts[index + 1] if index + 1 < len(parts) else None
    return None


def applies_to(fields: Mapping[str, Any], source_file: str, institutions: Sequence[Instance]) -> str | None:
    """A instituição a que o documento se aplica, ou nenhuma (a UEL ou a região fora do catálogo)."""
    scope = fields.get("scope")
    if scope == "local":
        folder = subfolder(source_file)
        candidates = [i for i in institutions if i.level == "local" and i.folder is not None and i.folder == folder]
    elif scope in ("regional", "distrital"):
        candidates = [i for i in institutions if i.level == scope and i.jurisdiction == fields.get("jurisdiction")]
    else:
        candidates = [i for i in institutions if i.level == scope]
    return candidates[0].id if len(candidates) == 1 else None


def issued_by(fields: Mapping[str, Any], institution: str | None, organs: Sequence[Instance]) -> list[str]:
    """Os órgãos que emitem o documento: a pasta do tipo nos `bodies` e a instituição do órgão."""
    if institution is None:
        return []
    bodies = set(fields.get("bodies") or ())
    return [o.id for o in organs if o.taxonomy in bodies and institution in o.parents]
