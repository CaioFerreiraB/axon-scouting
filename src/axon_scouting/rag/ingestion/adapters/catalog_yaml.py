"""Leitura do `corpus.yaml` (spec 102 contracts/corpus-catalog.md): só carrega; quem valida é o domínio."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import yaml


class CatalogUnreadable(ValueError):
    """O arquivo existe, mas não é YAML válido."""


def load_catalog(path: Path) -> tuple[object, Path]:
    """O objeto carregado e o diretório do catálogo (base dos caminhos dos PDFs). `FileNotFoundError` se não existe."""
    text = path.read_text(encoding="utf-8")
    try:
        return yaml.safe_load(text), path.resolve().parent
    except yaml.YAMLError as exc:
        raise CatalogUnreadable(f"{path}: YAML inválido: {exc}") from exc


TAXONOMY: Final = "taxonomia.yaml"
STRUCTURE: Final = "estrutura.yaml"


def load_structure(base_dir: Path) -> tuple[object, object] | None:
    """A taxonomia e a estrutura ao lado do catálogo (ADR-048); `None` sem as duas. Só uma delas é problema da
    validação: volta com a outra vazia."""
    paths = (base_dir / TAXONOMY, base_dir / STRUCTURE)
    if not any(path.exists() for path in paths):
        return None
    loaded: list[object] = []
    for path in paths:
        if not path.exists():
            loaded.append(None)
            continue
        try:
            loaded.append(yaml.safe_load(path.read_text(encoding="utf-8")))
        except yaml.YAMLError as exc:
            raise CatalogUnreadable(f"{path}: YAML inválido: {exc}") from exc
    return loaded[0], loaded[1]
