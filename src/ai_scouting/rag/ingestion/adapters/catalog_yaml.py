"""Leitura do `corpus.yaml` (spec 102 contracts/corpus-catalog.md): só carrega; quem valida é o domínio."""

from __future__ import annotations

from pathlib import Path

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
