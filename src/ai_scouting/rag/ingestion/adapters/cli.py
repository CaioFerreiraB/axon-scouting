"""CLI `ai-scouting-corpus` (spec 102 contracts/ingestion-cli.md; research R17): argumentos, saída e códigos.

Quem monta os adapters é o `pack.py` (`corpus_cli`), que entrega uma `Composition`. Códigos de saída: 0 sucesso; 1
rejeição (nada aplicado); 2 uso, configuração ou infraestrutura (nada aplicado). Segredos nunca aparecem na saída.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from collections.abc import Callable, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

from ai_scouting.rag.index.api import IndexUnavailable
from ai_scouting.rag.ingestion.adapters.catalog_yaml import CatalogUnreadable
from ai_scouting.rag.ingestion.domain.report import IngestionReport
from ai_scouting.rag.ingestion.service import IndexStatus, IngestionService

OK, REJECTED, FAILED = 0, 1, 2


class ConfigurationError(Exception):
    """Variável ausente ou inválida, modelo com hash errado: nada foi tentado."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Composition:
    index_name: str
    open_service: Callable[[bool], AbstractAsyncContextManager[IngestionService]]
    """Monta o serviço; o argumento diz se o comando codifica (e precisa do modelo de embedding)."""
    load_catalog: Callable[[Path], tuple[object, Path]]
    fetch_model: Callable[[Path], Sequence[tuple[Path, str]]]
    secrets: Sequence[str] = ()
    out: TextIO = field(default_factory=lambda: sys.stdout)
    err: TextIO = field(default_factory=lambda: sys.stderr)


def main(argv: Sequence[str], compose: Composition) -> int:
    args = _parser().parse_args(argv)
    try:
        return asyncio.run(_dispatch(args, compose))
    except (ConfigurationError, IndexUnavailable, OSError, ValueError) as exc:
        if isinstance(exc, CatalogUnreadable):
            _print(compose.err, compose, f"catálogo ilegível: {exc}")
            return REJECTED
        _print(compose.err, compose, f"erro: {exc}")
        return FAILED


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ai-scouting-corpus", description="Ingestão do corpus do pack scouting.")
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser(
        "ingest", help="publica um build que espelha o catálogo, reaproveitando o que não mudou"
    )
    ingest.add_argument("catalog", type=Path, help="caminho do corpus.yaml")
    commands.add_parser("status", help="versão publicada e documentos por linhagem")
    model = commands.add_parser("model", help="modelo de embedding")
    model_commands = model.add_subparsers(dest="model_command", required=True)
    fetch = model_commands.add_parser("fetch", help="baixa a revisão fixada do e5-small e confere os sha256")
    fetch.add_argument("directory", type=Path)
    return parser


async def _dispatch(args: argparse.Namespace, compose: Composition) -> int:
    if args.command == "model":
        for path, digest in compose.fetch_model(args.directory):
            _print(compose.out, compose, f"{path}  sha256 {digest}")
        return OK
    if args.command == "ingest":
        catalog, base_dir = compose.load_catalog(args.catalog)
        started = time.perf_counter()
        async with compose.open_service(True) as service:
            report = await service.ingest(catalog, base_dir)
        _print(compose.out, compose, _report(compose.index_name, report, time.perf_counter() - started))
        return OK if report.ok else REJECTED
    async with compose.open_service(False) as service:
        status = await service.status()
    _print(compose.out, compose, _status(compose.index_name, status))
    return OK


def _report(index: str, report: IngestionReport, seconds: float) -> str:
    if not report.ok:
        problems = "\n".join(f"  - {p}" for p in report.problems)
        return f"problemas:\n{problems}\nnada foi aplicado"
    assert report.version is not None
    lines = [f"índice {index} — build {report.build} publicado", f"versão: {report.version.describe()}", ""]
    lines.append(f"{'documento':<32} {'status':<12} {'chunks':>6}  origem")
    for doc in report.documents:
        origin = "reaproveitado" if doc.reused else "codificado"
        lines.append(f"{doc.doc_key:<32} {doc.status.value:<12} {doc.chunks:>6}  {origin}")
    if report.removed:
        lines += ["", f"removidos: {', '.join(report.removed)}"]
    warnings = [f"  {doc.doc_key}: {w}" for doc in report.documents for w in doc.warnings]
    if warnings:
        lines += ["avisos:", *warnings]
    minutes, rest = divmod(round(seconds), 60)
    lines.append(f"duração: {minutes}m{rest:02d}s")
    return "\n".join(lines)


def _status(index: str, status: IndexStatus) -> str:
    if status.build is None:
        return f"índice {index}: nenhum build publicado"
    lines = [f"índice {index} — build {status.build.name}", f"versão: {status.build.version.describe()}", ""]
    lines.append(f"{'linhagem':<24} {'versão':<10} {'status':<12} {'chunks':>6}")
    for doc in sorted(status.documents, key=lambda d: (d.fields.document_id, d.fields.version)):
        fields = doc.fields
        lines.append(f"{fields.document_id:<24} {fields.version:<10} {fields.status.value:<12} {len(doc.chunk_ids):>6}")
    lines.append(f"\n{len(status.documents)} documentos, {sum(len(d.chunk_ids) for d in status.documents)} chunks")
    return "\n".join(lines)


def _print(stream: TextIO, compose: Composition, text: str) -> None:
    for secret in compose.secrets:
        if secret:
            text = text.replace(secret, "***")
    print(text, file=stream)
