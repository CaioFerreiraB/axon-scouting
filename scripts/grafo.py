"""O grafo das normas por órgão (ADR-063; relatório "Diretoria Regional", Fase 2).

    uv run python scripts/grafo.py build

Lê `corpus/organizacao.yaml` (os tipos de órgão e de instituição, curados) e a árvore dos trechos do artefato do índice,
liga cada seção de norma vigente ao órgão que o título dela nomeia e grava `graph.jsonl` ao lado do índice do pacote.
Sem modelo e sem rede; roda em segundos. Refaça depois de cada ingestão ou mudança no `organizacao.yaml`.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from axon_scouting.rag.index.adapters.graph import GRAPH, write_graph  # noqa: E402
from axon_scouting.rag.index.api import NormSection  # noqa: E402
from axon_scouting.rag.ingestion.domain.graph import ChunkPlace, organs_from_mapping, sections  # noqa: E402

INDEX = ROOT / "src" / "axon_scouting" / "data" / "index"
ORGANIZATION = ROOT / "corpus" / "organizacao.yaml"


def places(index: Path) -> list[ChunkPlace]:
    found: list[ChunkPlace] = []
    with (index / "chunks.jsonl").open(encoding="utf-8") as lines:
        for line in lines:
            raw = json.loads(line)
            found.append(
                ChunkPlace(
                    chunk_id=raw["chunk_id"],
                    doc_key=raw["doc_key"],
                    document_type=raw["document_type"],
                    document_title=raw.get("document_title") or "",
                    scope=raw.get("scope"),
                    vigente=raw.get("status") == "vigente",
                    path=tuple((p["unit_id"], p["kind"], p.get("title") or "") for p in raw.get("path") or []),
                )
            )
    return found


def build(args: argparse.Namespace) -> int:
    organs = organs_from_mapping(yaml.safe_load(args.organization.read_text(encoding="utf-8")))
    found = sections(places(args.index), organs)
    write_graph(
        args.index / GRAPH,
        {organ.id: organ.terms for organ in organs},
        [
            NormSection(
                organ=s.organ, doc_key=s.doc_key, unit_id=s.unit_id, title=s.title, scope=s.scope, chunk_ids=s.chunk_ids
            )
            for s in found
        ],
    )
    per_organ = Counter(s.organ for s in found)
    missing = [organ.id for organ in organs if organ.id not in per_organ]
    print(f"{len(organs)} órgãos e instituições, {len(found)} seções → {args.index / GRAPH}")
    for organ, count in sorted(per_organ.items()):
        print(
            f"  {organ}: {count} seção(ões) — "
            + "; ".join(f"{s.title} ({s.doc_key})" for s in found if s.organ == organ)[:300]
        )
    print(f"sem seção em norma vigente: {', '.join(missing) or 'nenhum'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", type=Path, default=INDEX)
    parser.add_argument("--organization", type=Path, default=ORGANIZATION)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build", help="monta o grafo e grava graph.jsonl")
    args = parser.parse_args()
    return build(args)


if __name__ == "__main__":
    raise SystemExit(main())
