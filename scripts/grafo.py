"""O grafo de domínio do scouting (spec 106, ADR-064; contracts/grafo-cli.md).

    uv run python scripts/grafo.py build
    uv run python scripts/grafo.py query sobre tipo_orgao:diretoria_regional --papeis norma
    uv run python scripts/grafo.py query ligados tipo_orgao:diretoria_regional
    uv run python scripts/grafo.py query aplicavel_a inst:ge_inah
    uv run python scripts/grafo.py query reconhecer "Quem fiscaliza as contas da Diretoria Regional?"
    uv run python scripts/grafo.py plan "Quem fiscaliza as contas da Diretoria Regional?"
    uv run python scripts/grafo.py export --arrows [--entidade X --saltos N] > grafo.arrows.json

`build` lê `corpus/organizacao.yaml`, `corpus/dominio.yaml`, as fichas das atas (`corpus/resumos/atas/`) e o
artefato do índice (trechos com o caminho na árvore, catálogo, manifesto) e grava `graph.jsonl` (v2) ao lado do índice
do pacote. Sem modelo e sem rede; roda em segundos. Refaça depois de cada ingestão, ficha nova ou mudança nos YAML
curados. Saídas: 0, 1 (invariante: tipo sem papel) e 2 (YAML
inválido).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from axon_scouting.rag.graph.adapters.artifact import GRAPH, ArtifactGraphStore, write_graph_v2  # noqa: E402
from axon_scouting.rag.graph.adapters.sources import ArtifactSources  # noqa: E402
from axon_scouting.rag.graph.api import QUERIES, Evidence, PlanInvalid, reconhecer  # noqa: E402
from axon_scouting.rag.graph.domain.build.organs import OrganizationError  # noqa: E402
from axon_scouting.rag.graph.domain.build.records import BuildError, build_records  # noqa: E402
from axon_scouting.rag.graph.service import DomainGraphService  # noqa: E402

INDEX = ROOT / "src" / "axon_scouting" / "data" / "index"
ORGANIZATION = ROOT / "corpus" / "organizacao.yaml"
DOMAIN = ROOT / "corpus" / "dominio.yaml"
MINUTES = ROOT / "corpus" / "resumos" / "atas"


def build(args: argparse.Namespace) -> int:
    try:
        sources = ArtifactSources(args.index, args.organization, args.domain, args.minutes)
        records, report = build_records(
            organization=sources.organization(),
            chunks=list(sources.chunks()),
            documents=list(sources.documents()),
            manifest=sources.manifest(),
            organization_sha256=sources.organization_sha256(),
            domain=sources.domain(),
            domain_sha256=sources.domain_sha256(),
            minutes=list(sources.minutes()),
        )
    except OrganizationError as error:
        print(f"organizacao.yaml inválido: {error}", file=sys.stderr)
        return 2
    except BuildError as error:
        print(f"invariante: {error}", file=sys.stderr)
        return 1
    write_graph_v2(args.index / GRAPH, records)
    print(
        f"{report.entities} entidades, {report.versions} versões, {report.sections} seções (SOBRE), "
        f"{report.relations} relações de governança, {report.applies} APLICA_SE_A, {report.issued} EMITIDO_POR, "
        f"{report.events} eventos "
        f"→ {args.index / GRAPH}"
    )
    for entity, titles in sorted(report.per_entity.items()):
        print(f"  {entity}: {len(titles)} seção(ões) — " + "; ".join(titles)[:300])
    for line in report.dropped:
        print(f"  fora: {line}")
    return 0


def _service(args: argparse.Namespace) -> DomainGraphService:
    return DomainGraphService(ArtifactGraphStore(args.index))


def _print(evidence: list[Evidence], as_json: bool) -> None:
    if as_json:
        print(json.dumps([_jsonable(e) for e in evidence], ensure_ascii=False, indent=1, default=str))
        return
    for e in evidence:
        unit = e.unit_id or "(documento)"
        print(
            f"{e.doc_key:<45} {unit[-40:]:<40} {e.role:<11} {e.scope or '-':<9} {e.date or '-'!s:<10} "
            f"{e.edge.type} ← {e.edge.source} [{e.edge.origin}] → {e.target}"
        )
    print(f"{len(evidence)} evidência(s)")


def _jsonable(evidence: Evidence) -> dict[str, Any]:
    return {
        "query": evidence.query,
        "target": evidence.target,
        "doc_key": evidence.doc_key,
        "unit_id": evidence.unit_id,
        "chunk_ids": list(evidence.chunk_ids),
        "role": evidence.role,
        "scope": evidence.scope,
        "date": evidence.date,
        "edge": {"type": evidence.edge.type, "source": evidence.edge.source, "origin": evidence.edge.origin},
    }


def query(args: argparse.Namespace) -> int:
    service = _service(args)
    if args.consulta == "reconhecer":
        from axon_scouting.context.text import normalize

        aliases = asyncio.run(service.aliases())
        print(" ".join(reconhecer(normalize(" ".join(args.alvos)), aliases)) or "(nenhuma entidade)")
        return 0
    params: dict[str, Any] = {}
    if args.consulta == "sobre":
        params["entidades"] = list(args.alvos)
    elif args.alvos:
        key = {"aplicavel_a": "instituicao", "linha_do_tempo": "alvo", "ligados": "alvo"}.get(args.consulta)
        if key is None:
            print(f"{args.consulta}: sem argumento posicional", file=sys.stderr)
            return 2
        params[key] = args.alvos[0]
    for option in ("modo", "tipo", "emissor", "area", "ramo", "de", "tipo_de_unidade"):
        value = getattr(args, option, None)
        if value:
            params[option] = value
    for option in ("papeis", "tipos", "relacoes", "eventos"):
        value = getattr(args, option, None)
        if value:
            params[option] = value.split(",")
    try:
        evidence = asyncio.run(service.query(args.consulta, **params))
    except (PlanInvalid, NotImplementedError, TypeError) as error:
        print(f"{args.consulta}: {error}", file=sys.stderr)
        return 2
    _print(evidence, args.json)
    return 0


def plan(args: argparse.Namespace) -> int:
    """A derivação da 101 com os termos do grafo, como o `pack.py` compõe: mostra `entities` e o plano."""
    import dataclasses
    from datetime import date
    from types import MappingProxyType

    from axon_scouting.context.derivation import derive_context
    from axon_scouting.context.lexicon import LEXICON

    lexicon = dataclasses.replace(LEXICON, entity_terms=MappingProxyType(dict(asyncio.run(_service(args).aliases()))))
    day = date.fromisoformat(args.dia) if args.dia else date.today()
    context = derive_context(" ".join(args.pergunta), day, lexicon)
    print(
        json.dumps({"entities": list(context.entities), "plan": context.plan.to_json()}, ensure_ascii=False, indent=1)
    )
    return 0


def export(args: argparse.Namespace) -> int:
    """O grafo no formato do arrows.app, como `corpus/Organização Escoteira.revisado.json` (research R16)."""
    from axon_scouting.rag.graph.adapters.arrows import neighbourhood, to_arrows

    if not args.arrows:
        print("export: só --arrows por enquanto", file=sys.stderr)
        return 2
    graph = _service(args).graph()
    keep = neighbourhood(graph, args.entidade, args.saltos, units=not args.sem_unidades)
    print(json.dumps(to_arrows(graph, keep), ensure_ascii=False, indent=1))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", type=Path, default=INDEX)
    parser.add_argument("--organization", type=Path, default=ORGANIZATION)
    parser.add_argument("--domain", type=Path, default=DOMAIN)
    parser.add_argument("--minutes", type=Path, default=MINUTES)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build", help="monta o grafo e grava graph.jsonl (v2)")
    q = sub.add_parser("query", help="roda uma consulta do grafo, como a busca roda")
    q.add_argument("consulta", choices=[*QUERIES, "reconhecer"])
    q.add_argument("alvos", nargs="*")
    for option in ("--modo", "--tipo", "--emissor", "--area", "--ramo", "--de", "--tipo-de-unidade"):
        q.add_argument(option)
    for option in ("--papeis", "--tipos", "--relacoes", "--eventos"):
        q.add_argument(option, help="lista separada por vírgulas")
    q.add_argument("--json", action="store_true")
    p = sub.add_parser("plan", help="a derivação da 101 com os termos do grafo: entities e o plano")
    p.add_argument("pergunta", nargs="+")
    p.add_argument("--dia", help="dia de referência (AAAA-MM-DD); padrão, hoje")
    e = sub.add_parser("export", help="grava o grafo no formato do arrows.app")
    e.add_argument("--arrows", action="store_true")
    e.add_argument("--entidade")
    e.add_argument("--saltos", type=int, default=1)
    e.add_argument("--sem-unidades", action="store_true")
    args = parser.parse_args()
    return {"build": build, "query": query, "plan": plan, "export": export}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
