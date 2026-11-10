"""Fichas das atas e os resumos da busca global (ADR-062; relatório "Diretoria Regional", Fase G).

Dois passos, rodados no desenvolvimento, nunca na instalação:

    # 1. Extrai a ficha de cada ata selecionada, por um modelo do OpenRouter, e confere cada item contra a página.
    #    A chave vem só do ambiente (AXON_OPENROUTER_API_KEY); nada é gravado com ela.
    uv run python scripts/fichas.py extract --category Assembleias --scope regional
    uv run python scripts/fichas.py extract --category "Conselho Consultivo" --scope regional

    # 2. Monta os resumos (fichas e linha do tempo por órgão e ano, sem modelo) e grava `summaries.jsonl` ao lado do
    #    índice do pacote.
    uv run python scripts/fichas.py build

As fichas ficam em `corpus/resumos/atas/<doc_key>.yaml`, no git, para revisão. Uma ata só é reextraída quando o texto
dela muda (`extracao.fontes_sha256`) ou com `--force`. O texto vem das páginas gravadas no artefato do índice
(`documents.jsonl`), o mesmo que a ingestão leu.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from axon_scouting.rag.index.adapters.summaries import SUMMARIES, write_summaries  # noqa: E402
from axon_scouting.rag.index.api import SummaryEntry  # noqa: E402
from axon_scouting.rag.ingestion.domain.minutes import (  # noqa: E402
    MeetingRecord,
    MinutesDocument,
    card_parts,
    confirm,
    problems,
    record_from_mapping,
    record_to_mapping,
    timeline_parts,
)

INDEX = ROOT / "src" / "axon_scouting" / "data" / "index"
OUT = ROOT / "corpus" / "resumos" / "atas"
PROMPT = ROOT / "scripts" / "fichas-prompt.md"
URL = "https://openrouter.ai/api/v1/chat/completions"
KEY = "AXON_OPENROUTER_API_KEY"

_ITEM = {"type": "object", "additionalProperties": False}
SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["reuniao", "deliberacoes", "eleicoes", "numeros"],
    "properties": {
        "reuniao": {
            **_ITEM,
            "required": ["tipo", "data", "local"],
            "properties": {
                "tipo": {"type": "string", "enum": ["ordinaria", "extraordinaria", "outra"]},
                "data": {"type": "string"},
                "local": {"type": "string"},
            },
        },
        "deliberacoes": {
            "type": "array",
            "items": {
                **_ITEM,
                "required": ["tema", "decisao", "votacao", "pagina"],
                "properties": {
                    "tema": {"type": "string"},
                    "decisao": {"type": "string"},
                    "votacao": {"type": "string"},
                    "pagina": {"type": "integer"},
                },
            },
        },
        "eleicoes": {
            "type": "array",
            "items": {
                **_ITEM,
                "required": ["orgao", "cargo", "eleito", "mandato", "pagina"],
                "properties": {
                    "orgao": {"type": "string"},
                    "cargo": {"type": "string"},
                    "eleito": {"type": "string"},
                    "mandato": {"type": "string"},
                    "pagina": {"type": "integer"},
                },
            },
        },
        "numeros": {
            "type": "array",
            "items": {
                **_ITEM,
                "required": ["o_que", "valor", "pagina"],
                "properties": {"o_que": {"type": "string"}, "valor": {"type": "string"}, "pagina": {"type": "integer"}},
            },
        },
    },
}


def documents(index: Path) -> list[dict[str, Any]]:
    with (index / "documents.jsonl").open(encoding="utf-8") as lines:
        return [json.loads(line) for line in lines if line.strip()]


def minutes_document(raw: dict[str, Any]) -> MinutesDocument:
    fields = raw["fields"]
    effective = fields.get("effective_date")
    return MinutesDocument(
        doc_key=raw["doc_key"],
        title=fields["document_title"],
        category=fields.get("category") or "",
        bodies=tuple(fields.get("bodies") or ()),
        scope=fields.get("scope"),
        source_url=raw.get("source_url") or fields.get("source_url"),
        effective_date=date.fromisoformat(effective) if effective else None,
    )


def page_text(pages: list[str]) -> str:
    return "\n\n".join(f"[p. {number}]\n{text.strip()}" for number, text in enumerate(pages, start=1))


def call_model(model: str, prompt: str, text: str, key: str) -> dict[str, Any]:
    body = {
        "model": model,
        "temperature": 0,
        "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": text}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "ficha", "strict": True, "schema": SCHEMA}},
        "provider": {"require_parameters": True},
    }
    request = urllib.request.Request(
        URL,
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        raise RuntimeError(
            f"OpenRouter respondeu {error.code}: {error.read().decode('utf-8', 'replace')[:300]}"
        ) from None
    content = payload["choices"][0]["message"]["content"]
    data: dict[str, Any] = json.loads(content)
    data["_custo"] = (payload.get("usage") or {}).get("cost")
    return data


def extract_one(raw: dict[str, Any], *, model: str, prompt: str, key: str, force: bool) -> str:
    doc_key = raw["doc_key"]
    pages: list[str] = raw.get("pages") or []
    text = page_text(pages)
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    target = OUT / f"{doc_key}.yaml"
    if target.exists() and not force:
        existing = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
        if (existing.get("extracao") or {}).get("fontes_sha256") == sha:
            return f"{doc_key}: sem mudança"
    data = call_model(model, prompt, text, key)
    cost = data.pop("_custo", None)
    data["doc_key"] = doc_key
    _drop_incomplete(data)
    data["extracao"] = {"modelo": model, "data": datetime.now(UTC).date().isoformat(), "fontes_sha256": sha}
    record = confirm(record_from_mapping(data), pages)
    target.write_text(
        yaml.safe_dump(record_to_mapping(record), allow_unicode=True, sort_keys=False, width=120), encoding="utf-8"
    )
    total = len(record.deliberacoes) + len(record.eleicoes) + len(record.numeros)
    rejected = len(problems(record))
    return f"{doc_key}: {total - rejected}/{total} itens conferidos (custo {cost})"


REQUIRED = {"deliberacoes": ("tema", "decisao"), "eleicoes": ("cargo", "eleito"), "numeros": ("o_que", "valor")}


def _drop_incomplete(data: dict[str, Any]) -> None:
    """Item sem campo obrigatório (o modelo às vezes devolve uma eleição sem nome) sai, em vez de derrubar a ficha."""
    for key, fields in REQUIRED.items():
        data[key] = [
            item for item in data.get(key) or [] if all(str(item.get(field) or "").strip() for field in fields)
        ]


def extract(args: argparse.Namespace) -> int:
    key = os.environ.get(KEY)
    if not key:
        print(f"erro: {KEY} ausente no ambiente", file=sys.stderr)
        return 2
    prompt = PROMPT.read_text(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    selected = [
        raw
        for raw in documents(args.index)
        if raw["fields"]["document_type"] == "ata"
        and (not args.category or raw["fields"].get("category") == args.category)
        and (not args.scope or raw["fields"].get("scope") == args.scope)
        and (not args.doc_key or raw["doc_key"] in args.doc_key)
    ]
    print(f"{len(selected)} atas", flush=True)
    failures = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(extract_one, raw, model=args.model, prompt=prompt, key=key, force=args.force): raw["doc_key"]
            for raw in selected
        }
        for future in concurrent.futures.as_completed(futures):
            try:
                print(future.result(), flush=True)
            except Exception as error:  # uma ata que falha não derruba as outras
                failures += 1
                print(f"{futures[future]}: FALHOU — {error}", flush=True)
    return 1 if failures else 0


def build(args: argparse.Namespace) -> int:
    raws = {raw["doc_key"]: raw for raw in documents(args.index)}
    docs = {key: minutes_document(raw) for key, raw in raws.items()}
    records: list[MeetingRecord] = []
    for path in sorted(OUT.glob("*.yaml")):
        record = record_from_mapping(yaml.safe_load(path.read_text(encoding="utf-8")))
        if record.doc_key not in raws:
            print(f"aviso: {path.name} cita uma ata fora do índice; ignorada", file=sys.stderr)
            continue
        records.append(confirm(record, raws[record.doc_key].get("pages") or []))
    parts = [part for record in records for part in card_parts(record, docs[record.doc_key])]
    parts += timeline_parts(records, docs)
    entries = [
        SummaryEntry(
            summary_id=part.summary_id,
            kind=part.kind,
            title=part.title,
            text=part.text,
            year=part.year,
            meeting_date=part.meeting_date,
            doc_keys=part.doc_keys,
            bodies=part.bodies,
            scope=part.scope,
            page=part.page,
            source_url=part.source_url,
        )
        for part in parts
        if part.text.strip()
    ]
    write_summaries(args.index / SUMMARIES, entries)
    kinds = {kind: sum(1 for e in entries if e.kind == kind) for kind in ("ficha", "linha_do_tempo")}
    print(f"{len(records)} fichas → {len(entries)} resumos {kinds} em {args.index / SUMMARIES}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", type=Path, default=INDEX, help="diretório do artefato do índice")
    sub = parser.add_subparsers(dest="command", required=True)
    ex = sub.add_parser("extract", help="extrai e confere as fichas das atas")
    ex.add_argument("--category", help='categoria da taxonomia ("Assembleias", "Conselho Consultivo"…)')
    ex.add_argument("--scope", help="abrangência (regional, nacional…)")
    ex.add_argument("--doc-key", action="append", help="só esta ata (pode repetir)")
    ex.add_argument("--model", default="openai/gpt-4.1-mini")
    ex.add_argument("--workers", type=int, default=4)
    ex.add_argument("--force", action="store_true", help="reextrai mesmo sem mudança no texto")
    sub.add_parser("build", help="monta os resumos e grava summaries.jsonl")
    args = parser.parse_args()
    return extract(args) if args.command == "extract" else build(args)


if __name__ == "__main__":
    raise SystemExit(main())
