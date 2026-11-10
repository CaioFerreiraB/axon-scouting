"""O `fact_recall` antes e depois, por item e por grupo (spec 105 FR-026, research R26; ADR-064).

    python3 scripts/eval_diff.py ANTES.json DEPOIS.json [--golden eval/acervo.yaml] [--metric scouting:fact_recall]

Lê dois relatórios do `axon eval run` e imprime cada item cujo valor mudou (antes → depois), os itens que só existem
num dos dois, o agregado por grupo e o total. Os grupos vêm de `extra.group` do conjunto (`--golden`); sem ele, só o
total. É leitura: termina com 0 e não barra — quem barra é a linha de base do harness. Só biblioteca padrão; o YAML é
lido pelo PyYAML quando está instalado.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

METRIC = "scouting:fact_recall"


def _values(report: Path, metric: str) -> dict[str, float | None]:
    raw = json.loads(report.read_text(encoding="utf-8"))
    values: dict[str, float | None] = {}
    for item in raw.get("items") or []:
        value = (item.get("metrics") or {}).get(metric)
        values[str(item["id"])] = float(value) if isinstance(value, (int, float)) else None
    return values


def _groups(golden: Path | None) -> dict[str, str]:
    if golden is None:
        return {}
    import yaml  # PyYAML: dependência de desenvolvimento do pack

    data: dict[str, Any] = yaml.safe_load(golden.read_text(encoding="utf-8"))
    return {str(item["id"]): str((item.get("extra") or {}).get("group", "-")) for item in data.get("items") or []}


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--golden", type=Path, default=None, help="o conjunto, para os grupos (extra.group)")
    parser.add_argument("--metric", default=METRIC)
    args = parser.parse_args()

    before = _values(args.before, args.metric)
    after = _values(args.after, args.metric)
    groups = _groups(args.golden)

    changed = [i for i in before if i in after and before[i] != after[i]]
    print(f"{args.metric}: {args.before.name} → {args.after.name}")
    if changed:
        print("\nItens que mudaram:")
        for item in sorted(changed, key=lambda i: (groups.get(i, ""), i)):
            print(f"  {item:<45} {groups.get(item, '-'):<18} {_fmt(before[item])} → {_fmt(after[item])}")
    else:
        print("\nNenhum item mudou.")
    for label, only in (("só no antes", set(before) - set(after)), ("só no depois", set(after) - set(before))):
        if only:
            print(f"\nItens {label}: {', '.join(sorted(only))}")

    by_group: dict[str, tuple[list[float], list[float]]] = defaultdict(lambda: ([], []))
    for item in set(before) | set(after):
        pair = by_group[groups.get(item, "-")]
        if before.get(item) is not None:
            pair[0].append(before[item])  # type: ignore[arg-type]
        if after.get(item) is not None:
            pair[1].append(after[item])  # type: ignore[arg-type]
    if groups:
        print("\nPor grupo:")
        for group in sorted(by_group):
            b, a = by_group[group]
            print(f"  {group:<20} {_fmt(_mean(b))} → {_fmt(_mean(a))}")
    all_b = [v for v in before.values() if v is not None]
    all_a = [v for v in after.values() if v is not None]
    print(f"\nTotal: {_fmt(_mean(all_b))} → {_fmt(_mean(all_a))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
