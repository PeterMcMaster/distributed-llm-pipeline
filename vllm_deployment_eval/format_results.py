#!/usr/bin/env python3
"""Normalize stored vLLM evaluation results.

This rewrites derived reporting files from the canonical run artifacts:
`run_record.json` and `requests.jsonl`. Raw logs, GPU samples, metrics dumps, and
request records are left untouched.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

from run_experiment import (
    RESULTS_DIR,
    ROOT,
    SUMMARY_FIELDS,
    STRATEGY_RESULT_GROUPS,
    relative_run_dir,
    write_summary,
)


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def strategy_group(strategy: str | None) -> str | None:
    if not strategy:
        return None
    return STRATEGY_RESULT_GROUPS.get(strategy, strategy.replace("-", ""))


def discover_run_dirs(results_dir: Path) -> list[Path]:
    return sorted(path.parent for path in results_dir.rglob("run_record.json"))


def move_run_to_group(results_dir: Path, run_dir: Path, run_record: dict) -> Path:
    group = strategy_group(run_record.get("strategy"))
    if not group or run_dir.parent.name == group:
        return run_dir
    target = results_dir / group / run_dir.name
    if target == run_dir:
        return run_dir
    if target.exists():
        raise SystemExit(f"Refusing to overwrite existing result directory: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(run_dir), str(target))
    return target


def normalize_run(results_dir: Path, run_dir: Path) -> dict:
    run_record_path = run_dir / "run_record.json"
    run_record = json.loads(run_record_path.read_text(encoding="utf-8"))
    run_dir = move_run_to_group(results_dir, run_dir, run_record)
    run_record_path = run_dir / "run_record.json"
    run_record["run_dir"] = relative_run_dir(run_dir)
    run_record_path.write_text(
        json.dumps(run_record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_summary(run_dir, run_record, load_jsonl(run_dir / "requests.jsonl"))
    return run_record


def rewrite_indexes(results_dir: Path, records: list[dict]) -> None:
    records = sorted(records, key=lambda item: item.get("run_id", ""))
    with (results_dir / "experiments.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    with (results_dir / "summary_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field) for field in SUMMARY_FIELDS})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    if not results_dir.exists():
        raise SystemExit(f"Results directory does not exist: {results_dir}")

    records = [normalize_run(results_dir, run_dir) for run_dir in discover_run_dirs(results_dir)]
    rewrite_indexes(results_dir, records)
    print(f"Formatted {len(records)} runs under {results_dir.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
