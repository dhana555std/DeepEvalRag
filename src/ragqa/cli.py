"""Command line interface: ``ragqa run | multirun | validate-dataset``.

Examples
--------
    ragqa validate-dataset
    ragqa run                                   # single run, default judge (Groq)
    ragqa run --judge anthropic --only 'generator.*'
    ragqa multirun --runs 5                     # judge variance, API called once
    ragqa multirun --runs 3 --judges groq,openai,google --recollect

Exit codes (what CI consumes)
-----------------------------
0  quality gate passed
1  quality gate failed (a metric is below its threshold, or errored)
2  usage / configuration / dataset error (nothing was evaluated)

The CLI is a thin shell over ``ragqa.evaluation.run_sync``; the pytest suite and
``deepeval test run`` use the same function.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from ragqa.config import load_reporting
from ragqa.dataset import DatasetError, load_dataset
from ragqa.evaluation import RunOptions, run_sync
from ragqa.models import RunReport
from ragqa.reporting import write_reports


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dataset", default="datasets/golden.jsonl", help="golden dataset (jsonl)")
    p.add_argument("--config-dir", type=Path, default=None, help="config directory (default: ./config)")
    p.add_argument("--only", help="comma separated metric patterns, e.g. 'retriever.*,application.latency_ms'")
    p.add_argument("--limit", type=int, help="evaluate only the first N cases")
    p.add_argument("--concurrency", type=int, default=4, help="concurrent judge calls (lower if rate limited)")
    p.add_argument("--output-dir", help="report directory (default from reporting.yaml)")
    p.add_argument("--reporters", help="comma separated reporters (default from reporting.yaml)")
    p.add_argument("--title", help="report title")
    p.add_argument("--no-gate", action="store_true", help="always exit 0 (report only)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ragqa", description="RAG evaluation harness (DeepEval)")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="single evaluation run")
    _common(run)
    run.add_argument("--judge", help="judge provider (default: JUDGE_PROVIDER or judges.yaml default)")

    multi = sub.add_parser("multirun", help="repeat scoring N times (and/or across judges)")
    _common(multi)
    multi.add_argument("--runs", type=int, default=3, help="scoring repetitions per judge")
    multi.add_argument("--judges", help="comma separated judge providers")
    multi.add_argument("--recollect", action="store_true",
                       help="call the RAG API again for every run (measures application variance too)")

    val = sub.add_parser("validate-dataset", help="validate the golden dataset and exit")
    val.add_argument("--dataset", default="datasets/golden.jsonl")
    return parser


def _split(value: str | None) -> list[str] | None:
    return [v.strip() for v in value.split(",") if v.strip()] if value else None


def _print_summary(report: RunReport) -> None:
    for run in report.runs:
        print(f"\n[{run.run_id}] judge={run.judge} duration={run.duration_s:.1f}s")
        for a in run.aggregates:
            status = {True: "PASS", False: "FAIL", None: "info"}[a.passed]
            val = "n/a" if a.value is None else f"{a.value:.3f}"
            thr = "-" if a.threshold is None else f"{a.threshold}"
            err = f"  errors={a.errors}" if a.errors else ""
            print(f"  {status:4}  {a.key:<32} {val:>10}  (threshold {thr}){err}")
    if report.multi:
        print("\nAcross runs (mean ± std):")
        for m in report.multi:
            if m.mean is not None:
                print(f"  {m.judge:<28} {m.key:<30} {m.mean:.3f} ± {m.std:.3f}")
    print(f"\nRESULT: {'PASSED' if report.passed else 'FAILED'}")


def main(argv: list[str] | None = None) -> int:
    load_dotenv(override=False)
    args = build_parser().parse_args(argv)

    try:
        if args.command == "validate-dataset":
            cases, warnings = load_dataset(args.dataset)
            for w in warnings:
                print(f"warning: {w}")
            print(f"OK: {len(cases)} cases valid")
            return 0

        # Validate before spending any money on API / judge calls.
        load_dataset(args.dataset)
        opts = RunOptions(
            dataset=args.dataset, config_dir=args.config_dir, only=_split(args.only), limit=args.limit,
            concurrency=args.concurrency, title=args.title,
        )
        if args.command == "run":
            opts.judges = [args.judge] if args.judge else None
        else:
            opts.runs, opts.judges, opts.recollect = args.runs, _split(args.judges), args.recollect

        report = run_sync(opts)
        reporting = load_reporting(args.config_dir)
        files = write_reports(report, reporting, args.output_dir, _split(args.reporters))
        _print_summary(report)
        print("Reports: " + ", ".join(str(f) for f in files))
        return 0 if (report.passed or args.no_gate) else 1
    except (DatasetError, FileNotFoundError, KeyError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
