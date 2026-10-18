"""Convert the source spreadsheet (``golden-dataset.xlsx``) into the golden dataset (``datasets/golden.jsonl``).

Usage (re-run whenever the spreadsheet changes):

    uv run python scripts/xlsx_to_golden.py "datasets/source/golden-dataset.xlsx" -o datasets/golden.jsonl

Expected columns (header row, case-insensitive; other/blank columns such as ``Status`` are ignored):

    Q ID | Question | Expected Answer | Answer must come from

Mapping
-------
* ``Q ID``                  -> ``id``
* ``Question``              -> ``question``
* ``Expected Answer``       -> ``expected_answer`` (whitespace tidied, content untouched)
* ``Answer must come from`` -> the citation. The document number at the start of the cell
  (e.g. ``BXU548683_R`` from ``BXU548683_R Rev A - Comparative ...\\nSection 5 (p.6)``) becomes
  ``expected_citations``; the full original text is kept in ``source_reference`` for humans.
  The generator.citation metric counts a hit when the id appears inside a returned citation.
* Rows whose citation starts with ``Not in the documents`` become ``expect_refusal: true``
  (out-of-scope; the application should decline) with no citation. If the question is a
  prompt-injection / access-control probe (citation mentions "guardrail") the case is also a
  ``safety_probe``.
* Rows whose citation cell says "No citations" / "Answer is generic" (answer comes from the system
  prompt or guardrails, e.g. Q-00) become ``category: system-prompt`` with no citation and
  ``expect_refusal: true`` so retriever / generator / triad metrics skip them.
* Rows missing a question, an answer or (for answerable rows) a citation are SKIPPED and listed on
  stderr, as are fully blank rows. Nothing is invented.
* ``category`` is ``factual`` | ``out-of-scope`` | ``safety``; the document number is added as a tag.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import openpyxl

DOC_ID = re.compile(r"^\s*([A-Za-z]+\d[\w]*)")
NOT_IN_DOCS = re.compile(r"^\s*not in the documents", re.IGNORECASE)
NO_CITATION = re.compile(r"no citations?|answer is generic", re.IGNORECASE)


def tidy(value: object) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\t", " ").replace("\r", "")
    text = "\n".join(re.sub(r"[ ]+", " ", line).strip() for line in text.split("\n"))
    return re.sub(r"\n{2,}", "\n", text).strip()


def convert(path: Path) -> tuple[list[dict], list[str]]:
    ws = openpyxl.load_workbook(path, data_only=True).active
    header = {str(c).strip().lower(): i for i, c in enumerate(next(ws.iter_rows(max_row=1, values_only=True)))
              if c is not None}
    try:
        col = {k: header[k] for k in ("q id", "question", "expected answer", "answer must come from")}
    except KeyError as missing:
        raise SystemExit(f"error: column {missing} not found; headers are {sorted(header)}")

    cases: list[dict] = []
    skipped: list[str] = []
    for rownum, row in enumerate(ws.iter_rows(min_row=2, values_only=True), start=2):
        qid, q, a, c = (tidy(row[col[k]]) for k in ("q id", "question", "expected answer", "answer must come from"))
        if not any((qid, q, a, c)):
            continue  # blank row
        label = qid or f"row {rownum}"
        if not (q and a):
            skipped.append(f"{label}: missing {'question' if not q else 'expected answer'}")
            continue
        if NOT_IN_DOCS.match(c):
            guard = "guardrail" in c.lower()
            cases.append({"id": qid, "question": q, "expected_answer": a, "expected_citations": [],
                          "category": "safety" if guard else "out-of-scope", "tags": [],
                          "safety_probe": guard, "expect_refusal": True, "source_reference": None})
            continue
        if NO_CITATION.search(c):  # answered from the system prompt / guardrails, not the documents
            cases.append({"id": qid, "question": q, "expected_answer": a, "expected_citations": [],
                          "category": "system-prompt", "tags": [], "safety_probe": False,
                          "expect_refusal": True, "source_reference": c.replace("\n", " | ")})
            continue
        m = DOC_ID.match(c)
        if not c or not m:
            skipped.append(f"{label}: missing or unparsable citation {c[:40]!r}")
            continue
        doc = m.group(1)
        cases.append({"id": qid, "question": q, "expected_answer": a, "expected_citations": [doc],
                      "category": "factual", "tags": [doc], "safety_probe": False, "expect_refusal": False,
                      "source_reference": " | ".join(c.split("\n"))})
    return cases, skipped


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("xlsx", type=Path)
    ap.add_argument("-o", "--output", type=Path, default=Path("datasets/golden.jsonl"))
    args = ap.parse_args()

    cases, skipped = convert(args.xlsx)
    header = (f"# Generated by scripts/xlsx_to_golden.py from {args.xlsx.name}. Do not edit by hand; "
              "edit the spreadsheet and re-run.\n")
    args.output.write_text(header + "\n".join(json.dumps(c, ensure_ascii=False) for c in cases) + "\n")
    print(f"wrote {len(cases)} cases to {args.output}")
    for s in skipped:
        print(f"skipped {s}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
