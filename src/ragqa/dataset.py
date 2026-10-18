"""Golden dataset loader and validator.

The dataset is a JSON Lines file: one ``GoldenCase`` object per line; blank lines and lines
starting with ``#`` are ignored. ``load_dataset`` reports *every* problem at once (invalid JSON,
schema errors, duplicate ids, empty question / answer) in a single ``DatasetError`` so a broken
dataset is fixed in one pass and never burns judge tokens.

It also returns warnings (not errors). The one warning today: factual cases without
``expected_context`` - the isolated generator stage (``generation_golden``) is skipped for those cases.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from ragqa.models import GoldenCase

_MAX_LISTED = 5  # case ids named in the missing-context warning


class DatasetError(ValueError):
    """The dataset is invalid; ``errors`` lists every problem found (with line numbers)."""

    def __init__(self, path: str | Path, errors: list[str]) -> None:
        self.path = str(path)
        self.errors = errors
        super().__init__(f"{self.path}: {len(errors)} problem(s):\n  " + "\n  ".join(errors))


def load_dataset(path: str | Path) -> tuple[list[GoldenCase], list[str]]:
    """Parse and validate ``path``; returns ``(cases, warnings)`` or raises ``DatasetError``."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"dataset not found: {p}")

    cases: list[GoldenCase] = []
    errors: list[str] = []
    seen: set[str] = set()
    for lineno, raw in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {lineno}: invalid JSON ({exc.msg})")
            continue
        try:
            case = GoldenCase.model_validate(data)
        except ValidationError as exc:
            detail = "; ".join(f"{'.'.join(str(x) for x in e['loc']) or 'case'}: {e['msg']}" for e in exc.errors())
            errors.append(f"line {lineno}: schema error - {detail}")
            continue
        if case.id in seen:
            errors.append(f"line {lineno}: duplicate id '{case.id}'")
        seen.add(case.id)
        if not case.question.strip():
            errors.append(f"line {lineno} ({case.id}): empty question")
        if not case.expected_answer.strip():
            errors.append(f"line {lineno} ({case.id}): empty expected_answer")
        cases.append(case)

    if not errors and not cases:
        errors.append("the dataset contains no cases")
    if errors:
        raise DatasetError(p, errors)
    return cases, _warnings(cases)


def _warnings(cases: list[GoldenCase]) -> list[str]:
    """One summary warning for factual cases that cannot feed the isolated generator stage."""
    missing = [c.id for c in cases
               if not (c.safety_probe or c.expect_refusal or c.expected_context)]
    if not missing:
        return []
    listed = ", ".join(missing[:_MAX_LISTED]) + (", ..." if len(missing) > _MAX_LISTED else "")
    return [(f"{len(missing)} factual case(s) have no expected_context "
             f"(isolated generator stage is skipped for them): {listed}")]
