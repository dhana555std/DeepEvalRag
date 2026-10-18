"""Evaluation orchestration (single run / multi run), aggregation and quality gates."""

from ragqa.evaluation.runner import RunOptions, run_evaluation, run_sync

__all__ = ["RunOptions", "run_evaluation", "run_sync"]
