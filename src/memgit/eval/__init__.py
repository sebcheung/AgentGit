"""The eval suite: deterministic regression checks over a memory's history.

See ``case.py`` for the suite format and the closed check vocabulary,
``runner.py`` for how one case is evaluated, and ``sweep.py`` for finding
which commit broke one.

**This package imports ``memgit.core`` and ``memgit.retrieval`` and never
``memgit.agent``.** That import graph is the whole design claim made
mechanical: every check here reads the same offline, reproducible surface
``memgit diff``/``memgit show`` already use, so a suite runs with no API key
and no network — the property that makes it usable as a CI gate rather than
a demo. ``tests/test_eval_imports.py`` asserts this stays true.
"""

from __future__ import annotations

from memgit.eval.case import EvalCase, EvalFormatError, EvalSuite, load_suites
from memgit.eval.runner import CaseResult, CheckResult, SuiteResult, run_case, run_suite
from memgit.eval.sweep import FirstFailure, sweep

__all__ = [
    "EvalCase",
    "EvalSuite",
    "EvalFormatError",
    "load_suites",
    "CheckResult",
    "CaseResult",
    "SuiteResult",
    "run_case",
    "run_suite",
    "FirstFailure",
    "sweep",
]
