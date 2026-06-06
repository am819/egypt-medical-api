"""Pytest wrapper for offline evaluation scenarios."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from eval.evaluation_cases import EVALUATION_CASES
from eval.run_evaluation import _load_sample_dataset, _run_case


@pytest.fixture(autouse=True)
def _ensure_sample_data():
  _load_sample_dataset()


@pytest.mark.parametrize("case", EVALUATION_CASES, ids=[c["id"] for c in EVALUATION_CASES])
def test_evaluation_case(case):
  result = _run_case(case)
  assert result["passed"], f"Failed checks: {[c for c in result['checks'] if not c[2]]}"
