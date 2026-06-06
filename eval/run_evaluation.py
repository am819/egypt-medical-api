#!/usr/bin/env python3
"""
Offline evaluation — runs scripted scenarios with mocked Gemini (no API key needed).

Usage:
    python eval/run_evaluation.py
    python eval/run_evaluation.py --live   # optional: one real Gemini call (needs API keys)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
  try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
  except Exception:
    pass

from eval.evaluation_cases import EVALUATION_CASES
from pipeline import dataset
from pipeline.orchestrator import rag
from pipeline.retrieval import lookup_drugs_by_ingredients


def _load_sample_dataset() -> None:
  """Use built-in sample drugs if CSV missing (CI / local dev without data files)."""
  if not dataset.df.empty:
    return
  dataset.df = pd.DataFrame(
    [
      {
        "name_ar": "بانادول 500",
        "name_en": "Panadol 500",
        "ingredient_clean": "paracetamol",
        "active_ingredient": "paracetamol 500mg",
        "combined": "بانادول Panadol paracetamol",
      },
      {
        "name_ar": "بروفين اقراص",
        "name_en": "Brufen tablets",
        "ingredient_clean": "ibuprofen",
        "active_ingredient": "ibuprofen 400mg",
        "combined": "بروفين Brufen ibuprofen",
      },
      {
        "name_ar": "زيرتك 10",
        "name_en": "Zyrtec 10",
        "ingredient_clean": "cetirizine",
        "active_ingredient": "cetirizine 10mg",
        "combined": "زيرتك Zyrtec cetirizine",
      },
      {
        "name_ar": "باسط للمغص",
        "name_en": "Buscopan",
        "ingredient_clean": "hyoscine butylbromide",
        "active_ingredient": "hyoscine butylbromide 10mg",
        "combined": "باسط Buscopan hyoscine",
      },
    ]
  ).reset_index(drop=True)
  dataset.INGREDIENT_COL = "ingredient_clean"


def _count_row_ids(text: str) -> list[int]:
  return [int(m) for m in re.findall(r"\[#(\d+)\]", text)]


def _run_case(case: dict) -> dict:
  status = case.get("mock_status", "ok")
  llm_text = case.get("mock_llm")

  def fake_chat(query, history):
    if status != "ok" or llm_text is None:
      return None, status
    return llm_text, "ok"

  with patch("pipeline.orchestrator.chat", side_effect=fake_chat):
    response = rag(case["user_message"], case.get("history") or [])

  row_ids = _count_row_ids(response)
  checks = []

  for needle in case.get("must_contain", []):
    ok = needle in response
    checks.append(("contains", needle, ok))

  for needle in case.get("must_not_contain", []):
    ok = needle not in response
    checks.append(("not_contains", needle, ok))

  min_drugs = case.get("min_drug_matches", 0)
  checks.append(("min_row_ids", str(min_drugs), len(row_ids) >= min_drugs))

  for expected_id in case.get("expected_row_ids", []):
    checks.append(("row_id", str(expected_id), expected_id in row_ids))

  passed = all(c[2] for c in checks)
  return {
    "id": case["id"],
    "description": case["description"],
    "passed": passed,
    "checks": checks,
    "response_preview": response[:280].replace("\n", " "),
    "row_ids": row_ids,
  }


def run_offline_evaluation() -> int:
  _load_sample_dataset()
  results = [_run_case(c) for c in EVALUATION_CASES]
  passed = sum(1 for r in results if r["passed"])
  total = len(results)

  print("=" * 60)
  print("Egypt Medical API — Offline Evaluation")
  print("=" * 60)
  for r in results:
    icon = "PASS" if r["passed"] else "FAIL"
    print(f"\n[{icon}] {r['id']}: {r['description']}")
    if r["row_ids"]:
      print(f"       row_ids found: {r['row_ids']}")
    print(f"       preview: {r['response_preview'][:120]}...")
    for label, detail, ok in r["checks"]:
      if not ok:
        print(f"       FAIL {label}: {detail}")

  print("-" * 60)
  print(f"Score: {passed}/{total} ({100 * passed // total if total else 0}%)")
  print("=" * 60)
  return 0 if passed == total else 1


def run_live_smoke() -> int:
  from pipeline.gemini import GEMINI_API_KEYS, chat

  if not GEMINI_API_KEYS:
    print("SKIP live smoke: no GEMINI_API_KEY configured")
    return 0

  print("Live smoke: calling Gemini once...")
  text, status = chat("قل مرحباً بجملة واحدة بالعربي", [])
  if not text:
    print(f"FAIL live smoke: status={status}")
    return 1
  print(f"OK live smoke ({len(text)} chars): {text[:100]}...")
  return 0


def main() -> int:
  parser = argparse.ArgumentParser(description="Run offline evaluation suite")
  parser.add_argument("--live", action="store_true", help="Also run one live Gemini call")
  args = parser.parse_args()

  code = run_offline_evaluation()
  if args.live:
    code = max(code, run_live_smoke())
  return code


if __name__ == "__main__":
  raise SystemExit(main())
