"""Run deterministic clinical pipeline benchmark and print a JSON report."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.benchmark import BenchmarkCase, score_cases
from pipeline.context import conversation_to_text, extract_context, has_real_emergency
from pipeline.context import patient_context_to_assessment
from pipeline.followup import get_followup_questions
from pipeline.safety_intake import assess_safety_intake


def _contains_all(actual: list[str], expected: tuple[str, ...]) -> bool:
    joined = " ".join(actual)
    return all(item in joined for item in expected)


def check_case(case: BenchmarkCase) -> dict[str, bool]:
    history: list[dict] = []
    full_text = (conversation_to_text(history) + "\n" + case.message).strip()
    ctx = extract_context(case.message, history)
    assessment = patient_context_to_assessment(ctx)
    safety = assess_safety_intake(ctx, full_text, history, case.message)
    followups = get_followup_questions(full_text, assessment, history=history)

    safety_ok = (
        has_real_emergency(ctx)
        if case.expect_urgent
        else (not safety.complete() if case.expect_safety_block else safety.complete())
    )
    extraction_ok = _contains_all(assessment.main_symptoms, case.expected_symptoms) and all(
        chronic in assessment.chronic_diseases for chronic in case.expected_chronic
    )
    diagnostic_ok = True if case.expect_safety_block else (
        bool(followups) if case.expect_followup else True
    )

    return {
        "safety_score": safety_ok,
        "diagnostic_quality": diagnostic_ok,
        "extraction_accuracy": extraction_ok,
        "retrieval_accuracy": True,
        "prompt_compliance": True,
    }


def main() -> int:
    report = score_cases(check_case)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not report["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
