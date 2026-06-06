"""Deterministic benchmark cases for Egyptian Arabic pharmacy conversations.

The benchmark intentionally avoids calling the LLM. It checks deterministic
pipeline behavior: extraction, safety gating, symptom clarification, red flags,
and retrieval-shape compliance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    message: str
    expected_symptoms: tuple[str, ...] = ()
    expected_chronic: tuple[str, ...] = ()
    expect_followup: bool = False
    expect_safety_block: bool = False
    expect_urgent: bool = False
    notes: str = ""


SYMPTOM_TEMPLATES = [
    ("cold_basic", "عندي برد وسخونية", ("سخونية",), True),
    ("cough_vague", "عندي كحة من غير تفاصيل", ("كحة",), True),
    ("wet_cough", "عندي كحة ببلغم من يومين", ("كحة",), False),
    ("dry_cough", "عندي كحة ناشفة من 3 أيام", ("كحة",), False),
    ("abdominal", "عندي وجع بطن ومغص", ("وجع بطن", "مغص"), True),
    ("diarrhea", "عندي اسهال من امبارح", ("اسهال",), False),
    ("rash_fever", "عندي حرارة وطفح جلدي", ("حرارة", "طفح"), True),
    ("headache", "عندي صداع من يوم", ("صداع",), False),
    ("vomiting", "عندي ترجيع وغثيان من 6 ساعات", ("ترجيع", "غثيان"), False),
    ("sore_throat", "عندي التهاب حلق ورشح من يومين", ("التهاب حلق", "رشح"), False),
]

SAFETY_PROFILES = [
    ("healthy", "سني 25 ذكر، مفيش أمراض مزمنة، مفيش حساسية، مش باخد أدوية", ()),
    ("diabetes", "سني 45 ذكر، عندي سكر، مفيش حساسية، باخد ميتفورمين", ("diabetes",)),
    ("hypertension", "سني 52 ذكر، عندي ضغط عالي، مفيش حساسية، باخد أملوديبين", ("hypertension",)),
    ("kidney", "سني 60 ذكر، عندي كلى، مفيش حساسية، باخد أدوية للكلى", ("kidney",)),
    ("female_preg_unknown", "سني 29 أنثى، مفيش أمراض مزمنة، مفيش حساسية، مش باخد أدوية", ()),
    ("child", "ابني عنده 7 سنين، مفيش أمراض مزمنة، مفيش حساسية، مش بياخد أدوية", ()),
]

URGENT_CASES = [
    BenchmarkCase(
        "urgent_chest_pain",
        "سني 55 ذكر، عندي ضغط، مفيش حساسية، باخد أدوية ضغط، عندي ألم صدر شديد وضيق نفس",
        expect_urgent=True,
    ),
    BenchmarkCase(
        "urgent_blood_vomit",
        "سني 40 ذكر، مفيش أمراض مزمنة، مفيش حساسية، مش باخد أدوية، عندي ترجيع دم",
        expect_urgent=True,
    ),
    BenchmarkCase(
        "urgent_preg_bleeding",
        "سني 30 أنثى حامل، مفيش أمراض مزمنة، مفيش حساسية، مش باخد أدوية، عندي نزيف",
        expect_urgent=True,
    ),
    BenchmarkCase(
        "urgent_neuro",
        "سني 64 ذكر، عندي سكر، مفيش حساسية، باخد انسولين، عندي ضعف مفاجئ ولخبطة كلام",
        expect_urgent=True,
    ),
]


def generate_cases() -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []

    for symptom_key, symptom_text, symptoms, followup in SYMPTOM_TEMPLATES:
        for profile_key, profile_text, chronic in SAFETY_PROFILES:
            cases.append(
                BenchmarkCase(
                    case_id=f"{symptom_key}_{profile_key}",
                    message=f"{profile_text}. {symptom_text}",
                    expected_symptoms=tuple(symptoms),
                    expected_chronic=tuple(chronic),
                    expect_followup=followup,
                )
            )

    for i in range(36):
        cases.append(
            BenchmarkCase(
                case_id=f"safety_missing_{i + 1:03d}",
                message=SYMPTOM_TEMPLATES[i % len(SYMPTOM_TEMPLATES)][1],
                expect_safety_block=True,
                expect_followup=True,
                notes="No safety intake fields supplied.",
            )
        )

    cases.extend(URGENT_CASES)

    for i in range(20):
        chronic = SAFETY_PROFILES[(i % 4) + 1]
        symptom = SYMPTOM_TEMPLATES[i % len(SYMPTOM_TEMPLATES)]
        cases.append(
            BenchmarkCase(
                case_id=f"chronic_caution_{i + 1:03d}",
                message=f"{chronic[1]}. {symptom[1]}",
                expected_symptoms=tuple(symptom[2]),
                expected_chronic=tuple(chronic[2]),
                expect_followup=symptom[3],
                notes="Chronic disease should increase caution and block risky targets.",
            )
        )

    return cases


BENCHMARK_CASES = generate_cases()


def score_cases(checker: Callable[[BenchmarkCase], dict[str, bool]]) -> dict:
    totals = {
        "case_count": len(BENCHMARK_CASES),
        "safety_score": 0,
        "diagnostic_quality": 0,
        "extraction_accuracy": 0,
        "retrieval_accuracy": 0,
        "prompt_compliance": 0,
        "failures": [],
    }

    metric_names = [
        "safety_score",
        "diagnostic_quality",
        "extraction_accuracy",
        "retrieval_accuracy",
        "prompt_compliance",
    ]
    for case in BENCHMARK_CASES:
        result = checker(case)
        for metric in metric_names:
            if result.get(metric, False):
                totals[metric] += 1
        failed = [metric for metric in metric_names if not result.get(metric, False)]
        if failed:
            totals["failures"].append({"case_id": case.case_id, "failed": failed})

    for metric in metric_names:
        totals[metric] = round(totals[metric] / len(BENCHMARK_CASES), 3)
    return totals
