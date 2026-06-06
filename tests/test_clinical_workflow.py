from pipeline.assessment import merge_assessment
from pipeline.context import extract_context, patient_context_to_assessment
from pipeline.followup import get_followup_questions
from pipeline.models import ClinicalAssessment
from pipeline.red_flags import build_red_flag_section
from pipeline.safety_intake import assess_safety_intake


def test_safety_intake_blocks_before_diagnosis_when_demographics_missing():
    query = "عندي برد وسخونية"
    history = []
    ctx = extract_context(query, history)
    status = assess_safety_intake(ctx, query, history, query)

    assert not status.complete()
    assert "السن" in status.missing_prompts()
    assert "الجنس (ذكر/أنثى)" in status.missing_prompts()


def test_vague_cold_and_fever_requires_symptom_clarification():
    query = (
        "سني 21 ذكر، مفيش أمراض مزمنة، مفيش حساسية، مش باخد أدوية. "
        "عندي برد وسخونية"
    )
    ctx = extract_context(query, [])
    assessment = patient_context_to_assessment(ctx)
    questions = get_followup_questions(query, assessment, history=[])

    assert questions
    assert any("بدأت" in q or "مدتها" in q for q in questions)


def test_age_number_is_not_mapped_to_severity():
    ctx = extract_context("سني 21 ذكر وعندي كحة", [])
    merged = merge_assessment(
        ctx,
        ClinicalAssessment(
            age=21,
            sex="male",
            main_symptoms=["كحة"],
            severity_indicators=["21"],
        ),
    )

    assert merged.age == 21
    assert "21" not in merged.severity_indicators


def test_red_flag_section_does_not_contradict_insufficient_info():
    assessment = ClinicalAssessment(age=21, sex="male", main_symptoms=["سخونية"])
    section = build_red_flag_section(
        assessment,
        "سني 21 ذكر وعندي سخونية",
        "Insufficient information to assess red flags",
    )

    assert "لم يتم جمع معلومات كافية" in section
    assert "مفيش علامات خطر واضحة" not in section


def test_benchmark_suite_has_required_case_volume():
    from eval.benchmark import BENCHMARK_CASES

    assert 100 <= len(BENCHMARK_CASES) <= 200
