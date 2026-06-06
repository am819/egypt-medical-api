"""Enhanced red-flag detection and medical review warnings."""

from pipeline.context import normalize_text
from pipeline.models import ClinicalAssessment

_FEVER_KW = ["حرارة", "سخونية", "سخونيه", "حراره", "fever"]
_RASH_KW = ["طفح", "هرش", "rash", "بقع", "احمرار"]
_SEVERE_KW = ["شديد", "شديدة", "شديده", "severe", "مش قادر", "تعبان جدا"]
_CHRONIC_SEVERE_COMBOS = [
    (["diabetes", "سكر"], "مريض سكري مع أعراض حادة — يحتاج متابعة طبية قريبة"),
    (["hypertension", "ضغط"], "مريض ضغط مع أعراض مقلقة — راقب الأعراض واستشر طبيب"),
    (["heart", "قلب"], "مريض قلب مع أعراض جديدة — يحتاج تقييم طبي"),
    (["kidney", "كلى"], "مريض كلى مع أعراض حادة — احذر واستشر طبيب"),
]


def _text_has_any(text: str, keywords: list[str]) -> bool:
    norm = normalize_text(text)
    return any(normalize_text(k) in norm for k in keywords)


def _has_fever(assessment: ClinicalAssessment, text: str) -> bool:
    return (
        _text_has_any(text, _FEVER_KW)
        or any(_text_has_any(s, _FEVER_KW) for s in assessment.severity_indicators)
    )


def _has_rash(assessment: ClinicalAssessment, text: str) -> bool:
    symptoms = " ".join(assessment.main_symptoms + assessment.associated_symptoms) + " " + text
    return _text_has_any(symptoms, _RASH_KW)


def _has_severe_symptoms(assessment: ClinicalAssessment, text: str) -> bool:
    combined = (
        " ".join(assessment.severity_indicators)
        + " ".join(assessment.main_symptoms)
        + " " + text
    )
    return _text_has_any(combined, _SEVERE_KW) or bool(assessment.red_flags)


def detect_additional_red_flags(
    assessment: ClinicalAssessment,
    full_text: str,
) -> list[str]:
    """Deterministic red-flag and medical-review warnings."""
    flags: list[str] = list(assessment.red_flags)

    if _has_fever(assessment, full_text) and _has_rash(assessment, full_text):
        flags.append(
            "الحرارة مع طفح جلدي قد تحتاج مراجعة طبية عاجلة — "
            "خصوصاً لو الطفح بيتنشر بسرعة أو مع صداع شديد أو تيبس رقبة"
        )

    if _has_severe_symptoms(assessment, full_text) and assessment.chronic_diseases:
        for chronic_keys, warning in _CHRONIC_SEVERE_COMBOS:
            if any(c in assessment.chronic_diseases for c in chronic_keys):
                flags.append(warning)
                break

    if assessment.pregnant and _has_severe_symptoms(assessment, full_text):
        flags.append("حامل مع أعراض شديدة — استشيري طبيب فوراً")

    if assessment.diarrhea_blood:
        flags.append("إسهال مع دم — يحتاج تقييم طبي ولا يُعالج ذاتياً")

    if assessment.cough_type == "wet" and _text_has_any(full_text, ["دم", "blood"]):
        flags.append("كحة ببلغم مع دم — يحتاج مراجعة طبية")

    seen: set[str] = set()
    unique: list[str] = []
    for f in flags:
        key = f[:50]
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return unique


def info_sufficient_for_red_flag_clearance(
    assessment: ClinicalAssessment,
    full_text: str,
) -> bool:
    """Only allow 'no red flags' when enough information has been collected."""
    from pipeline.followup import information_sufficient_for_dx

    if not assessment.main_symptoms:
        return False
    if assessment.age is None:
        return False
    if assessment.sex == "unknown":
        return False
    return information_sufficient_for_dx(full_text, assessment)


def build_red_flag_section(
    assessment: ClinicalAssessment,
    full_text: str,
    llm_assessment: str = "",
) -> str:
    """Build section 6 text — never falsely claim no red flags."""
    flags = detect_additional_red_flags(assessment, full_text)

    if llm_assessment and llm_assessment not in flags:
        flags.append(llm_assessment)

    if not info_sufficient_for_red_flag_clearance(assessment, full_text):
        prefix = (
            "⚠️ لم يتم جمع معلومات كافية بعد لتقييم علامات الخطر بشكل كامل — "
            "راقب الأعراض وأخبرني بأي تغيير."
        )
        if flags:
            return prefix + "\n" + "\n".join(f"🚨 {f}" for f in flags)
        return prefix

    if not flags:
        return (
            "بناءً على المعلومات المتوفرة حالياً، مفيش علامات خطر واضحة — "
            "لكن راقب الأعراض ولو زادت اكشف."
        )

    return "\n".join(f"🚨 {f}" for f in flags)
