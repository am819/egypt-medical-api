"""Adaptive symptom-specific follow-up questions before differential diagnosis."""

from pipeline.context import normalize_text
from pipeline.models import ClinicalAssessment

MAX_FOLLOWUP_QUESTIONS = 3

_FEVER_KW = ["حرارة", "سخونية", "سخونيه", "حراره", "fever"]
_RASH_KW = ["طفح", "هرش", "rash", "حساسية جلد", "بقع", "احمرار"]
_COUGH_KW = ["كحة", "كحه", "cough", "سعال"]
_ABDOMINAL_KW = ["بطن", "معدة", "معده", "abdominal", "مغص", "وجع بطن", "ألم بطن"]


def _text_has_any(text: str, keywords: list[str]) -> bool:
    norm = normalize_text(text)
    return any(normalize_text(k) in norm for k in keywords)


def _text_has_any_group(text: str, groups: list[list[str]]) -> bool:
    """True if at least one keyword from each group is present."""
    norm = normalize_text(text)
    for group in groups:
        if not any(normalize_text(k) in norm for k in group):
            return False
    return True


def _has_fever(assessment: ClinicalAssessment, text: str) -> bool:
    return (
        _text_has_any(text, _FEVER_KW)
        or bool(assessment.severity_indicators)
        and _text_has_any(" ".join(assessment.severity_indicators), _FEVER_KW)
    )


def _has_rash(assessment: ClinicalAssessment, text: str) -> bool:
    symptoms = assessment.main_symptoms + assessment.associated_symptoms
    symptom_text = " ".join(symptoms) + " " + text
    return _text_has_any(symptom_text, _RASH_KW)


def _has_cough(assessment: ClinicalAssessment, text: str) -> bool:
    symptoms = assessment.main_symptoms + assessment.associated_symptoms
    return _text_has_any(" ".join(symptoms) + " " + text, _COUGH_KW)


def _has_abdominal(assessment: ClinicalAssessment, text: str) -> bool:
    symptoms = assessment.main_symptoms + assessment.associated_symptoms
    return _text_has_any(" ".join(symptoms) + " " + text, _ABDOMINAL_KW)


def _fever_rash_questions(text: str) -> list[str]:
    questions = []
    if not _text_has_any_group(text, [
        ["فين", "مكان", "وجه", "جسم", "أطراف", "صدر", "ظهر", "يد", "رجل"],
    ]):
        questions.append("الطفح أو الحساسية ظهرت فين بالظبط؟ (الوجه، الجسم، الأطراف...)")
    if not _text_has_any(text, ["شكل", "بقع", "فقاعات", "متقشر", "احمر", "أحمر", "دوائر", "نقط"]):
        questions.append("شكل الطفح عامل إزاي؟ (بقع حمراء، فقاعات، متقشر...)")
    if not _text_has_any(text, ["هرش", "بيتحك", "حكة", "منتشر", "بيزيد", "بيتنشر", "spread", "itch"]):
        questions.append("فيه هرش أو الطفح بيتنشر بسرعة؟")
    return questions


def _cough_questions(text: str, assessment: ClinicalAssessment) -> list[str]:
    questions = []
    if assessment.cough_type == "unknown" and not _text_has_any(text, [
        "جاف", "جافه", "جافة", "ناشف", "ببلغم", "بلغم", "productive", "dry",
    ]):
        questions.append("الكحة جافة ولا ببلغم؟")
    if not assessment.symptom_duration and not _text_has_any(text, [
        "يوم", "ساعة", "اسبوع", "أسبوع", "من امبارح", "من أمس", "duration",
    ]):
        questions.append("الكحة بدأت من إمتى؟ ومدتها قد إيه؟")
    if (
        assessment.cough_type == "wet"
        or _text_has_any(text, ["ببلغم", "بلغم", "productive"])
    ) and not _text_has_any(text, [
        "أصفر", "اصفر", "أخضر", "اخضر", "أبيض", "ابيض", "دم", "لون",
    ]):
        questions.append("لو في بلغم، لونه إيه؟ (أبيض، أصفر، أخضر...)")
    return questions


def _abdominal_questions(text: str, assessment: ClinicalAssessment) -> list[str]:
    questions = []
    if not _text_has_any(text, [
        "يمين", "شمال", "فوق", "تحت", "سرة", "حوض", "أعلى", "اسفل", "اسفل",
    ]):
        questions.append("الألم فين بالظبط في البطن؟ (يمين، شمال، فوق، تحت...)")
    if not _text_has_any(text, [
        "شديد", "خفيف", "متوسط", "طاعن", "متحمل", "severity", "شدة",
    ]) and not assessment.severity_indicators:
        questions.append("شدة الألم قد إيه؟ (خفيف، متوسط، شديد جداً)")
    if len(assessment.associated_symptoms) < 1 and not _text_has_any(text, [
        "غثيان", "قيء", "ترجيع", "إسهال", "اسهال", "حرارة", "انتفاخ", "جفاف",
    ]):
        questions.append("فيه أعراض تانية مع الألم؟ (غثيان، قيء، إسهال، حرارة...)")
    return questions


def _general_questions(text: str, assessment: ClinicalAssessment) -> list[str]:
    questions = []
    has_symptoms = bool(assessment.main_symptoms or assessment.associated_symptoms)
    if has_symptoms and not assessment.symptom_duration and not _text_has_any(text, [
        "يوم", "ساعة", "اسبوع", "أسبوع", "من امبارح", "من أمس",
    ]):
        questions.append("الأعراض بدأت من إمتى؟ ومدتها قد إيه؟")
    return questions


def get_followup_questions(full_text: str, assessment: ClinicalAssessment) -> list[str]:
    """Return up to 3 symptom-specific follow-up questions still unanswered."""
    collected: list[str] = []

    if _has_fever(assessment, full_text) and _has_rash(assessment, full_text):
        collected.extend(_fever_rash_questions(full_text))

    if _has_cough(assessment, full_text):
        collected.extend(_cough_questions(full_text, assessment))

    if _has_abdominal(assessment, full_text):
        collected.extend(_abdominal_questions(full_text, assessment))

    if not collected:
        collected.extend(_general_questions(full_text, assessment))

    seen: set[str] = set()
    unique: list[str] = []
    for q in collected:
        key = q[:40]
        if key not in seen:
            seen.add(key)
            unique.append(q)

    return unique[:MAX_FOLLOWUP_QUESTIONS]


def information_sufficient_for_dx(full_text: str, assessment: ClinicalAssessment) -> bool:
    """True when enough clinical detail exists to proceed to differential diagnosis."""
    return len(get_followup_questions(full_text, assessment)) == 0


def compute_info_completeness(full_text: str, assessment: ClinicalAssessment) -> float:
    """0.0–1.0 score for how complete the clinical picture is."""
    checks = 0
    passed = 0

    def check(condition: bool) -> None:
        nonlocal checks, passed
        checks += 1
        if condition:
            passed += 1

    check(assessment.age is not None)
    check(assessment.sex != "unknown")
    check(bool(assessment.main_symptoms))
    check(bool(assessment.symptom_duration) or _text_has_any(full_text, ["يوم", "ساعة", "اسبوع"]))
    check(not _has_cough(assessment, full_text) or assessment.cough_type != "unknown"
          or _text_has_any(full_text, ["جاف", "ببلغم", "بلغم"]))
    check(not (_has_fever(assessment, full_text) and _has_rash(assessment, full_text))
          or len(_fever_rash_questions(full_text)) == 0)
    check(not _has_abdominal(assessment, full_text) or len(_abdominal_questions(full_text, assessment)) == 0)

    return passed / checks if checks else 0.5


def format_followup_response(questions: list[str]) -> str:
    lines = [
        "عشان أقدر أقيّم حالتك بدقة، محتاج أعرف شوية تفاصيل:",
        "",
    ]
    for i, q in enumerate(questions, 1):
        lines.append(f"{i}. {q}")
    lines.append("")
    lines.append("جاوبني على اللي تعرفه، ولو في حاجة مش متأكد منها قولي.")
    return "\n".join(lines)
