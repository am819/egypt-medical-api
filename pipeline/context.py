"""Regex-based patient context extraction and intake gate."""

import re
from dataclasses import dataclass, field
from typing import Optional

from pipeline.models import ClinicalAssessment


AR_NUMS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

COMMON_SYMPTOMS = [
    "حرارة", "سخونية", "كحة", "كحه", "رشح", "احتقان", "التهاب حلق", "زكام",
    "إسهال", "اسهال", "ترجيع", "قيء", "غثيان", "مغص", "صداع", "دوخة",
    "ضيق نفس", "وجع صدر", "ألم صدر", "حرقان بول", "حرقان", "طفح", "هرش",
    "ألم بطن", "وجع بطن", "ألم معدة", "حموضة", "إمساك", "امساك", "ألم ظهر",
]

REAL_RED_FLAG_PATTERNS = [
    (r"ضيق نفس شديد|مش عارف اتنفس|نهجان شديد|اختناق", "ضيق نفس شديد"),
    (r"ألم صدر شديد|وجع صدر شديد|ضغط على الصدر", "ألم صدر شديد"),
    (r"فقدان وعي|اغماء|إغماء|مش واعي", "إغماء/فقدان وعي"),
    (r"تشنجات|convulsion|seizure", "تشنجات"),
    (r"قيء دم|ترجيع دم|دم في القيء", "قيء دم"),
    (r"براز أسود|دم في البراز|نزيف شرجي", "نزيف هضمي محتمل"),
    (r"ضعف مفاجئ|ميل في الوجه|لخبطة كلام|تلعثم مفاجئ", "أعراض عصبية حادة"),
    (r"طفح .* مع ضيق نفس|تورم اللسان|تورم الشفايف", "حساسية شديدة محتملة"),
    (r"حرارة فوق ?40|سخونية فوق ?40", "حرارة شديدة جدًا"),
    (r"جفاف شديد|مش بيشرب|قلة بول شديدة|مفيش بول", "جفاف شديد"),
    (r"ألم بطن شديد جدًا|بطن ناشفة|تيبس البطن", "ألم بطن حاد"),
    (r"حامل.*نزيف|نزيف.*حامل", "نزيف أثناء الحمل"),
]

CONDITION_KEYWORDS = {
    "kidney": ["كلى", "فشل كلوي", "renal", "kidney"],
    "liver": ["كبد", "التهاب كبد", "cirrhosis", "liver"],
    "ulcer": ["قرحة", "نزيف معدة", "قرحة معدة"],
    "hypertension": ["ضغط", "ضغط عالي", "hypertension"],
    "diabetes": ["سكر", "سكري", "diabetes"],
    "asthma": ["ربو", "asthma"],
    "heart": ["قلب", "فشل قلبي", "ذبحة", "heart"],
}

REAL_URGENT_FLAGS = [
    "ضيق نفس شديد", "ألم صدر شديد", "إغماء", "تشنجات",
    "قيء دم", "نزيف", "شلل مفاجئ", "حرارة فوق 40",
]


@dataclass
class PatientContext:
    age: Optional[int] = None
    sex: str = "unknown"
    pregnant: Optional[bool] = None
    breastfeeding: Optional[bool] = None
    duration_text: str = ""
    fever_text: str = ""
    symptoms: list = field(default_factory=list)
    allergies: list = field(default_factory=list)
    chronic_conditions: list = field(default_factory=list)
    current_meds: list = field(default_factory=list)
    complaint_text: str = ""
    red_flags: list = field(default_factory=list)
    cough_type: str = ""
    diarrhea_blood: bool = False
    diarrhea_fever: bool = False
    dental_swelling: bool = False
    dental_pus: bool = False
    is_caregiver: bool = False
    child_age: Optional[int] = None


def normalize_text(text: str) -> str:
    text = (text or "").translate(AR_NUMS)
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    text = text.replace("ة", "ه").replace("ى", "ي")
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def dedupe_keep_order(items: list) -> list:
    seen, out = set(), []
    for item in items:
        v = item.strip()
        if not v:
            continue
        key = v.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(v)
    return out


def conversation_to_text(history: list) -> str:
    return "\n".join(
        msg.get("content", "") for msg in (history or []) if msg.get("role") == "user"
    )


def arabic_words_to_number(words: str) -> Optional[int]:
    word_map = {
        "واحد": 1, "واحده": 1, "واحدة": 1, "اثنان": 2, "اثنين": 2, "اتنين": 2,
        "ثلاثة": 3, "تلاته": 3, "تلاتة": 3, "اربع": 4, "اربعة": 4, "اربعه": 4,
        "أربع": 4, "أربعة": 4, "خمسة": 5, "خمسه": 5, "ستة": 6, "سته": 6,
        "سبعة": 7, "سبعه": 7, "ثمانية": 8, "ثمانيه": 8, "تسعة": 9, "تسعه": 9,
        "عشرة": 10, "عشره": 10,
    }
    for w, num in word_map.items():
        if w in words:
            return num
    return None


def extract_age(text: str) -> Optional[int]:
    text = text.translate(AR_NUMS)
    patterns = [
        r"(\d{1,3})\s*سنه", r"عندي\s*(\d{1,3})\s*سنه",
        r"السن\s*(\d{1,3})", r"age\s*[:=]?\s*(\d{1,3})",
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            age = int(m.group(1))
            if 0 < age < 120:
                return age
    for n in re.findall(r"\b(\d{1,3})\b", text):
        age = int(n)
        if age < 1 or age > 110:
            continue
        if re.search(rf"{age}\s*ساعة", text) or re.search(rf"{age}\s*يوم", text):
            continue
        if "سنة" in text or "سنين" in text:
            return age
        if age >= 10:
            return age
    return None


def extract_sex(text: str) -> str:
    norm = normalize_text(text)
    if any(w in norm for w in ["ذكر", "راجل", "male"]):
        return "male"
    if any(w in norm for w in ["انثى", "انثي", "ست", "بنت", "female", "حامل", "مرضع"]):
        return "female"
    return "unknown"


def extract_duration(text: str) -> str:
    text = normalize_text(text)
    m = re.search(r"من\s+([^\s]+)\s+(يوم|ساعة|ساعه|ساعات|ايام|أيام|دقيقة|دقائق)", text)
    if m:
        num_word, unit = m.group(1), m.group(2)
        num = int(num_word) if num_word.isdigit() else arabic_words_to_number(num_word)
        if num:
            if "دقيق" in unit:
                return f"{num} دقيقة"
            if "ساع" in unit:
                return f"{num} ساعة"
            return f"{num} يوم"
    m = re.search(r"(\d+|واحد|اثنين|اتنين|ثلاثة|تلاته|اربع|اربعة)\s+(يوم|ساعة|دقيقة|دقائق)", text)
    if m:
        num_word, unit = m.group(1), m.group(2)
        num = int(num_word) if num_word.isdigit() else arabic_words_to_number(num_word)
        if num:
            if "دقيق" in unit:
                return f"{num} دقيقة"
            if "ساع" in unit:
                return f"{num} ساعة"
            return f"{num} يوم"
    if re.search(r"يومين", text):
        return "يومين"
    if re.search(r"دلوقتي|النهارده|لسه من شويه", text):
        return "أقل من يوم"
    if re.search(r"امبارح|أمس", text):
        return "يوم"
    return ""


def has_negation_response(text: str) -> bool:
    norm = normalize_text(text)
    return any(neg in norm for neg in ["لا", "مفيش", "مش", "مافيش", "لأ", "لالا", "لا لا"])


def parse_pregnancy_breastfeeding(text: str):
    norm = normalize_text(text)
    pregnant, breastfeeding = None, None
    if re.search(r"(لا|مش|مفيش|لأ|لست)\s*(حامل|حامل؟)", norm) or "غير حامل" in norm or "مش حامل" in norm:
        pregnant = False
    if re.search(r"انا حامل|بنت حامل|حامل في", norm):
        pregnant = True
    if re.search(r"(لا|مش|مفيش|لأ)\s*(مرضع|ترضع|رضاعة)", norm):
        breastfeeding = False
    if re.search(r"(مرضع|برضع|باخد رضاعة)", norm):
        breastfeeding = True
    return pregnant, breastfeeding


def extract_list_after_keywords(text: str, keywords: list, check_negation: bool = True) -> list:
    if check_negation and has_negation_response(text):
        return []
    found = []
    for kw in keywords:
        m = re.search(rf"{kw}\s*[:：]?\s*([^\n\.،]+)", text, re.IGNORECASE)
        if m:
            raw = m.group(1)
            if has_negation_response(raw):
                return []
            for item in re.split(r"[,،/|+]", raw):
                item = item.strip()
                if item and not has_negation_response(item):
                    found.append(item)
    return dedupe_keep_order(found)


def _has_hypertension_context(norm: str) -> bool:
    """Avoid false positives from chest pressure (ضغط على الصدر)."""
    if re.search(r"ضغط\s*(على\s*)?الصدر|وجع\s*صدر|الم\s*صدر|صدر.*ضغط", norm):
        return False
    return bool(re.search(
        r"ضغط\s*عالي|امراض\s*ضغط|مرض\s*ضغط|مريض\s*ضغط|عندي\s*ضغط|ارتفاع\s*ضغط|hypertension",
        norm,
    ))


def extract_conditions(text: str) -> list:
    norm = normalize_text(text)
    found: list[str] = []
    for canonical, kws in CONDITION_KEYWORDS.items():
        if canonical == "hypertension":
            if _has_hypertension_context(norm):
                found.append(canonical)
            continue
        if any(k in norm for k in kws):
            found.append(canonical)
    return dedupe_keep_order(found)


def extract_symptoms(text: str) -> list:
    norm = normalize_text(text)
    return [s for s in COMMON_SYMPTOMS if normalize_text(s) in norm]


def extract_cough_type(text: str) -> str:
    if re.search(r"ببلغم|معاها بلغم|كحة بلغم|كحه بلغم", text, re.IGNORECASE):
        return "wet"
    if re.search(r"جافه|جافة|ناشفة|كحة جافه|كحه ناشفه", text, re.IGNORECASE):
        return "dry"
    return "unknown"


def extract_diarrhea_flags(text: str):
    blood = bool(re.search(r"دم في البراز|براز فيه دم|دم مع البراز", text, re.IGNORECASE))
    fever = bool(re.search(r"حراره|سخونيه", text, re.IGNORECASE))
    return blood, fever


def extract_dental_flags(text: str):
    swelling = bool(re.search(r"تورم|ورم في اللثه|وجه وارم", text, re.IGNORECASE))
    pus = bool(re.search(r"صديد|ريحة كريهه|إفرازات", text, re.IGNORECASE))
    return swelling, pus


def extract_context(query: str, history: list) -> PatientContext:
    full_text = (conversation_to_text(history) + "\n" + query).strip()
    norm = normalize_text(full_text)
    is_caregiver = bool(re.search(r"ابني|بنتي|طفلي|العيل|البنت|الولد", norm))
    child_age = None
    if is_caregiver:
        m = re.search(r"عنده\s*(\d+)\s*سنه", norm)
        if m:
            child_age = int(m.group(1))
    sex = extract_sex(full_text)
    pregnant, breastfeeding = parse_pregnancy_breastfeeding(full_text)
    duration_text = extract_duration(full_text)
    fever_match = re.search(
        r"(حراره\s*\d+(?:\.\d+)?|سخونيه\s*\d+(?:\.\d+)?|حراره|سخونيه|سخونية)",
        norm, re.IGNORECASE,
    )
    fever_text = fever_match.group(1).strip() if fever_match else ""
    allergies = extract_list_after_keywords(
        full_text, ["حساسيه", "allergy", "allergies", "allergic to"],
        check_negation=False,
    )
    current_meds = extract_list_after_keywords(
        full_text, ["ادويه", "ادوية", "meds", "medications", "باخد", "باخد علاج"],
    )
    chronic_conditions = extract_conditions(full_text)
    symptoms = extract_symptoms(full_text)
    red_flags = []
    for pattern, label in REAL_RED_FLAG_PATTERNS:
        if re.search(pattern, norm, re.IGNORECASE):
            red_flags.append(label)
    cough_type = extract_cough_type(full_text)
    diarrhea_blood, diarrhea_fever = extract_diarrhea_flags(full_text)
    dental_swelling, dental_pus = extract_dental_flags(full_text)
    age = extract_age(full_text)
    if child_age and not age:
        age = child_age
    return PatientContext(
        age=age, sex=sex, pregnant=pregnant, breastfeeding=breastfeeding,
        duration_text=duration_text, fever_text=fever_text, symptoms=symptoms,
        allergies=allergies, chronic_conditions=chronic_conditions, current_meds=current_meds,
        complaint_text=query.strip(), red_flags=dedupe_keep_order(red_flags),
        cough_type=cough_type, diarrhea_blood=diarrhea_blood, diarrhea_fever=diarrhea_fever,
        dental_swelling=dental_swelling, dental_pus=dental_pus,
        is_caregiver=is_caregiver, child_age=child_age,
    )


def has_real_emergency(ctx: PatientContext) -> bool:
    text = (ctx.complaint_text + " " + " ".join(ctx.symptoms)).lower()
    return any(flag in text for flag in REAL_URGENT_FLAGS) or bool(ctx.red_flags)


def is_greeting(text: str) -> bool:
    greeting_words = [
        "هاي", "هلا", "سلام", "عامل ايه", "عامل اي", "ازيك", "اخبارك", "صباح", "مساء", "يعمعلم",
    ]
    text_norm = normalize_text(text)
    if len(text_norm) < 15 or any(g in text_norm for g in greeting_words):
        if not any(normalize_text(s) in text_norm for s in COMMON_SYMPTOMS[:10]):
            return True
    return False


def intake_gate(ctx: PatientContext, history: list, query: str) -> Optional[str]:
    """Emergency and greeting handling only — mandatory safety intake is in safety_intake.py."""
    if has_real_emergency(ctx):
        return "🚨 فيه علامات خطر حقيقية - لازم تروح الطوارئ فوراً."
    if is_greeting(query) and ctx.age is None and not ctx.symptoms:
        return (
            "أهلاً بك! أنا دكتور مساعد. عرفني على أعراضك عشان أقدر أساعدك.\n\n"
            "قبل أي تقييم، محتاج أعرف: السن، الجنس، الأمراض المزمنة، الحساسية، والأدوية الحالية."
        )
    return None


def build_patient_summary(ctx: PatientContext) -> str:
    def val(v, fallback="غير مذكور"):
        if v is None:
            return fallback
        if isinstance(v, list):
            return ", ".join(v) if v else fallback
        if isinstance(v, bool):
            return "نعم" if v else "لا"
        return str(v).strip() or fallback

    return f"""ملخص الحالة المهيكل:
- العمر: {val(ctx.age)}
- الجنس: {val(ctx.sex)}
- حمل/رضاعة: {val(ctx.pregnant or ctx.breastfeeding)}
- مدة الأعراض (إذا ذكرت): {val(ctx.duration_text)}
- الحرارة: {val(ctx.fever_text)}
- الأعراض المستخرجة: {val(ctx.symptoms)}
- الحساسية: {val(ctx.allergies)}
- الأمراض المزمنة: {val(ctx.chronic_conditions)}
- الأدوية الحالية: {val(ctx.current_meds)}
- علامات خطر: {val(ctx.red_flags)}
- نوع الكحة: {ctx.cough_type}
- دم/حرارة بالإسهال: {val(ctx.diarrhea_blood or ctx.diarrhea_fever)}
"""


def patient_context_to_assessment(ctx: PatientContext) -> ClinicalAssessment:
    cough = ctx.cough_type if ctx.cough_type in ("wet", "dry", "unknown") else "unknown"
    sex = ctx.sex if ctx.sex in ("male", "female", "unknown") else "unknown"
    return ClinicalAssessment(
        age=ctx.age,
        sex=sex,
        pregnant=ctx.pregnant,
        breastfeeding=ctx.breastfeeding,
        chronic_diseases=list(ctx.chronic_conditions),
        drug_allergies=list(ctx.allergies),
        current_medications=list(ctx.current_meds),
        main_symptoms=list(ctx.symptoms),
        associated_symptoms=[],
        symptom_duration=ctx.duration_text,
        severity_indicators=[ctx.fever_text] if ctx.fever_text else [],
        red_flags=list(ctx.red_flags),
        cough_type=cough,
        diarrhea_blood=ctx.diarrhea_blood,
        diarrhea_fever=ctx.diarrhea_fever,
    )
