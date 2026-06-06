"""Phase 5 — safety filtering and therapy-class deduplication."""

import re
from typing import Optional

from pipeline.context import dedupe_keep_order
from pipeline.models import ClinicalAssessment, IngredientMatch, SafetyNote

INGREDIENT_SAFETY_RULES = {
    "ibuprofen": {
        "avoid_in_pregnancy": True,
        "avoid_conditions": ["kidney", "ulcer", "hypertension", "diabetes"],
        "caution_conditions": ["heart", "asthma"],
        "min_age": 12,
        "therapy_class": "nsaid",
    },
    "diclofenac": {
        "avoid_in_pregnancy": True,
        "avoid_conditions": ["kidney", "ulcer", "heart", "hypertension", "diabetes"],
        "caution_conditions": ["asthma"],
        "min_age": 14,
        "therapy_class": "nsaid",
    },
    "naproxen": {
        "avoid_in_pregnancy": True,
        "avoid_conditions": ["kidney", "ulcer", "hypertension", "diabetes"],
        "caution_conditions": ["heart"],
        "min_age": 12,
        "therapy_class": "nsaid",
    },
    "pseudoephedrine": {
        "avoid_in_pregnancy": True,
        "avoid_conditions": ["hypertension", "heart"],
        "caution_conditions": ["diabetes"],
        "min_age": 12,
        "therapy_class": "decongestant",
    },
    "loratadine": {"min_age": 2, "therapy_class": "antihistamine"},
    "cetirizine": {"min_age": 2, "therapy_class": "antihistamine"},
    "chlorpheniramine": {"min_age": 2, "therapy_class": "antihistamine"},
    "diphenhydramine": {"min_age": 2, "therapy_class": "antihistamine"},
    "dextromethorphan": {"min_age": 6, "therapy_class": "cough_suppressant"},
    "codeine": {"min_age": 12, "therapy_class": "cough_suppressant"},
    "loperamide": {
        "min_age": 12,
        "avoid_conditions": ["ulcerative colitis"],
        "caution_conditions": ["liver"],
        "therapy_class": "antidiarrheal",
    },
    "paracetamol": {"caution_conditions": ["liver"], "therapy_class": "antipyretic_analgesic"},
    "acetaminophen": {"caution_conditions": ["liver"], "therapy_class": "antipyretic_analgesic"},
    "omeprazole": {"therapy_class": "antacid"},
    "oral rehydration salts": {"therapy_class": "oral_rehydration"},
}

EXCLUDED_FORMS = [
    "vial", "ampoule", "injection", "infusion", "iv", "i.v", "suppository",
    "امبول", "حقن", "وريدي", "امبولة", "لبوس",
]

BABY_KEYWORDS = [
    "teething", "baby", "infant", "toddler", "child",
    "تسنين", "رضع", "أطفال", "طفل", "رضيع",
]

PARACETAMOL_KEYWORDS = ["paracetamol", "acetaminophen", "بانادول", "panadol", "باندول"]

THERAPY_CLASSES: dict[str, str] = {}
for _ing, _rule in INGREDIENT_SAFETY_RULES.items():
    if "therapy_class" in _rule:
        THERAPY_CLASSES[_ing] = _rule["therapy_class"]


def ingredient_rule_keys(active_ingredient: str) -> list:
    ai = active_ingredient.lower()
    return [k for k in INGREDIENT_SAFETY_RULES if k in ai]


def _ingredient_base(active_ingredient: str) -> str:
    return re.split(r"[+/\s\-]", active_ingredient.lower())[0].strip()


def screen_ingredient_safety(active_ingredient: str, assessment: ClinicalAssessment):
    ai, reasons = active_ingredient.lower(), []
    if any(
        allergy.lower() in ai or ai in allergy.lower()
        for allergy in assessment.drug_allergies
    ):
        reasons.append("مستبعد بسبب حساسية مذكورة")
    for key in ingredient_rule_keys(ai):
        rule = INGREDIENT_SAFETY_RULES.get(key, {})
        if "min_age" in rule and assessment.age and assessment.age < rule["min_age"]:
            reasons.append(f"عمر أقل من {rule['min_age']} سنة")
        if rule.get("avoid_in_pregnancy") and assessment.pregnant:
            reasons.append("مستبعد أثناء الحمل")
        if rule.get("avoid_in_breastfeeding") and assessment.breastfeeding:
            reasons.append("مستبعد أثناء الرضاعة")
        for cond in rule.get("avoid_conditions", []):
            if cond in assessment.chronic_diseases:
                reasons.append(f"مستبعد بسبب {cond}")
    if "loperamide" in ai and (assessment.diarrhea_blood or assessment.diarrhea_fever):
        reasons.append("يمنع لوبيراميد مع دم أو حرارة في الإسهال")
    if "dextromethorphan" in ai and assessment.cough_type == "wet":
        reasons.append("دا مضاد سعال - مش مناسب للكحة ببلغم")
    nsaids = ["ibuprofen", "diclofenac", "naproxen"]
    if any(n in ai for n in nsaids) and (
        "hypertension" in assessment.chronic_diseases
        or "diabetes" in assessment.chronic_diseases
    ):
        reasons.append("مضادات الالتهاب ممنوعة تماماً مع الضغط أو السكر")
    return len(reasons) == 0, reasons


def caution_notes_for_context(active_ingredient: str, assessment: ClinicalAssessment) -> list:
    ai, notes = active_ingredient.lower(), []
    for key in ingredient_rule_keys(ai):
        rule = INGREDIENT_SAFETY_RULES.get(key, {})
        for cond in rule.get("caution_conditions", []):
            if cond in assessment.chronic_diseases:
                notes.append(f"يحتاج حذر مع {cond}")
    if assessment.diarrhea_blood or assessment.diarrhea_fever:
        notes.append("الإسهال مع دم/حرارة يستدعي طبيباً")
    if assessment.cough_type == "wet" and "dextromethorphan" in ai:
        notes.append("للكحة ببلغم الأفضل طارد بلغم مش مضاد سعال")
    nsaids = ["ibuprofen", "diclofenac", "naproxen"]
    if any(n in ai for n in nsaids) and (
        "hypertension" in assessment.chronic_diseases
        or "diabetes" in assessment.chronic_diseases
    ):
        notes.append("⚠️ خطير: قد يرفع الضغط ويؤثر على الكلى")
    return dedupe_keep_order(notes)


def is_baby_drug(name_ar: str, name_en: str, age: Optional[int]) -> bool:
    if age is not None and age > 12:
        name_comb = (name_ar + " " + name_en).lower()
        return any(kw in name_comb for kw in BABY_KEYWORDS)
    return False


def has_excluded_form(name_ar: str, name_en: str) -> bool:
    combined = (name_ar + " " + name_en).lower()
    return any(f in combined for f in EXCLUDED_FORMS)


def therapy_class_for_ingredient(ingredient: str) -> Optional[str]:
    ai = ingredient.lower()
    for key, cls in THERAPY_CLASSES.items():
        if key in ai:
            return cls
    return None


def patient_already_on_paracetamol(assessment: ClinicalAssessment) -> bool:
    for med in assessment.current_medications:
        med_l = med.lower()
        if any(kw in med_l for kw in PARACETAMOL_KEYWORDS):
            return True
    return False


def apply_safety_filters(
    matches: list[IngredientMatch],
    assessment: ClinicalAssessment,
) -> tuple[list[IngredientMatch], list[SafetyNote]]:
    """Dedupe therapy classes, active ingredient bases, and paracetamol overlap."""
    notes: list[SafetyNote] = []
    seen_classes: set[str] = set()
    seen_bases: set[str] = set()
    filtered: list[IngredientMatch] = []

    on_paracetamol = patient_already_on_paracetamol(assessment)

    for match in matches:
        base = _ingredient_base(match.ingredient)
        if base in seen_bases:
            continue

        therapy_cls = therapy_class_for_ingredient(match.ingredient)
        if therapy_cls and therapy_cls in seen_classes:
            notes.append(SafetyNote(
                message=f"تم استبعاد {match.ingredient} لتجنب تكرار نفس نوع العلاج",
                severity="info",
            ))
            continue

        if on_paracetamol and base in ("paracetamol", "acetaminophen"):
            notes.append(SafetyNote(
                message="المريض يتناول بالفعل باراسيتامول — تجنب التكرار",
                severity="warning",
            ))
            continue

        seen_bases.add(base)
        if therapy_cls:
            seen_classes.add(therapy_cls)

        if "hypertension" in assessment.chronic_diseases or "diabetes" in assessment.chronic_diseases:
            if therapy_cls == "nsaid":
                continue

        filtered.append(match)

    if assessment.chronic_diseases:
        chronic = ", ".join(assessment.chronic_diseases)
        notes.append(SafetyNote(
            message=f"عندك مرض مزمن ({chronic}) — دي أدوية مؤقتة، لازم متابعة طبيب",
            severity="caution",
        ))

    return filtered, notes
