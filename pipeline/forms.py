"""Phase 6 — route and dosage form validation."""

from pipeline.context import normalize_text
from pipeline.models import ClinicalAssessment, DrugRow, IngredientMatch

FORM_RULES: dict[str, dict[str, list[str]]] = {
    "sore_throat": {
        "prefer": ["lozenge", "lozenges", "mouthwash", "gargle", "syrup", "tablet", "spray", "قرص", "استحلاب", "غرغرة", "مضمضة"],
        "exclude": ["cream", "ointment", "eye drop", "injection", "suppository", "كريم", "مرهم", "قطرة عين", "حقن", "لبوس"],
    },
    "cough": {
        "prefer": ["syrup", "tablet", "capsule", "suspension", "شراب", "كبسول", "قرص"],
        "exclude": ["cream", "eye drop", "injection", "suppository", "كريم", "قطرة عين", "حقن"],
    },
    "fever": {
        "prefer": ["tablet", "syrup", "capsule", "suppository", "قرص", "شراب", "كبسول"],
        "exclude": ["cream", "eye drop", "injection", "كريم", "قطرة عين", "حقن وريدي"],
    },
    "diarrhea": {
        "prefer": ["tablet", "capsule", "sachet", "powder", "solution", "قرص", "كبسول", "أكياس"],
        "exclude": ["cream", "eye drop", "injection", "كريم", "قطرة عين"],
    },
    "nausea": {
        "prefer": ["tablet", "syrup", "suppository", "قرص", "شراب"],
        "exclude": ["cream", "eye drop", "injection", "كريم"],
    },
    "heartburn": {
        "prefer": ["tablet", "syrup", "suspension", "sachet", "قرص", "شراب"],
        "exclude": ["eye drop", "injection", "cream", "قطرة عين", "حقن"],
    },
    "rash": {
        "prefer": ["cream", "ointment", "lotion", "gel", "tablet", "كريم", "مرهم", "لوشن"],
        "exclude": ["eye drop", "injection", "قطرة عين", "حقن"],
    },
    "nasal": {
        "prefer": ["nasal", "spray", "drops", "solution", "بخاخ", "قطرة أنف", "محلول"],
        "exclude": ["cream", "tablet", "injection", "كريم", "حقن"],
    },
}

SYMPTOM_TO_CATEGORY: dict[str, str] = {
    "التهاب حلق": "sore_throat",
    "حرارة": "fever",
    "سخونية": "fever",
    "كحة": "cough",
    "كحه": "cough",
    "إسهال": "diarrhea",
    "اسهال": "diarrhea",
    "غثيان": "nausea",
    "قيء": "nausea",
    "ترجيع": "nausea",
    "حموضة": "heartburn",
    "طفح": "rash",
    "هرش": "rash",
    "رشح": "nasal",
    "احتقان": "nasal",
    "زكام": "nasal",
}

TARGET_TO_CATEGORY: dict[str, str] = {
    "throat_antiseptic": "sore_throat",
    "mucolytic": "cough",
    "cough_suppressant": "cough",
    "antipyretic": "fever",
    "analgesic": "fever",
    "antidiarrheal": "diarrhea",
    "oral_rehydration": "diarrhea",
    "antiemetic": "nausea",
    "antacid": "heartburn",
    "antipruritic": "rash",
    "topical_analgesic": "rash",
    "decongestant": "nasal",
    "nasal_saline": "nasal",
    "antihistamine": "nasal",
}


def detect_complaint_categories(assessment: ClinicalAssessment) -> list[str]:
    categories: list[str] = []
    all_symptoms = assessment.main_symptoms + assessment.associated_symptoms
    for symptom in all_symptoms:
        norm = normalize_text(symptom)
        for key, cat in SYMPTOM_TO_CATEGORY.items():
            if normalize_text(key) in norm and cat not in categories:
                categories.append(cat)
    if not categories:
        categories.append("fever")
    return categories


def _form_score(drug: DrugRow, rules: dict[str, list[str]]) -> int:
    combined = (drug.name_ar + " " + drug.name_en).lower()
    for ex in rules.get("exclude", []):
        if ex.lower() in combined:
            return -100
    score = 0
    for pref in rules.get("prefer", []):
        if pref.lower() in combined:
            score += 10
    return score


def filter_drugs_for_form(drug: DrugRow, categories: list[str], target: str) -> bool:
    """Return True if drug passes form validation."""
    cat = TARGET_TO_CATEGORY.get(target)
    if cat and cat not in categories:
        categories = categories + [cat]

    combined = (drug.name_ar + " " + drug.name_en).lower()
    for category in categories:
        rules = FORM_RULES.get(category, {})
        for ex in rules.get("exclude", []):
            if ex.lower() in combined:
                return False
    return True


def apply_form_filters(
    matches: list[IngredientMatch],
    assessment: ClinicalAssessment,
) -> list[IngredientMatch]:
    categories = detect_complaint_categories(assessment)
    filtered: list[IngredientMatch] = []

    for match in matches:
        valid_drugs = [d for d in match.drugs if filter_drugs_for_form(d, categories, match.target)]
        if not valid_drugs:
            continue

        cat = TARGET_TO_CATEGORY.get(match.target)
        rules = FORM_RULES.get(cat or categories[0], {})
        valid_drugs.sort(key=lambda d: _form_score(d, rules), reverse=True)
        valid_drugs = [d for d in valid_drugs if _form_score(d, rules) >= 0]

        if not valid_drugs:
            continue

        filtered.append(IngredientMatch(
            target=match.target,
            ingredient=match.ingredient,
            rationale=match.rationale,
            priority=match.priority,
            drugs=valid_drugs[:2],
            safety_notes=match.safety_notes,
        ))

    return filtered
