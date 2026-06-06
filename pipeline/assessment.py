"""Phase 1-3 assessment merge and validation."""

import json
import re
from typing import Optional

from pipeline.context import (
    CONDITION_KEYWORDS,
    PatientContext,
    dedupe_keep_order,
    normalize_text,
    patient_context_to_assessment,
)
from pipeline.clinical_gates import (
    MIN_COMPLETENESS_FOR_CONFIDENCE,
    MIN_COMPLETENESS_FOR_DX,
    MIN_COMPLETENESS_FOR_MEDS,
)
from pipeline.followup import compute_info_completeness
from pipeline.models import (
    ClinicalAssessment,
    ClinicalPipelineResult,
    DifferentialDiagnosis,
    TherapeuticTarget,
)
from pipeline.red_flags import detect_additional_red_flags

# Prefer broad categories when clinical detail is limited
_SPECIFIC_TO_BROAD: list[tuple[str, str]] = [
    (r"pharyngitis|tonsillitis|strep", "Upper Respiratory Tract Infection"),
    (r"bronchitis|pneumonia", "Lower Respiratory Tract Infection"),
    (r"gastroenteritis|food poisoning", "Acute Gastrointestinal Illness"),
    (r"uti|urinary tract|cystitis", "Urinary Tract Condition"),
    (r"migraine|tension headache", "Headache Syndrome"),
    (r"dermatitis|eczema|urticaria", "Skin Condition"),
    (r"sinusitis|rhinitis", "Upper Respiratory Tract Infection"),
]


def _normalize_chronic_diseases(conditions: list[str]) -> list[str]:
    """Map Arabic/English chronic labels to canonical keys."""
    canonical: list[str] = []
    for cond in conditions:
        norm = normalize_text(cond)
        matched = False
        for key, kws in CONDITION_KEYWORDS.items():
            if norm == key or any(normalize_text(k) in norm for k in kws):
                canonical.append(key)
                matched = True
                break
        if not matched and cond.strip():
            canonical.append(cond.strip())
    return dedupe_keep_order(canonical)


def merge_assessment(regex_ctx: PatientContext, llm: ClinicalAssessment) -> ClinicalAssessment:
    """Merge regex baseline with LLM assessment; regex red flags are never dropped."""
    base = patient_context_to_assessment(regex_ctx)

    age = regex_ctx.age if regex_ctx.age is not None else llm.age
    sex = regex_ctx.sex if regex_ctx.sex != "unknown" else llm.sex
    pregnant = regex_ctx.pregnant if regex_ctx.pregnant is not None else llm.pregnant
    breastfeeding = (
        regex_ctx.breastfeeding if regex_ctx.breastfeeding is not None else llm.breastfeeding
    )

    chronic = _normalize_chronic_diseases(
        dedupe_keep_order(base.chronic_diseases + llm.chronic_diseases)
    )
    allergies = dedupe_keep_order(base.drug_allergies + llm.drug_allergies)
    meds = dedupe_keep_order(base.current_medications + llm.current_medications)
    main_symptoms = dedupe_keep_order(base.main_symptoms + llm.main_symptoms)
    red_flags = dedupe_keep_order(base.red_flags + llm.red_flags)

    cough_type = regex_ctx.cough_type if regex_ctx.cough_type != "unknown" else llm.cough_type
    if cough_type not in ("wet", "dry", "unknown"):
        cough_type = "unknown"

    return ClinicalAssessment(
        age=age,
        sex=sex if sex in ("male", "female", "unknown") else "unknown",
        pregnant=pregnant,
        breastfeeding=breastfeeding,
        chronic_diseases=chronic,
        drug_allergies=allergies,
        current_medications=meds,
        main_symptoms=main_symptoms,
        associated_symptoms=dedupe_keep_order(llm.associated_symptoms),
        symptom_duration=llm.symptom_duration or base.symptom_duration,
        severity_indicators=dedupe_keep_order(base.severity_indicators + llm.severity_indicators),
        red_flags=red_flags,
        cough_type=cough_type,
        diarrhea_blood=regex_ctx.diarrhea_blood or llm.diarrhea_blood,
        diarrhea_fever=regex_ctx.diarrhea_fever or llm.diarrhea_fever,
    )


def min_confidence(conf: str) -> str:
    return "medium" if conf == "high" else conf


def _has_supporting_evidence(
    condition: str,
    assessment: ClinicalAssessment,
    full_text: str,
    completeness: float,
) -> bool:
    """Confidence requires symptoms + duration + context, not label alone."""
    if completeness < MIN_COMPLETENESS_FOR_CONFIDENCE:
        return False
    if not assessment.main_symptoms:
        return False
    if not assessment.symptom_duration and "يوم" not in full_text and "ساع" not in full_text:
        return False
    lowered = condition.lower()
    symptom_text = " ".join(assessment.main_symptoms).lower()
    if "respiratory" in lowered or "pharyng" in lowered:
        return any(s in symptom_text for s in ["كحة", "كحه", "رشح", "حلق", "زكام"])
    if "gastro" in lowered or "intestinal" in lowered:
        return any(s in symptom_text for s in ["بطن", "اسهال", "إسهال", "قيء", "ترجيع", "مغص"])
    if "skin" in lowered or "dermat" in lowered:
        return any(s in symptom_text for s in ["طفح", "هرش", "حساسية"])
    if "headache" in lowered:
        return "صداع" in symptom_text
    return completeness >= 0.85


def _broaden_condition(condition: str, completeness: float) -> str:
    """Replace overly specific labels with broader categories when info is limited."""
    if completeness >= 0.75:
        return condition
    lowered = condition.lower()
    for pattern, broad in _SPECIFIC_TO_BROAD:
        if re.search(pattern, lowered, re.IGNORECASE):
            return broad
    if completeness < 0.5 and len(condition.split()) > 4:
        return " ".join(condition.split()[:3]) + " (احتمالية عامة)"
    return condition


def calibrate_differential(
    differential: DifferentialDiagnosis,
    assessment: ClinicalAssessment,
    full_text: str,
) -> DifferentialDiagnosis:
    """Downgrade overconfident diagnoses; prefer broad categories when info is incomplete."""
    completeness = compute_info_completeness(full_text, assessment)
    calibrated_confidences: list[str] = []
    calibrated_conditions: list[str] = []

    for i, cond in enumerate(differential.possible_conditions):
        raw_conf = (
            differential.confidence_levels[i]
            if i < len(differential.confidence_levels)
            else "medium"
        )

        conf = raw_conf
        if completeness < MIN_COMPLETENESS_FOR_DX:
            conf = "low"
        elif completeness < MIN_COMPLETENESS_FOR_CONFIDENCE:
            conf = "low" if raw_conf == "high" else min_confidence(raw_conf)
        elif completeness < 0.85 and conf == "high":
            conf = "medium"

        if conf in ("medium", "high") and not _has_supporting_evidence(
            cond, assessment, full_text, completeness,
        ):
            conf = "low"

        calibrated_confidences.append(conf)
        calibrated_conditions.append(_broaden_condition(cond, completeness))

    return DifferentialDiagnosis(
        possible_conditions=calibrated_conditions,
        confidence_levels=calibrated_confidences,
        red_flag_assessment=differential.red_flag_assessment,
        requires_urgent_care=differential.requires_urgent_care,
    )


def enrich_clinical_result(
    result: ClinicalPipelineResult,
    full_text: str,
) -> ClinicalPipelineResult:
    """Post-process LLM output: merge red flags, calibrate confidence."""
    extra_flags = detect_additional_red_flags(result.assessment, full_text)
    merged_flags = dedupe_keep_order(result.assessment.red_flags + extra_flags)
    assessment = result.assessment.model_copy(update={"red_flags": merged_flags})
    differential = calibrate_differential(result.differential, assessment, full_text)

    completeness = compute_info_completeness(full_text, assessment)
    status = result.status
    targets = list(result.therapeutic_targets)

    if completeness < MIN_COMPLETENESS_FOR_DX:
        status = "needs_info"
        differential = DifferentialDiagnosis(
            possible_conditions=[],
            confidence_levels=[],
            red_flag_assessment=differential.red_flag_assessment,
            requires_urgent_care=differential.requires_urgent_care,
        )
        targets = []

    if completeness < MIN_COMPLETENESS_FOR_MEDS:
        targets = []

    if status == "ready" and completeness < MIN_COMPLETENESS_FOR_DX:
        status = "needs_info"

    return result.model_copy(update={
        "assessment": assessment,
        "differential": differential,
        "therapeutic_targets": targets,
        "status": status,
    })


def parse_clinical_pipeline(raw_text: str, regex_ctx: PatientContext) -> Optional[ClinicalPipelineResult]:
    """Parse Gemini JSON response into ClinicalPipelineResult."""
    if not raw_text:
        return None
    try:
        text = raw_text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        data = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse clinical JSON: {e}")
        return None

    try:
        assessment = merge_assessment(regex_ctx, ClinicalAssessment(**data.get("assessment", {})))
        diff_data = data.get("differential", {})
        conditions = (diff_data.get("possible_conditions") or [])[:3]
        confidences = (diff_data.get("confidence_levels") or [])[:3]
        while len(confidences) < len(conditions):
            confidences.append("medium")

        differential = DifferentialDiagnosis(
            possible_conditions=conditions,
            confidence_levels=confidences,
            red_flag_assessment=diff_data.get("red_flag_assessment", ""),
            requires_urgent_care=bool(diff_data.get("requires_urgent_care", False)),
        )

        targets = [
            TherapeuticTarget(**t)
            for t in (data.get("therapeutic_targets") or [])
            if isinstance(t, dict) and t.get("target")
        ]

        status = data.get("status", "needs_info")
        if status not in ("needs_info", "ready", "urgent"):
            status = "needs_info"

        result = ClinicalPipelineResult(
            status=status,
            patient_message_ar=data.get("patient_message_ar", ""),
            missing_fields=data.get("missing_fields") or [],
            assessment=assessment,
            differential=differential,
            therapeutic_targets=targets,
            non_drug_advice=data.get("non_drug_advice") or [],
        )
        return result
    except Exception as e:
        print(f"❌ Failed to build ClinicalPipelineResult: {e}")
        return None
