"""Central thresholds and gate helpers for the clinical pipeline."""

from pipeline.followup import compute_info_completeness, information_sufficient_for_dx
from pipeline.models import ClinicalAssessment
from pipeline.red_flags import info_sufficient_for_red_flag_clearance

MIN_COMPLETENESS_FOR_DX = 0.65
MIN_COMPLETENESS_FOR_MEDS = 0.75
MIN_COMPLETENESS_FOR_CONFIDENCE = 0.75


def minimum_clinical_dataset_complete(
    full_text: str,
    assessment: ClinicalAssessment,
) -> bool:
    """True when enough data exists for differential diagnosis."""
    if not information_sufficient_for_dx(full_text, assessment):
        return False
    return compute_info_completeness(full_text, assessment) >= MIN_COMPLETENESS_FOR_DX


def can_produce_differential(
    full_text: str,
    assessment: ClinicalAssessment,
) -> bool:
    return minimum_clinical_dataset_complete(full_text, assessment)


def can_recommend_medications(
    full_text: str,
    assessment: ClinicalAssessment,
) -> bool:
    """Medications require full clinical dataset + red-flag clearance."""
    if not minimum_clinical_dataset_complete(full_text, assessment):
        return False
    if compute_info_completeness(full_text, assessment) < MIN_COMPLETENESS_FOR_MEDS:
        return False
    return info_sufficient_for_red_flag_clearance(assessment, full_text)
