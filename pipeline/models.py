"""Pydantic models for the 7-phase clinical pipeline."""

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class ClinicalAssessment(BaseModel):
    age: Optional[int] = None
    sex: Literal["male", "female", "unknown"] = "unknown"
    pregnant: Optional[bool] = None
    breastfeeding: Optional[bool] = None
    chronic_diseases: list[str] = Field(default_factory=list)
    drug_allergies: list[str] = Field(default_factory=list)
    current_medications: list[str] = Field(default_factory=list)
    main_symptoms: list[str] = Field(default_factory=list)
    associated_symptoms: list[str] = Field(default_factory=list)
    symptom_duration: str = ""
    severity_indicators: list[str] = Field(default_factory=list)
    red_flags: list[str] = Field(default_factory=list)
    cough_type: Literal["wet", "dry", "unknown"] = "unknown"
    diarrhea_blood: bool = False
    diarrhea_fever: bool = False


class DifferentialDiagnosis(BaseModel):
    possible_conditions: list[str] = Field(default_factory=list)
    confidence_levels: list[Literal["high", "medium", "low"]] = Field(default_factory=list)
    red_flag_assessment: str = ""
    requires_urgent_care: bool = False


class TherapeuticTarget(BaseModel):
    target: str
    rationale: str = ""
    priority: Literal["essential", "optional"] = "essential"


class DrugRow(BaseModel):
    row_id: int
    name_ar: str = ""
    name_en: str = ""
    active_ingredient: str = ""
    safety_cautions: list[str] = Field(default_factory=list)


class IngredientMatch(BaseModel):
    target: str
    ingredient: str
    rationale: str = ""
    priority: Literal["essential", "optional"] = "essential"
    drugs: list[DrugRow] = Field(default_factory=list)
    safety_notes: list[str] = Field(default_factory=list)


class SafetyNote(BaseModel):
    message: str
    severity: Literal["info", "caution", "warning"] = "caution"


class ClinicalPipelineResult(BaseModel):
    status: Literal["needs_info", "ready", "urgent"] = "needs_info"
    patient_message_ar: str = ""
    missing_fields: list[str] = Field(default_factory=list)
    assessment: ClinicalAssessment = Field(default_factory=ClinicalAssessment)
    differential: DifferentialDiagnosis = Field(default_factory=DifferentialDiagnosis)
    therapeutic_targets: list[TherapeuticTarget] = Field(default_factory=list)
    non_drug_advice: list[str] = Field(default_factory=list)


def clinical_pipeline_json_schema() -> dict[str, Any]:
    """Gemini responseSchema for structured Phases 1-3."""
    return {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": ["needs_info", "ready", "urgent"],
            },
            "patient_message_ar": {"type": "string"},
            "missing_fields": {
                "type": "array",
                "items": {"type": "string"},
            },
            "assessment": {
                "type": "object",
                "properties": {
                    "age": {"type": "integer", "nullable": True},
                    "sex": {"type": "string", "enum": ["male", "female", "unknown"]},
                    "pregnant": {"type": "boolean", "nullable": True},
                    "breastfeeding": {"type": "boolean", "nullable": True},
                    "chronic_diseases": {"type": "array", "items": {"type": "string"}},
                    "drug_allergies": {"type": "array", "items": {"type": "string"}},
                    "current_medications": {"type": "array", "items": {"type": "string"}},
                    "main_symptoms": {"type": "array", "items": {"type": "string"}},
                    "associated_symptoms": {"type": "array", "items": {"type": "string"}},
                    "symptom_duration": {"type": "string"},
                    "severity_indicators": {"type": "array", "items": {"type": "string"}},
                    "red_flags": {"type": "array", "items": {"type": "string"}},
                    "cough_type": {
                        "type": "string",
                        "enum": ["wet", "dry", "unknown"],
                    },
                    "diarrhea_blood": {"type": "boolean"},
                    "diarrhea_fever": {"type": "boolean"},
                },
                "required": [
                    "sex",
                    "chronic_diseases",
                    "drug_allergies",
                    "current_medications",
                    "main_symptoms",
                    "associated_symptoms",
                    "symptom_duration",
                    "severity_indicators",
                    "red_flags",
                    "cough_type",
                    "diarrhea_blood",
                    "diarrhea_fever",
                ],
            },
            "differential": {
                "type": "object",
                "properties": {
                    "possible_conditions": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "confidence_levels": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["high", "medium", "low"],
                        },
                    },
                    "red_flag_assessment": {"type": "string"},
                    "requires_urgent_care": {"type": "boolean"},
                },
                "required": [
                    "possible_conditions",
                    "confidence_levels",
                    "red_flag_assessment",
                    "requires_urgent_care",
                ],
            },
            "therapeutic_targets": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "target": {"type": "string"},
                        "rationale": {"type": "string"},
                        "priority": {
                            "type": "string",
                            "enum": ["essential", "optional"],
                        },
                    },
                    "required": ["target", "rationale", "priority"],
                },
            },
            "non_drug_advice": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": [
            "status",
            "patient_message_ar",
            "assessment",
            "differential",
            "therapeutic_targets",
        ],
    }
