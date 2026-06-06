"""Therapeutic target registry and symptom-aware pruning."""

import json
from pathlib import Path

from pipeline.models import ClinicalAssessment, TherapeuticTarget

_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "therapeutic_targets.json"

ALLOWED_TARGETS: dict = {}


def _load_registry() -> dict:
    global ALLOWED_TARGETS
    if ALLOWED_TARGETS:
        return ALLOWED_TARGETS
    with open(_DATA_PATH, encoding="utf-8") as f:
        ALLOWED_TARGETS = json.load(f)
    return ALLOWED_TARGETS


def get_target_config(target: str) -> dict | None:
    registry = _load_registry()
    key = target.strip().lower().replace(" ", "_").replace("-", "_")
    return registry.get(key)


def validate_targets(
    targets: list[TherapeuticTarget],
    assessment: ClinicalAssessment,
) -> list[TherapeuticTarget]:
    """Filter LLM targets to registry entries and apply symptom-aware pruning."""
    registry = _load_registry()
    validated: list[TherapeuticTarget] = []
    seen_targets: set[str] = set()

    blocked = _blocked_targets(assessment)

    for t in targets:
        key = t.target.strip().lower().replace(" ", "_").replace("-", "_")
        if key not in registry or key in blocked:
            continue
        if key in seen_targets:
            continue
        seen_targets.add(key)
        validated.append(
            TherapeuticTarget(target=key, rationale=t.rationale, priority=t.priority)
        )

    validated.sort(key=lambda x: (0 if x.priority == "essential" else 1))
    return validated[:3]


def _blocked_targets(assessment: ClinicalAssessment) -> set[str]:
    blocked: set[str] = set()
    chronic = set(assessment.chronic_diseases)

    if assessment.cough_type == "wet":
        blocked.add("cough_suppressant")

    if assessment.diarrhea_blood or assessment.diarrhea_fever:
        blocked.add("antidiarrheal")

    if chronic & {"hypertension", "diabetes", "kidney", "heart", "ulcer"}:
        blocked.add("topical_analgesic")

    if assessment.age is not None and assessment.age < 6:
        blocked.add("cough_suppressant")

    if assessment.age is not None and assessment.age < 12:
        blocked.add("antidiarrheal")

    # NSAID risk — prefer antipyretic over analgesic when chronic disease present
    if chronic & {"hypertension", "diabetes", "kidney"}:
        blocked.add("analgesic")

    return blocked


def ingredient_candidates_for_target(target: str) -> list[str]:
    cfg = get_target_config(target)
    if not cfg:
        return []
    return list(cfg.get("ingredients", []))


def search_keywords_for_target(target: str) -> list[str]:
    cfg = get_target_config(target)
    if not cfg:
        return []
    return list(cfg.get("search_keywords", []))


def therapy_class_for_ingredient(ingredient: str) -> str | None:
    registry = _load_registry()
    ing = ingredient.lower()
    for _target, cfg in registry.items():
        for candidate in cfg.get("ingredients", []):
            if candidate.lower() in ing or ing in candidate.lower():
                return cfg.get("therapy_class")
    return None
