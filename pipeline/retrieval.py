"""Phase 4 — dataset-constrained drug retrieval by therapeutic target."""

import re
from rapidfuzz import process

from pipeline import dataset
from pipeline.models import ClinicalAssessment, DrugRow, IngredientMatch, TherapeuticTarget
from pipeline.safety import (
    caution_notes_for_context,
    has_excluded_form,
    is_baby_drug,
    screen_ingredient_safety,
)
from pipeline.targets import ingredient_candidates_for_target, search_keywords_for_target


def _score_row_for_target(row: dict, target_keywords: list[str]) -> int:
    combined = (
        row.get("combined", "")
        or (row.get("name_ar", "") + " " + row.get("name_en", "") + " " + row.get(dataset.INGREDIENT_COL, ""))
    ).lower()
    score = 0
    for kw in target_keywords:
        if kw.lower() in combined:
            score += 10
    return score


def get_matching_drugs_for_ingredient(
    ingredient: str,
    assessment: ClinicalAssessment,
    target_keywords: list[str] | None = None,
    max_results: int = 2,
) -> list[DrugRow]:
    if dataset.df.empty:
        return []
    skip_terms = {"useful for cough", "useful for pain", "علاج", "دواء", "unknown"}
    if ingredient in skip_terms or len(ingredient) < 3:
        return []

    keywords = target_keywords or []
    query = ingredient + " " + " ".join(keywords)
    cand_ids = dataset.semantic_candidate_indices(query, top_k=60)
    cand_texts = [(idx, dataset.df.iloc[idx].get(dataset.INGREDIENT_COL, "")) for idx in cand_ids]
    if not cand_texts:
        return []

    fuzzy = process.extract(ingredient, [t[1] for t in cand_texts], limit=30)
    base_map: dict[str, list] = {}

    for hit in fuzzy:
        pos, score = hit[2], hit[1]
        if score < 85:
            continue
        idx = cand_texts[pos][0]
        row = dataset.df.iloc[idx]
        ai = row.get(dataset.INGREDIENT_COL, "").strip().lower()
        name_en = row.get("name_en", "")
        name_ar = row.get("name_ar", "")
        if is_baby_drug(name_ar, name_en, assessment.age):
            continue
        if has_excluded_form(name_ar, name_en):
            continue
        allowed, _ = screen_ingredient_safety(ai, assessment)
        if not allowed:
            continue
        base = re.split(r"[+/\s\-]", ai)[0].strip()
        row_dict = row.to_dict()
        row_dict["row_id"] = idx
        row_dict["safety_cautions"] = caution_notes_for_context(ai, assessment)
        row_dict["_form_score"] = _score_row_for_target(row_dict, keywords)
        base_map.setdefault(base, []).append((score + row_dict["_form_score"], row_dict))

    if not base_map:
        return []

    best_base = max(base_map.keys(), key=lambda b: max(s for s, _ in base_map[b]))
    items = sorted(base_map[best_base], key=lambda x: x[0], reverse=True)

    results: list[DrugRow] = []
    for _, row_dict in items[:max_results]:
        results.append(DrugRow(
            row_id=row_dict["row_id"],
            name_ar=row_dict.get("name_ar", ""),
            name_en=row_dict.get("name_en", ""),
            active_ingredient=row_dict.get(dataset.INGREDIENT_COL, ""),
            safety_cautions=row_dict.get("safety_cautions", []),
        ))
    return results


def retrieve_for_targets(
    targets: list[TherapeuticTarget],
    assessment: ClinicalAssessment,
) -> list[IngredientMatch]:
    """For each therapeutic target, find best ingredient with dataset matches."""
    matches: list[IngredientMatch] = []
    used_bases: set[str] = set()

    for target in targets:
        candidates = ingredient_candidates_for_target(target.target)
        keywords = search_keywords_for_target(target.target)
        best_match: IngredientMatch | None = None

        for ingredient in candidates:
            base = re.split(r"[+/\s\-]", ingredient.lower())[0].strip()
            if base in used_bases:
                continue

            drugs = get_matching_drugs_for_ingredient(
                ingredient, assessment, target_keywords=keywords, max_results=2,
            )
            if drugs:
                best_match = IngredientMatch(
                    target=target.target,
                    ingredient=ingredient,
                    rationale=target.rationale,
                    priority=target.priority,
                    drugs=drugs,
                    safety_notes=drugs[0].safety_cautions if drugs else [],
                )
                used_bases.add(base)
                break

        if best_match:
            matches.append(best_match)

    return matches
