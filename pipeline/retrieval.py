"""Drug lookup by active ingredient from the local CSV database."""

import re
from dataclasses import dataclass

from rapidfuzz import process

from pipeline import dataset

_INJECTION_PATTERNS = (
    r"\b(inj|injection|ampoule|amp|vial|iv\b|intravenous|حقن|امبول)",
)
_BABY_PATTERNS = (r"\bbaby\b", r"\binfant\b", r"\bpediatric\b", r"رضيع", r"أطفال")


@dataclass
class DrugMatch:
    row_id: int
    name_ar: str
    name_en: str
    active_ingredient: str
    matched_ingredient: str


def _normalize_ingredient(name: str) -> str:
    return re.split(r"[+/\s\-]", name.lower().strip())[0]


def _is_excluded_ingredient(ingredient: str, excluded: set[str]) -> bool:
    base = _normalize_ingredient(ingredient)
    for ex in excluded:
        ex_base = _normalize_ingredient(ex)
        if base == ex_base or base in ex_base or ex_base in base:
            return True
    return False


def _has_excluded_form(name_ar: str, name_en: str) -> bool:
    combined = f"{name_ar} {name_en}".lower()
    return any(re.search(p, combined, re.I) for p in _INJECTION_PATTERNS)


def lookup_drugs_by_ingredients(
    ingredients: list[str],
    excluded: list[str] | None = None,
    max_per_ingredient: int = 2,
) -> list[DrugMatch]:
    """Find drugs in CSV matching LLM-provided active ingredients."""
    if dataset.df.empty or not ingredients:
        return []

    excluded_set = {_normalize_ingredient(x) for x in (excluded or []) if x.strip()}
    skip_terms = {"useful for cough", "useful for pain", "علاج", "دواء", "unknown", "none", "n/a"}
    results: list[DrugMatch] = []
    used_bases: set[str] = set()

    col = dataset.INGREDIENT_COL
    all_ingredients = dataset.df[col].tolist()

    for ingredient in ingredients:
        ing = ingredient.strip()
        if not ing or ing.lower() in skip_terms or len(ing) < 3:
            continue
        if _is_excluded_ingredient(ing, excluded_set):
            continue

        base = _normalize_ingredient(ing)
        if base in used_bases:
            continue

        fuzzy = process.extract(ing, all_ingredients, limit=30)
        ingredient_hits: list[tuple[int, int]] = []

        for _text, score, idx in fuzzy:
            if score < 85:
                continue
            row = dataset.df.iloc[idx]
            name_ar = str(row.get("name_ar", ""))
            name_en = str(row.get("name_en", ""))
            if _has_excluded_form(name_ar, name_en):
                continue
            ai = str(row.get(col, "")).strip()
            row_base = _normalize_ingredient(ai)
            if row_base != base and base not in row_base and row_base not in base:
                continue
            ingredient_hits.append((score, idx))

        ingredient_hits.sort(key=lambda x: x[0], reverse=True)
        seen_rows: set[int] = set()
        count = 0

        for _score, idx in ingredient_hits:
            if idx in seen_rows:
                continue
            seen_rows.add(idx)
            row = dataset.df.iloc[idx]
            results.append(
                DrugMatch(
                    row_id=int(idx),
                    name_ar=str(row.get("name_ar", "")),
                    name_en=str(row.get("name_en", "")),
                    active_ingredient=str(row.get(col, "")),
                    matched_ingredient=ing,
                )
            )
            count += 1
            if count >= max_per_ingredient:
                break

        if count > 0:
            used_bases.add(base)

    return results
