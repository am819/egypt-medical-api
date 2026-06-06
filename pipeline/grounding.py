"""Strict dataset grounding — verify every drug row against the CSV."""

from pipeline import dataset
from pipeline.models import DrugRow, IngredientMatch


def verify_drug_row(drug: DrugRow) -> DrugRow | None:
    """
    Resolve and verify a drug row against the dataset.
    Returns None if row_id is invalid or row cannot be resolved.
    """
    if dataset.df.empty:
        return None

    row_id = drug.row_id
    if row_id is None or row_id < 0 or row_id >= len(dataset.df):
        print(f"⚠️ Invalid row_id {row_id} — drug excluded")
        return None

    row = dataset.df.iloc[row_id]
    name_ar = str(row.get("name_ar", "")).strip()
    name_en = str(row.get("name_en", "")).strip()
    active = str(row.get(dataset.INGREDIENT_COL, "")).strip()

    if not name_ar and not name_en:
        print(f"⚠️ Empty drug names at row {row_id} — excluded")
        return None

    return DrugRow(
        row_id=row_id,
        name_ar=name_ar,
        name_en=name_en,
        active_ingredient=active,
        safety_cautions=drug.safety_cautions,
    )


def verify_ingredient_matches(matches: list[IngredientMatch]) -> list[IngredientMatch]:
    """Filter matches to only dataset-verified drugs."""
    verified: list[IngredientMatch] = []
    for match in matches:
        valid_drugs: list[DrugRow] = []
        for drug in match.drugs:
            v = verify_drug_row(drug)
            if v:
                valid_drugs.append(v)
        if valid_drugs:
            verified.append(IngredientMatch(
                target=match.target,
                ingredient=match.ingredient,
                rationale=match.rationale,
                priority=match.priority,
                drugs=valid_drugs,
                safety_notes=match.safety_notes,
            ))
    return verified
