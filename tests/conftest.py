"""Shared fixtures for pipeline tests."""

import pandas as pd
import pytest

from pipeline import dataset


@pytest.fixture
def sample_drug_df() -> pd.DataFrame:
  return pd.DataFrame(
    [
      {
        "name_ar": "بانادول 500",
        "name_en": "Panadol 500",
        "ingredient_clean": "paracetamol",
        "active_ingredient": "paracetamol 500mg",
        "combined": "بانادول Panadol paracetamol",
      },
      {
        "name_ar": "بروفين اقراص",
        "name_en": "Brufen tablets",
        "ingredient_clean": "ibuprofen",
        "active_ingredient": "ibuprofen 400mg",
        "combined": "بروفين Brufen ibuprofen",
      },
      {
        "name_ar": "زيرتك 10",
        "name_en": "Zyrtec 10",
        "ingredient_clean": "cetirizine",
        "active_ingredient": "cetirizine 10mg",
        "combined": "زيرتك Zyrtec cetirizine",
      },
      {
        "name_ar": "باسط للمغص",
        "name_en": "Buscopan",
        "ingredient_clean": "hyoscine butylbromide",
        "active_ingredient": "hyoscine butylbromide 10mg",
        "combined": "باسط Buscopan hyoscine",
      },
      {
        "name_ar": "باراسيتامول حقن",
        "name_en": "Paracetamol injection",
        "ingredient_clean": "paracetamol",
        "active_ingredient": "paracetamol",
        "combined": "paracetamol injection",
      },
    ]
  )


@pytest.fixture
def loaded_dataset(sample_drug_df, monkeypatch):
  """Point dataset.df at the in-memory sample table."""
  monkeypatch.setattr(dataset, "df", sample_drug_df.reset_index(drop=True))
  monkeypatch.setattr(dataset, "INGREDIENT_COL", "ingredient_clean")
  monkeypatch.setattr(dataset, "index", None)
  return sample_drug_df
