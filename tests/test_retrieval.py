"""Unit tests for ingredient-based drug lookup."""

import pytest

from pipeline.retrieval import lookup_drugs_by_ingredients


def test_lookup_finds_paracetamol(loaded_dataset):
  matches = lookup_drugs_by_ingredients(["paracetamol"])
  assert len(matches) >= 1
  assert matches[0].row_id == 0
  assert "paracetamol" in matches[0].active_ingredient.lower()


def test_lookup_skips_injection_form(loaded_dataset):
  matches = lookup_drugs_by_ingredients(["paracetamol"], max_per_ingredient=5)
  names = " ".join(m.name_en.lower() for m in matches)
  assert "injection" not in names


def test_lookup_respects_excluded(loaded_dataset):
  matches = lookup_drugs_by_ingredients(
    ["paracetamol", "ibuprofen"],
    excluded=["ibuprofen"],
  )
  ingredients = {m.matched_ingredient.lower() for m in matches}
  assert "ibuprofen" not in ingredients


def test_lookup_multiple_ingredients(loaded_dataset):
  matches = lookup_drugs_by_ingredients(["paracetamol", "cetirizine"])
  row_ids = {m.row_id for m in matches}
  assert 0 in row_ids
  assert 2 in row_ids


def test_lookup_empty_when_no_dataframe(monkeypatch):
  from pipeline import dataset

  monkeypatch.setattr(dataset, "df", __import__("pandas").DataFrame())
  assert lookup_drugs_by_ingredients(["paracetamol"]) == []


def test_lookup_skips_short_ingredient_names(loaded_dataset):
  assert lookup_drugs_by_ingredients(["ab"]) == []
