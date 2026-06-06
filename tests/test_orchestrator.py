"""Unit tests for CLINICAL_PLAN parsing and orchestrator formatting."""

from unittest.mock import patch

import pytest

from pipeline.gemini import CLINICAL_PLAN_MARKER
from pipeline.orchestrator import (
  format_drug_section,
  parse_clinical_plan,
  rag,
  strip_clinical_plan,
)
from pipeline.retrieval import DrugMatch


def _sample_plan_text(convo: str = "تمام يا فندم") -> str:
  return f"""{convo}

{CLINICAL_PLAN_MARKER}
INGREDIENTS: paracetamol, cetirizine
EXCLUDED_INGREDIENTS: ibuprofen
ESCALATION_LEVEL: none
DIAGNOSIS_CONFIDENCE: medium
NON_DRUG_ADVICE: راحة | سوائل دافئة
"""


def test_parse_clinical_plan_extracts_fields():
  plan = parse_clinical_plan(_sample_plan_text())
  assert plan is not None
  assert plan["ingredients"] == ["paracetamol", "cetirizine"]
  assert plan["excluded"] == ["ibuprofen"]
  assert plan["escalation"] == "none"
  assert plan["confidence"] == "medium"
  assert plan["non_drug_advice"] == ["راحة", "سوائل دافئة"]


def test_parse_clinical_plan_returns_none_without_marker():
  assert parse_clinical_plan("أهلاً، احكيلي عن أعراضك") is None


def test_strip_clinical_plan_removes_block():
  visible = strip_clinical_plan(_sample_plan_text("محادثة"))
  assert CLINICAL_PLAN_MARKER not in visible
  assert visible == "محادثة"


def test_format_drug_section_includes_row_id():
  section = format_drug_section(
    [
      DrugMatch(
        row_id=42,
        name_ar="بانادول",
        name_en="Panadol",
        active_ingredient="paracetamol",
        matched_ingredient="paracetamol",
      )
    ]
  )
  assert "[#42]" in section
  assert "بانادول" in section


def test_format_drug_section_empty():
  assert "مفيش أدوية مطابقة" in format_drug_section([])


@patch("pipeline.orchestrator.chat")
def test_rag_intake_only_no_plan(mock_chat, loaded_dataset):
  mock_chat.return_value = ("أهلاً! ممكن تقولي سنك وجنسك؟", "ok")
  out = rag("عندي برد", [])
  assert out == "أهلاً! ممكن تقولي سنك وجنسك؟"
  assert "[#" not in out


@patch("pipeline.orchestrator.chat")
def test_rag_with_plan_returns_drugs_and_row_ids(mock_chat, loaded_dataset):
  mock_chat.return_value = (_sample_plan_text("يبدو عندك برد"), "ok")
  out = rag("21 سنة ذكر", [{"role": "user", "content": "عندي برد"}])
  assert "يبدو عندك برد" in out
  assert CLINICAL_PLAN_MARKER not in out
  assert "─── الأدوية من القاعدة ───" in out
  assert "[#0]" in out  # paracetamol -> Panadol row
  assert "بروفين" not in out  # ibuprofen excluded


@patch("pipeline.orchestrator.chat")
def test_rag_rate_limit_message(mock_chat, loaded_dataset):
  mock_chat.return_value = (None, "rate_limit")
  out = rag("test", [])
  assert out == "معلش، الخدمة مشغولة دلوقتي. حاول تاني بعد شوية."


@patch("pipeline.orchestrator.chat")
def test_rag_empty_response_fallback(mock_chat, loaded_dataset):
  mock_chat.return_value = ("   ", "ok")
  out = rag("test", [])
  assert out == "معلش، الخدمة مشغولة دلوقتي. حاول تاني بعد شوية."
