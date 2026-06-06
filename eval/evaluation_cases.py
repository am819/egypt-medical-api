"""Offline evaluation scenarios (mocked LLM responses)."""

from pipeline.gemini import CLINICAL_PLAN_MARKER

EVALUATION_CASES = [
  {
    "id": "intake_no_plan",
    "description": "First message — conversational intake only",
    "user_message": "عندي برد ومغص",
    "history": [],
    "mock_llm": "أهلاً بك! عشان أساعدك بأمان، محتاج أعرف سنك وجنسك وأي أمراض مزمنة أو حساسية.",
    "must_contain": ["أهلاً", "سنك"],
    "must_not_contain": [CLINICAL_PLAN_MARKER, "[#"],
    "min_drug_matches": 0,
  },
  {
    "id": "cold_plan_with_drugs",
    "description": "Complete case — CLINICAL_PLAN + drug rows from CSV",
    "user_message": "21 سنة ذكر مفيش سكر مفيش حساسية",
    "history": [
      {"role": "user", "content": "عندي برد ومغص"},
      {"role": "assistant", "content": "محتاج أعرف سنك"},
    ],
    "mock_llm": f"""تمام يا فندم، يبدو عندك برد مع مغص بسيط.

{CLINICAL_PLAN_MARKER}
INGREDIENTS: paracetamol, hyoscine butylbromide, cetirizine
EXCLUDED_INGREDIENTS: ibuprofen
ESCALATION_LEVEL: none
DIAGNOSIS_CONFIDENCE: medium
NON_DRUG_ADVICE: راحة | سوائل دافئة
""",
    "must_contain": ["تمام", "─── الأدوية من القاعدة ───", "[#", "paracetamol"],
    "must_not_contain": [CLINICAL_PLAN_MARKER, "INGREDIENTS:"],
    "min_drug_matches": 2,
    "expected_row_ids": [0, 2, 3],
  },
  {
    "id": "rate_limit",
    "description": "Gemini unavailable — busy Arabic message",
    "user_message": "test",
    "history": [],
    "mock_llm": None,
    "mock_status": "rate_limit",
    "must_contain": ["معلش، الخدمة مشغولة"],
    "must_not_contain": ["[#"],
    "min_drug_matches": 0,
  },
  {
    "id": "plan_no_csv_match",
    "description": "CLINICAL_PLAN with unknown ingredient",
    "user_message": "علاج",
    "history": [],
    "mock_llm": f"""حاضر.

{CLINICAL_PLAN_MARKER}
INGREDIENTS: totallyunknowniumxyz
EXCLUDED_INGREDIENTS: none
ESCALATION_LEVEL: none
DIAGNOSIS_CONFIDENCE: low
NON_DRUG_ADVICE: راحة
""",
    "must_contain": ["مفيش أدوية مطابقة"],
    "must_not_contain": [CLINICAL_PLAN_MARKER],
    "min_drug_matches": 0,
  },
]
