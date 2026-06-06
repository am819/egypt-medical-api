"""Chat orchestrator — LLM conversation + drug lookup from CLINICAL_PLAN."""

import re

from pipeline.gemini import CLINICAL_PLAN_MARKER, chat
from pipeline.retrieval import DrugMatch, lookup_drugs_by_ingredients

_BUSY_MSG = "معلش، الخدمة مشغولة دلوقتي. حاول تاني بعد شوية."


def _failure_message(status: str) -> str:
    if status == "rate_limit":
        return _BUSY_MSG
    if status == "no_api_key":
        return "عذراً، مفتاح Gemini غير مُعد — راجع GEMINI_API_KEY على Railway."
    return "عذراً، حدث خطأ في الاتصال."


def _parse_field(block: str, field: str) -> str:
    match = re.search(rf"^{field}:\s*(.+)$", block, re.MULTILINE | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _split_list(value: str) -> list[str]:
    if not value or value.lower() in ("none", "n/a", "-"):
        return []
    parts = re.split(r"[,|]", value)
    return [p.strip() for p in parts if p.strip() and p.strip().lower() not in ("none", "n/a")]


def parse_clinical_plan(text: str) -> dict | None:
    """Extract CLINICAL_PLAN fields from LLM response."""
    if CLINICAL_PLAN_MARKER not in text:
        return None
    idx = text.index(CLINICAL_PLAN_MARKER)
    block = text[idx:]
    return {
        "ingredients": _split_list(_parse_field(block, "INGREDIENTS")),
        "excluded": _split_list(_parse_field(block, "EXCLUDED_INGREDIENTS")),
        "escalation": _parse_field(block, "ESCALATION_LEVEL"),
        "confidence": _parse_field(block, "DIAGNOSIS_CONFIDENCE"),
        "non_drug_advice": _split_list(_parse_field(block, "NON_DRUG_ADVICE")),
    }


def strip_clinical_plan(text: str) -> str:
    """Remove the CLINICAL_PLAN block from user-visible text."""
    if CLINICAL_PLAN_MARKER not in text:
        return text.strip()
    return text[: text.index(CLINICAL_PLAN_MARKER)].strip()


def format_drug_section(matches: list[DrugMatch]) -> str:
    if not matches:
        return "─── الأدوية من القاعدة ───\n⚠️ مفيش أدوية مطابقة في القاعدة للمواد الفعالة المقترحة."
    lines = ["─── الأدوية من القاعدة ───"]
    for m in matches:
        display = m.name_ar or m.name_en or "—"
        en = f" ({m.name_en})" if m.name_en and m.name_en != m.name_ar else ""
        lines.append(f"💊 {display}{en} [#{m.row_id}]")
        if m.active_ingredient:
            lines.append(f"   • {m.active_ingredient}")
    return "\n".join(lines)


def _rag_inner(query: str, history: list) -> str:
    raw, status = chat(query, history)
    if not raw:
        return _failure_message(status)

    plan = parse_clinical_plan(raw)
    visible = strip_clinical_plan(raw)

    if not plan or not plan["ingredients"]:
        return visible or raw

    drugs = lookup_drugs_by_ingredients(
        plan["ingredients"],
        excluded=plan.get("excluded"),
        max_per_ingredient=2,
    )
    drug_section = format_drug_section(drugs)

    parts = [visible] if visible else []
    if plan.get("non_drug_advice"):
        parts.append("📋 نصائح:\n" + "\n".join(f"   • {a}" for a in plan["non_drug_advice"]))
    parts.append(drug_section)
    return "\n\n".join(parts)


def rag(query: str, history: list) -> str:
    result = _rag_inner(query, history)
    if not result or not result.strip():
        return _BUSY_MSG
    return result
