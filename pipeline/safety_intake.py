"""Mandatory safety intake before clinical assessment or drug recommendations."""

import re
from dataclasses import dataclass
from typing import Literal, Optional

from pipeline.context import (
    PatientContext,
    has_negation_response,
    normalize_text,
    parse_pregnancy_breastfeeding,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pipeline.session_state import ConversationState

FieldStatus = Literal["unknown", "none", "reported"]

_CHRONIC_BOT_KW = ["امراض مزمن", "أمراض مزمن", "مرض مزمن", "مزمنة", "مزمنه", "chronic"]
_ALLERGY_BOT_KW = ["حساسية", "حساسيه", "allergy", "allergies"]
_MEDS_BOT_KW = ["ادوية حالية", "أدوية حالية", "ادويه حاليا", "بتاخد ادوية", "بتاكل ادوية", "medications", "current meds"]

_CHRONIC_NONE_RE = [
    r"مفيش\s*(امراض|أمراض)\s*مزمن",
    r"(لا|مش|مفيش|معنديش|ماعنديش)\s*(عندي\s*)?(امراض|أمراض)\s*مزمن",
    r"لا\s*يوجد\s*(امراض|أمراض)\s*مزمن",
    r"(انا|أنا)\s*سليم",
    r"مش\s*عندي\s*(سكر|ضغط|كلى|قلب)\s*$",
    r"no\s*chronic",
]
_ALLERGY_NONE_RE = [
    r"مفيش\s*حساس",
    r"(لا|مش|مفيش|معنديش|ماعنديش)\s*(عندي\s*)?حساس",
    r"لا\s*يوجد\s*حساس",
    r"no\s*allerg",
]
_MEDS_NONE_RE = [
    r"مباخدش\s*ادو",
    r"مش\s*باخد\s*ادو",
    r"مش\s*بياخد\s*ادو",
    r"مش\s*بتاخد\s*ادو",
    r"مفيش\s*ادو",
    r"(لا|مش|مفيش)\s*(باخد|باخدش)\s*ادو",
    r"لا\s*يوجد\s*ادو",
    r"no\s*meds",
    r"not\s*taking\s*any\s*med",
]


@dataclass
class SafetyIntakeStatus:
    age_ok: bool
    sex_ok: bool
    chronic: FieldStatus
    allergies: FieldStatus
    medications: FieldStatus

    def complete(self) -> bool:
        return (
            self.age_ok
            and self.sex_ok
            and self.chronic != "unknown"
            and self.allergies != "unknown"
            and self.medications != "unknown"
        )

    def missing_prompts(self) -> list[str]:
        missing: list[str] = []
        if not self.age_ok:
            missing.append("السن")
        if not self.sex_ok:
            missing.append("الجنس (ذكر/أنثى)")
        if self.chronic == "unknown":
            missing.append("هل عندك أمراض مزمنة؟ (لو مفيش قولّي «مفيش»)")
        if self.allergies == "unknown":
            missing.append("هل عندك حساسية من أدوية؟ (لو مفيش قولّي «مفيش»)")
        if self.medications == "unknown":
            missing.append("هل بتاخد أدوية حالياً؟ (لو مفيش قولّي «مفيش»)")
        return missing


def _matches_any(text: str, patterns: list[str]) -> bool:
    norm = normalize_text(text)
    return any(re.search(p, norm, re.IGNORECASE) for p in patterns)


def _bot_asked_about(history: list, query: str, keywords: list[str]) -> bool:
    """True if the bot asked about this field in any recent assistant turn."""
    for msg in reversed(history or []):
        if msg.get("role") != "assistant":
            continue
        bot_text = normalize_text(msg.get("content", ""))
        if any(normalize_text(k) in bot_text for k in keywords):
            return True
    return False


def last_assistant_message(history: list) -> str:
    for msg in reversed(history or []):
        if msg.get("role") == "assistant":
            return msg.get("content", "")
    return ""


def _explicit_none_in_text(text: str, patterns: list[str]) -> bool:
    norm = normalize_text(text)
    if any(re.search(p, norm, re.IGNORECASE) for p in patterns):
        return True
    if has_negation_response(text) and len(norm) < 50:
        return True
    return False


def _field_status_reported_or_none(
    reported_items: list,
    full_text: str,
    none_patterns: list[str],
    history: list,
    query: str,
    bot_keywords: list[str],
) -> FieldStatus:
    if reported_items:
        return "reported"
    if _matches_any(full_text, none_patterns):
        return "none"
    if _bot_asked_about(history, query, bot_keywords) and _explicit_none_in_text(query, none_patterns):
        return "none"
    return "unknown"


def _status_from_session_or_text(
    session_status: FieldStatus,
    reported_items: list,
    full_text: str,
    none_patterns: list[str],
    history: list,
    query: str,
    bot_keywords: list[str],
) -> FieldStatus:
    if session_status in ("none", "reported"):
        return session_status
    return _field_status_reported_or_none(
        reported_items, full_text, none_patterns,
        history, query, bot_keywords,
    )


def assess_safety_intake(
    ctx: PatientContext,
    full_text: str,
    history: list,
    query: str,
    *,
    session: "ConversationState | None" = None,
) -> SafetyIntakeStatus:
    if session is not None:
        fields = {
            "age_ok": session.age is not None,
            "sex_ok": session.sex is not None and session.sex != "unknown",
            "chronic": session.chronic,
            "allergies": session.allergies,
            "medications": session.medications,
        }
    else:
        fields = {
            "age_ok": ctx.age is not None,
            "sex_ok": ctx.sex != "unknown",
            "chronic": "unknown",
            "allergies": "unknown",
            "medications": "unknown",
        }
    chronic = _status_from_session_or_text(
        fields["chronic"], ctx.chronic_conditions, full_text, _CHRONIC_NONE_RE,
        history, query, _CHRONIC_BOT_KW,
    )
    allergies = _status_from_session_or_text(
        fields["allergies"], ctx.allergies, full_text, _ALLERGY_NONE_RE,
        history, query, _ALLERGY_BOT_KW,
    )
    medications = _status_from_session_or_text(
        fields["medications"], ctx.current_meds, full_text, _MEDS_NONE_RE,
        history, query, _MEDS_BOT_KW,
    )
    return SafetyIntakeStatus(
        age_ok=fields["age_ok"] or ctx.age is not None,
        sex_ok=fields["sex_ok"] or ctx.sex != "unknown",
        chronic=chronic,
        allergies=allergies,
        medications=medications,
    )


def safety_intake_complete(
    ctx: PatientContext,
    full_text: str,
    history: list,
    query: str,
    *,
    session: "ConversationState | None" = None,
) -> bool:
    return assess_safety_intake(
        ctx, full_text, history, query, session=session,
    ).complete()


def format_safety_intake_response(status: SafetyIntakeStatus) -> str:
    missing = status.missing_prompts()
    if not missing:
        return ""
    lines = [
        "قبل ما أقترح أي دواء، محتاج أتأكد من:",
        "",
    ]
    for item in missing:
        lines.append(f"• {item}")
    lines.append("")
    lines.append("لو مفيش أمراض مزمنة / حساسية / أدوية حالية، قولّي بوضوح «مفيش».")
    return "\n".join(lines)


def pregnancy_check_missing(ctx: PatientContext, query: str) -> Optional[str]:
    """Additional check for females 18+ before medication (not part of core 5)."""
    if ctx.sex != "female" or ctx.age is None or ctx.age < 18:
        return None
    if ctx.pregnant is not None or ctx.breastfeeding is not None:
        return None
    preg, breast = parse_pregnancy_breastfeeding(query)
    if preg is not None:
        ctx.pregnant = preg
    if breast is not None:
        ctx.breastfeeding = breast
    if ctx.pregnant is None and ctx.breastfeeding is None:
        return "هل أنتِ حامل أو مرضعة؟ (لو لأ قولّي «لا»)"
    return None
