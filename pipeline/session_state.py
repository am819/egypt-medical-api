"""Derive persistent conversation state from full chat history."""

import re
from dataclasses import dataclass, field
from typing import Optional

from pipeline.context import (
    PatientContext,
    extract_age,
    extract_sex,
    has_negation_response,
    normalize_text,
    parse_pregnancy_breastfeeding,
)
from typing import Literal

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
    r"مفيش\s*ادو",
    r"(لا|مش|مفيش)\s*(باخد|باخدش)\s*ادو",
    r"لا\s*يوجد\s*ادو",
    r"no\s*meds",
    r"not\s*taking\s*any\s*med",
]


def _matches_any(text: str, patterns: list[str]) -> bool:
    norm = normalize_text(text)
    return any(re.search(p, norm, re.IGNORECASE) for p in patterns)

_AGE_BOT_KW = ["السن", "عمرك", "عمرك كام", "كم عمر", "age"]
_SEX_BOT_KW = ["الجنس", "ذكر", "انثى", "أنثى", "male", "female", "راجل ولا"]


@dataclass
class ConversationState:
    """Fields confirmed across turns — survives regex re-extraction gaps."""

    age: Optional[int] = None
    sex: Optional[str] = None
    chronic: FieldStatus = "unknown"
    allergies: FieldStatus = "unknown"
    medications: FieldStatus = "unknown"
    chronic_conditions: list[str] = field(default_factory=list)
    allergy_items: list[str] = field(default_factory=list)
    medication_items: list[str] = field(default_factory=list)
    asked_followup_keys: set[str] = field(default_factory=set)
    red_flag_screening_done: bool = False


def _bot_mentioned(text: str, keywords: list[str]) -> bool:
    norm = normalize_text(text)
    return any(normalize_text(k) in norm for k in keywords)


def _scan_history_pairs(history: list) -> list[tuple[str, str]]:
    """Return (assistant_msg, following_user_msg) pairs."""
    pairs: list[tuple[str, str]] = []
    pending_bot = ""
    for msg in history or []:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "assistant":
            pending_bot = content
        elif role == "user" and pending_bot:
            pairs.append((pending_bot, content))
            pending_bot = ""
    return pairs


def _status_from_reply(
    bot_text: str,
    user_reply: str,
    bot_keywords: list[str],
    none_patterns: list[str],
    reported_items: list[str],
) -> Optional[FieldStatus]:
    if not _bot_mentioned(bot_text, bot_keywords):
        return None
    if reported_items:
        return "reported"
    if _matches_any(user_reply, none_patterns):
        return "none"
    norm = normalize_text(user_reply)
    if has_negation_response(user_reply) and len(norm) < 60:
        return "none"
    if user_reply.strip() and not has_negation_response(user_reply):
        return "reported"
    return None


def derive_conversation_state(history: list, query: str, ctx: PatientContext) -> ConversationState:
    """Rebuild confirmed intake fields and follow-up progress from the transcript."""
    state = ConversationState(
        age=ctx.age,
        sex=ctx.sex if ctx.sex != "unknown" else None,
        chronic_conditions=list(ctx.chronic_conditions),
        allergy_items=list(ctx.allergies),
        medication_items=list(ctx.current_meds),
    )

    full_user_text = "\n".join(
        msg.get("content", "") for msg in (history or []) if msg.get("role") == "user"
    )
    full_user_text = (full_user_text + "\n" + query).strip()

    if state.chronic_conditions:
        state.chronic = "reported"
    elif _matches_any(full_user_text, _CHRONIC_NONE_RE):
        state.chronic = "none"
        state.chronic_conditions = []

    if state.allergy_items:
        state.allergies = "reported"
    elif _matches_any(full_user_text, _ALLERGY_NONE_RE):
        state.allergies = "none"
        state.allergy_items = []

    if state.medication_items:
        state.medications = "reported"
    elif _matches_any(full_user_text, _MEDS_NONE_RE):
        state.medications = "none"
        state.medication_items = []

    for bot_text, user_reply in _scan_history_pairs(history):
        age = extract_age(user_reply)
        if age and _bot_mentioned(bot_text, _AGE_BOT_KW):
            state.age = age
        sex = extract_sex(user_reply)
        if sex != "unknown" and _bot_mentioned(bot_text, _SEX_BOT_KW):
            state.sex = sex

        chronic_status = _status_from_reply(
            bot_text, user_reply, _CHRONIC_BOT_KW, _CHRONIC_NONE_RE, state.chronic_conditions,
        )
        if chronic_status:
            state.chronic = chronic_status
            if chronic_status == "none":
                state.chronic_conditions = []

        allergy_status = _status_from_reply(
            bot_text, user_reply, _ALLERGY_BOT_KW, _ALLERGY_NONE_RE, state.allergy_items,
        )
        if allergy_status:
            state.allergies = allergy_status
            if allergy_status == "none":
                state.allergy_items = []

        meds_status = _status_from_reply(
            bot_text, user_reply, _MEDS_BOT_KW, _MEDS_NONE_RE, state.medication_items,
        )
        if meds_status:
            state.medications = meds_status
            if meds_status == "none":
                state.medication_items = []

    if state.age is None:
        state.age = extract_age(query) or extract_age(full_user_text)
    if state.sex is None:
        sex = extract_sex(query) or extract_sex(full_user_text)
        if sex != "unknown":
            state.sex = sex

    for msg in history or []:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        if "الطفح" in content or "الحساسية ظهرت" in content:
            state.asked_followup_keys.add("rash_location")
        if "شكل الطفح" in content:
            state.asked_followup_keys.add("rash_appearance")
        if "هرش" in content and "تنشر" in content:
            state.asked_followup_keys.add("rash_spread")
        if "الكحة جافة" in content or "ببلغم" in content:
            state.asked_followup_keys.add("cough_type")
        if "الكحة بدأت" in content:
            state.asked_followup_keys.add("cough_duration")
        if "لونه إيه" in content and "بلغم" in content:
            state.asked_followup_keys.add("sputum_color")
        if "الألم فين" in content and "البطن" in content:
            state.asked_followup_keys.add("abdominal_location")
        if "شدة الألم" in content:
            state.asked_followup_keys.add("abdominal_severity")
        if "أعراض تانية" in content:
            state.asked_followup_keys.add("abdominal_associated")
        if "الأعراض بدأت" in content:
            state.asked_followup_keys.add("symptom_duration")
        if "علامات خطر" in content or "ضيق نفس" in content or "ألم صدر" in content:
            state.red_flag_screening_done = True

    return state


def apply_conversation_state(ctx: PatientContext, state: ConversationState) -> PatientContext:
    """Merge persisted state into extracted context."""
    if state.age is not None:
        ctx.age = state.age
    if state.sex:
        ctx.sex = state.sex
    if state.chronic == "none":
        ctx.chronic_conditions = []
    elif state.chronic_conditions:
        ctx.chronic_conditions = list(state.chronic_conditions)
    if state.allergies == "none":
        ctx.allergies = []
    elif state.allergy_items:
        ctx.allergies = list(state.allergy_items)
    if state.medications == "none":
        ctx.current_meds = []
    elif state.medication_items:
        ctx.current_meds = list(state.medication_items)
    return ctx


def intake_field_status(state: ConversationState) -> dict[str, FieldStatus | bool]:
    """Expose intake completion for safety_intake without re-deriving."""
    return {
        "age_ok": state.age is not None,
        "sex_ok": state.sex is not None and state.sex != "unknown",
        "chronic": state.chronic,
        "allergies": state.allergies,
        "medications": state.medications,
    }
