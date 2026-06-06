"""Main RAG orchestrator — wires all pipeline phases."""

from pipeline.assessment import enrich_clinical_result
from pipeline.clinical_gates import can_recommend_medications
from pipeline.context import (
    conversation_to_text,
    extract_context,
    has_real_emergency,
    intake_gate,
)
from pipeline.formatter import (
    format_assessment_only_response,
    format_final_response,
    format_temporary_response,
    format_urgent_response,
)
from pipeline.followup import format_followup_response, get_followup_questions
from pipeline.forms import apply_form_filters
from pipeline.gemini import call_structured_clinical
from pipeline.grounding import verify_ingredient_matches
from pipeline.retrieval import retrieve_for_targets
from pipeline.red_flags import (
    format_red_flag_screening_response,
    get_red_flag_screening_questions,
)
from pipeline.safety import apply_safety_filters
from pipeline.safety_intake import (
    assess_safety_intake,
    format_safety_intake_response,
    last_assistant_message,
    pregnancy_check_missing,
    safety_intake_complete,
)
from pipeline.session_state import apply_conversation_state, derive_conversation_state
from pipeline.targets import validate_targets


URGENT_TEMP_KEYWORDS = [
    "مؤقت", "علاج مؤقت", "دواء مؤقت", "اقترح",
    "مش قادر اروح", "حاجة تخفف", "بس عايز حاجة",
]


def _asks_for_temporary(query: str) -> bool:
    return any(kw in query.lower() for kw in URGENT_TEMP_KEYWORDS)


def _full_conversation_text(query: str, history: list) -> str:
    return (conversation_to_text(history) + "\n" + query).strip()


def _pregnancy_block(ctx, query: str) -> str | None:
    """Pre-medication check for females 18+."""
    missing = pregnancy_check_missing(ctx, query)
    if missing:
        return f"قبل ما أقترح أدوية، محتاج أعرف: {missing}"
    return None


def _finalize_matches(clinical, targets):
    matches = retrieve_for_targets(targets, clinical.assessment)
    matches, safety_notes = apply_safety_filters(matches, clinical.assessment)
    matches = apply_form_filters(matches, clinical.assessment)
    matches = verify_ingredient_matches(matches)
    return matches, safety_notes


def _block_medications_without_safety(ctx, full_text, history, query, session) -> str | None:
    if not safety_intake_complete(ctx, full_text, history, query, session=session):
        status = assess_safety_intake(ctx, full_text, history, query, session=session)
        return format_safety_intake_response(status)
    return None


def _symptom_clarification_response(ctx, full_text, history, session) -> str | None:
    """Ask targeted clinical questions before letting the LLM diagnose."""
    assessment = ctx.to_assessment() if hasattr(ctx, "to_assessment") else None
    if assessment is None:
        from pipeline.context import patient_context_to_assessment

        assessment = patient_context_to_assessment(ctx)

    red_flag_questions = get_red_flag_screening_questions(ctx, full_text, session=session)
    if red_flag_questions:
        return format_red_flag_screening_response(red_flag_questions)

    questions = get_followup_questions(
        full_text,
        assessment,
        history=history,
        session=session,
    )
    if questions:
        return format_followup_response(questions)
    return None


def _handle_needs_info(clinical, full_text, history, query, ctx, session) -> str:
    """One-shot response for critical safety gaps — no conversational fallback."""
    if clinical.patient_message_ar:
        return clinical.patient_message_ar
    missing = list(clinical.missing_fields or [])
    if not missing:
        status = assess_safety_intake(ctx, full_text, history, query, session=session)
        missing = status.missing_prompts()
    if missing:
        return "عشان أكمل التقييم بأمان، محتاج أعرف: " + "، ".join(missing[:3]) + "."
    return format_assessment_only_response(clinical, full_text)


def rag(query: str, history: list) -> str:
    """Core RAG inference — single-pass structured clinical pipeline."""
    full_text = _full_conversation_text(query, history)
    last_bot = last_assistant_message(history)
    temporary_override = "روح الطوارئ فوراً" in last_bot and _asks_for_temporary(query)

    ctx = extract_context(query, history)
    session = derive_conversation_state(history, query, ctx)
    ctx = apply_conversation_state(ctx, session)

    gate = intake_gate(ctx, history, query)
    if gate:
        return gate

    safety_block = _block_medications_without_safety(
        ctx, full_text, history, query, session,
    )
    if safety_block:
        return safety_block

    clarification = _symptom_clarification_response(ctx, full_text, history, session)
    if clarification:
        return clarification

    if temporary_override:
        preg = _pregnancy_block(ctx, query)
        if preg:
            return preg

        clinical, status = call_structured_clinical(
            query, history, ctx, temporary_override=True,
        )
        if status == "rate_limit":
            return "معلش، النظام مشغول دلوقتي — استنى شوية."
        if not clinical:
            return "عذراً، حدث خطأ في الاتصال."
        clinical = enrich_clinical_result(clinical, full_text)
        if clinical.status == "needs_info" or not clinical.therapeutic_targets:
            return "عذراً، لم أستطع اقتراح علاج مؤقت مناسب. يرجى التوجه للطوارئ فوراً."

        targets = validate_targets(clinical.therapeutic_targets, clinical.assessment)
        matches, safety_notes = _finalize_matches(clinical, targets)
        return format_temporary_response(clinical, matches, safety_notes, full_text)

    clinical, status = call_structured_clinical(query, history, ctx)
    if status == "rate_limit":
        return "معلش، النظام مشغول دلوقتي — استنى شوية."
    if not clinical:
        return "عذراً، حدث خطأ في الاتصال."

    clinical = enrich_clinical_result(clinical, full_text)

    if clinical.status == "needs_info":
        return _handle_needs_info(clinical, full_text, history, query, ctx, session)

    if (
        clinical.status == "urgent"
        or clinical.differential.requires_urgent_care
        or has_real_emergency(ctx)
    ):
        return format_urgent_response(clinical, full_text)

    targets = validate_targets(clinical.therapeutic_targets, clinical.assessment)
    can_meds = bool(targets) and can_recommend_medications(full_text, clinical.assessment)

    if not can_meds:
        if clinical.differential.possible_conditions:
            return format_final_response(clinical, [], [], full_text)
        return format_assessment_only_response(clinical, full_text)

    preg = _pregnancy_block(ctx, query)
    if preg:
        return preg

    matches, safety_notes = _finalize_matches(clinical, targets)
    return format_final_response(clinical, matches, safety_notes, full_text)
