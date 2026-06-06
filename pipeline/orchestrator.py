"""Main RAG orchestrator — wires all 7 pipeline phases."""

from pipeline.context import extract_context, has_real_emergency, intake_gate
from pipeline.formatter import format_final_response, format_temporary_response, format_urgent_response
from pipeline.forms import apply_form_filters
from pipeline.gemini import call_intake_conversational, call_structured_clinical
from pipeline.retrieval import retrieve_for_targets
from pipeline.safety import apply_safety_filters
from pipeline.targets import validate_targets


URGENT_TEMP_KEYWORDS = [
    "مؤقت", "علاج مؤقت", "دواء مؤقت", "اقترح",
    "مش قادر اروح", "حاجة تخفف", "بس عايز حاجة",
]


def _last_bot_message(history: list) -> str:
    for msg in reversed(history or []):
        if msg.get("role") == "assistant":
            return msg.get("content", "")
    return ""


def _asks_for_temporary(query: str) -> bool:
    return any(kw in query.lower() for kw in URGENT_TEMP_KEYWORDS)


def rag(query: str, history: list) -> str:
    """Core RAG inference — 7-phase clinical pipeline."""
    last_bot = _last_bot_message(history)
    temporary_override = "روح الطوارئ فوراً" in last_bot and _asks_for_temporary(query)

    ctx = extract_context(query, history)

    if temporary_override:
        clinical, status = call_structured_clinical(
            query, history, ctx, temporary_override=True,
        )
        if status == "rate_limit":
            return "معلش، النظام مشغول دلوقتي — استنى شوية."
        if not clinical:
            return "عذراً، حدث خطأ في الاتصال."
        if clinical.status == "needs_info" or not clinical.therapeutic_targets:
            return "عذراً، لم أستطع اقتراح علاج مؤقت مناسب. يرجى التوجه للطوارئ فوراً."

        targets = validate_targets(clinical.therapeutic_targets, clinical.assessment)
        matches = retrieve_for_targets(targets, clinical.assessment)
        matches, safety_notes = apply_safety_filters(matches, clinical.assessment)
        matches = apply_form_filters(matches, clinical.assessment)
        return format_temporary_response(clinical, matches, safety_notes)

    gate = intake_gate(ctx, history, query)
    if gate:
        return gate

    clinical, status = call_structured_clinical(query, history, ctx)
    if status == "rate_limit":
        return "معلش، النظام مشغول دلوقتي — استنى شوية."
    if not clinical:
        return "عذراً، حدث خطأ في الاتصال."

    if clinical.status == "needs_info":
        if clinical.patient_message_ar:
            return clinical.patient_message_ar
        conv, conv_status = call_intake_conversational(query, history, ctx)
        if conv:
            return conv
        missing = clinical.missing_fields
        if missing:
            return "عشان أساعدك، محتاج أعرف: " + "، ".join(missing) + "."
        return "محتاج معلومات أكتر عن حالتك عشان أقدر أساعدك."

    if (
        clinical.status == "urgent"
        or clinical.differential.requires_urgent_care
        or has_real_emergency(ctx)
    ):
        return format_urgent_response(clinical)

    targets = validate_targets(clinical.therapeutic_targets, clinical.assessment)
    if not targets:
        intro = clinical.patient_message_ar or "محتاج أعرف أكتر عن الأعراض."
        return intro + "\n\n⚠️ لم أتمكن من تحديد أهداف علاجية آمنة مناسبة لحالتك."

    matches = retrieve_for_targets(targets, clinical.assessment)
    matches, safety_notes = apply_safety_filters(matches, clinical.assessment)
    matches = apply_form_filters(matches, clinical.assessment)

    return format_final_response(clinical, matches, safety_notes)
