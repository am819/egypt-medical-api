"""Phase 7 — structured Arabic medical response formatter."""

from pipeline.models import ClinicalPipelineResult, IngredientMatch, SafetyNote
from pipeline.red_flags import build_red_flag_section

CONFIDENCE_AR = {"high": "عالية", "medium": "متوسطة", "low": "منخفضة"}


def _format_assessment_summary(clinical: ClinicalPipelineResult) -> str:
    a = clinical.assessment
    lines = [
        f"- العمر: {a.age if a.age is not None else 'غير مذكور'}",
        f"- الجنس: {a.sex}",
        f"- حمل/رضاعة: {'نعم' if (a.pregnant or a.breastfeeding) else 'لا' if a.pregnant is False else 'غير مذكور'}",
        f"- الأعراض الرئيسية: {', '.join(a.main_symptoms) or 'غير مذكورة'}",
        f"- أعراض مرافقة: {', '.join(a.associated_symptoms) or 'لا يوجد'}",
        f"- مدة الأعراض: {a.symptom_duration or 'غير مذكورة'}",
        f"- شدة: {', '.join(a.severity_indicators) or 'غير محددة'}",
        f"- أمراض مزمنة: {', '.join(a.chronic_diseases) or 'لا يوجد'}",
        f"- حساسية: {', '.join(a.drug_allergies) or 'لا يوجد'}",
        f"- أدوية حالية: {', '.join(a.current_medications) or 'لا يوجد'}",
    ]
    return "\n".join(lines)


def _format_differential(clinical: ClinicalPipelineResult) -> str:
    d = clinical.differential
    if not d.possible_conditions:
        return "لم يتم تحديد احتمالات محددة — الأعراض تحتاج متابعة."

    lines = []
    for i, cond in enumerate(d.possible_conditions):
        conf = d.confidence_levels[i] if i < len(d.confidence_levels) else "medium"
        lines.append(
            f"{i + 1}. الأعراض دي **ممكن تكون متوافقة مع** {cond} "
            f"(ثقة: {CONFIDENCE_AR.get(conf, conf)})"
        )
    return "\n".join(lines)


def _format_ingredients(matches: list[IngredientMatch]) -> str:
    if not matches:
        return "لم يتم تحديد مواد فعالة مناسبة بعد الفلاتر."
    lines = []
    for m in matches:
        reason = f" — {m.rationale}" if m.rationale else ""
        lines.append(f"- **{m.ingredient.title()}** (هدف: {m.target}){reason}")
    return "\n".join(lines)


def _format_drugs(matches: list[IngredientMatch]) -> str:
    if not matches:
        return "⚠️ مفيش أدوية متوفرة في القاعدة مطابقة بعد فلاتر الأمان والشكل الدوائي."
    blocks = []
    for m in matches:
        blocks.append(f"🔹 **المادة الفعالة: {m.ingredient.title()}** (هدف: {m.target})")
        for d in m.drugs:
            display_name = d.name_ar or d.name_en or "—"
            en_name = f" | {d.name_en}" if d.name_en and d.name_en != d.name_ar else ""
            blocks.append(f"   💊 **{display_name}**{en_name}")
            if d.active_ingredient:
                blocks.append(f"      • Active Ingredient: {d.active_ingredient}")
            blocks.append(f"      • Row ID: {d.row_id}")
        blocks.append("")
    return "\n".join(blocks).strip()


def _format_safety_notes(
    matches: list[IngredientMatch],
    safety_notes: list[SafetyNote],
    clinical: ClinicalPipelineResult,
) -> str:
    lines = []
    for note in safety_notes:
        prefix = "⚠️" if note.severity in ("caution", "warning") else "ℹ️"
        lines.append(f"{prefix} {note.message}")
    for m in matches:
        for d in m.drugs:
            for c in d.safety_cautions:
                line = f"⚠️ {d.name_ar}: {c}"
                if line not in lines:
                    lines.append(line)
    if clinical.non_drug_advice:
        lines.append("📋 نصائح غير دوائية:")
        for adv in clinical.non_drug_advice:
            lines.append(f"   • {adv}")
    return "\n".join(lines) if lines else "لا توجد ملاحظات أمان إضافية."


def format_urgent_response(clinical: ClinicalPipelineResult, full_text: str = "") -> str:
    intro = clinical.patient_message_ar or "فيه علامات خطر محتملة."
    red_section = build_red_flag_section(
        clinical.assessment,
        full_text,
        clinical.differential.red_flag_assessment,
    )
    parts = [
        intro,
        "",
        "🚨 **روح الطوارئ فوراً.**",
        "",
        "## ملخص الحالة",
        _format_assessment_summary(clinical),
        "",
        "## تقييم علامات الخطر",
        red_section,
    ]
    return "\n".join(parts)


def format_temporary_response(
    clinical: ClinicalPipelineResult,
    matches: list[IngredientMatch],
    safety_notes: list[SafetyNote],
    full_text: str = "",
) -> str:
    body = format_final_response(clinical, matches, safety_notes, full_text)
    warning = (
        "🚨 **تنبيه خطير:** حالتك تستدعي الطوارئ فوراً. "
        "الأدوية التالية هي حل مؤقت فقط حتى تتمكن من الوصول للمستشفى.\n\n"
    )
    return warning + body


def format_final_response(
    clinical: ClinicalPipelineResult,
    matches: list[IngredientMatch],
    safety_notes: list[SafetyNote] | None = None,
    full_text: str = "",
) -> str:
    safety_notes = safety_notes or []
    intro = clinical.patient_message_ar.strip()
    sections = []

    if intro:
        sections.append(intro)
        sections.append("")

    sections.extend([
        "## 1. ملخص الحالة",
        _format_assessment_summary(clinical),
        "",
        "## 2. احتمالات الحالة",
        _format_differential(clinical),
        "",
        "## 3. المواد الفعالة المقترحة",
        _format_ingredients(matches),
        "",
        "## 4. أدوية متوفرة من القاعدة",
        _format_drugs(matches),
        "",
        "## 5. ملاحظات أمان",
        _format_safety_notes(matches, safety_notes, clinical),
        "",
        "## 6. علامات تحتاج مراجعة طبية",
        build_red_flag_section(
            clinical.assessment,
            full_text,
            clinical.differential.red_flag_assessment,
        ),
    ])

    all_low = (
        clinical.differential.confidence_levels
        and all(c == "low" for c in clinical.differential.confidence_levels)
    )
    if all_low:
        sections.append("")
        sections.append("⚠️ الثقة التشخيصية منخفضة — لو الأعراض زادت، اكشف.")

    return "\n".join(sections)
