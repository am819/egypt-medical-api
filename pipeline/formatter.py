"""Phase 7 — structured Arabic medical response formatter."""

from pipeline.models import ClinicalPipelineResult, IngredientMatch, SafetyNote

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
        lines.append(f"{i + 1}. الأعراض دي ممكن تكون متوافقة مع **{cond}** (ثقة: {CONFIDENCE_AR.get(conf, conf)})")
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
            blocks.append(f"   💊 **{d.name_ar or '—'}** | {d.name_en or '—'}")
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
                if c not in lines:
                    lines.append(f"⚠️ {d.name_ar}: {c}")
    if clinical.non_drug_advice:
        lines.append("📋 نصائح غير دوائية:")
        for adv in clinical.non_drug_advice:
            lines.append(f"   • {adv}")
    return "\n".join(lines) if lines else "لا توجد ملاحظات أمان إضافية."


def _format_red_flags(clinical: ClinicalPipelineResult) -> str:
    flags = list(clinical.assessment.red_flags)
    if clinical.differential.red_flag_assessment:
        flags.append(clinical.differential.red_flag_assessment)
    all_low = (
        clinical.differential.confidence_levels
        and all(c == "low" for c in clinical.differential.confidence_levels)
    )
    if all_low:
        flags.append("الثقة التشخيصية منخفضة — لو الأعراض زادت، اكشف فوراً.")
    if not flags:
        return "لا توجد علامات خطر إضافية حالياً — راقب الأعراض."
    return "\n".join(f"🚨 {f}" for f in flags)


def format_urgent_response(clinical: ClinicalPipelineResult) -> str:
    intro = clinical.patient_message_ar or "فيه علامات خطر محتملة."
    parts = [
        intro,
        "",
        "🚨 **روح الطوارئ فوراً.**",
        "",
        "## ملخص الحالة",
        _format_assessment_summary(clinical),
        "",
        "## تقييم علامات الخطر",
        clinical.differential.red_flag_assessment or _format_red_flags(clinical),
    ]
    return "\n".join(parts)


def format_temporary_response(
    clinical: ClinicalPipelineResult,
    matches: list[IngredientMatch],
    safety_notes: list[SafetyNote],
) -> str:
    body = format_final_response(clinical, matches, safety_notes)
    warning = (
        "🚨 **تنبيه خطير:** حالتك تستدعي الطوارئ فوراً. "
        "الأدوية التالية هي حل مؤقت فقط حتى تتمكن من الوصول للمستشفى.\n\n"
    )
    return warning + body


def format_final_response(
    clinical: ClinicalPipelineResult,
    matches: list[IngredientMatch],
    safety_notes: list[SafetyNote] | None = None,
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
        _format_red_flags(clinical),
    ])

    all_low = (
        clinical.differential.confidence_levels
        and all(c == "low" for c in clinical.differential.confidence_levels)
    )
    if all_low:
        sections.append("")
        sections.append("⚠️ الثقة التشخيصية منخفضة — لو الأعراض زادت، اكشف.")

    return "\n".join(sections)
