"""Gemini API client with JSON structured output support."""

import os
import re
import time
from typing import Any, Optional

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from pipeline.assessment import parse_clinical_pipeline
from pipeline.context import PatientContext, build_patient_summary
from pipeline.models import ClinicalPipelineResult, clinical_pipeline_json_schema
from pipeline.prompts import CLINICAL_PIPELINE_PROMPT, INTAKE_PROMPT, TEMPORARY_TREATMENT_NOTE

# Accept Railway-style names (spaces) and standard underscore names.
_GEMINI_KEY_ENV_ALIASES: tuple[tuple[str, ...], ...] = (
    ("GEMINI_API_KEY", "GEMINI API KEY", "GOOGLE_API_KEY"),
    ("GEMINI_API_KEY_2", "GEMINI API KEY 2", "GEMINI_API_KEY2"),
    ("GEMINI_API_KEY_3", "GEMINI API KEY 3", "GEMINI_API_KEY3"),
)


def _first_env_value(names: tuple[str, ...]) -> str:
    for name in names:
        val = os.getenv(name, "").strip()
        if val:
            return val
    return ""


def _looks_like_gemini_api_key(key: str) -> bool:
    """Google AI Studio keys start with AIza; OAuth/Vertex tokens (AQ.) won't work here."""
    return key.startswith("AIza")


def _load_gemini_api_keys() -> list:
    keys: list[str] = []
    for aliases in _GEMINI_KEY_ENV_ALIASES:
        val = _first_env_value(aliases)
        if not val:
            continue
        if not _looks_like_gemini_api_key(val):
            print(
                f"Skipping {aliases[0]}: expected Google AI Studio key (AIza...), "
                f"got {val[:8]}..."
            )
            continue
        keys.append(val)
    return keys


GEMINI_API_KEYS: list = _load_gemini_api_keys()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_TIMEOUT_SEC = int(os.getenv("GEMINI_TIMEOUT_SEC", "25"))
RPM_LIMIT = 10
MIN_INTERVAL = 60.0 / RPM_LIMIT
_last_call_time: float = 0.0
_gemini_key_index: int = 0


def _trim_history_for_gemini(history: list, max_messages: int = 10) -> list:
    h = history or []
    return h if len(h) <= max_messages else h[-max_messages:]


def _gemini_key_exhausted(error: dict) -> bool:
    code = error.get("code")
    if code in (429, 403):
        return True
    status = str(error.get("status", "")).upper()
    return "RESOURCE_EXHAUSTED" in status or "QUOTA" in status


def call_gemini(
    messages: list,
    system_prompt: str,
    *,
    json_schema: Optional[dict[str, Any]] = None,
    max_tokens: int = 3072,
) -> tuple[Optional[str], str]:
    global _last_call_time, _gemini_key_index
    elapsed = time.time() - _last_call_time
    if elapsed < MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - elapsed)

    if not GEMINI_API_KEYS:
        print("No Gemini API keys configured (set GEMINI_API_KEY or GEMINI API KEY)")
        return None, "no_api_key"

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    gen_config: dict[str, Any] = {
        "temperature": 0.1,
        "maxOutputTokens": max_tokens,
    }
    if json_schema:
        gen_config["responseMimeType"] = "application/json"
        gen_config["responseSchema"] = json_schema

    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": messages,
        "generationConfig": gen_config,
    }
    n_keys = len(GEMINI_API_KEYS)

    for key_attempt in range(n_keys):
        api_key = GEMINI_API_KEYS[_gemini_key_index]
        headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
        try:
            _last_call_time = time.time()
            r = requests.post(url, headers=headers, json=payload, timeout=GEMINI_TIMEOUT_SEC)
            resp = r.json()
            if "error" in resp:
                err = resp["error"]
                print(f"❌ Gemini API error (key {_gemini_key_index + 1}/{n_keys}): {err}")
                if _gemini_key_exhausted(err) and key_attempt < n_keys - 1:
                    _gemini_key_index = (_gemini_key_index + 1) % n_keys
                    print(f"🔑 Switching to Gemini API key {_gemini_key_index + 1}/{n_keys}")
                    continue
                if _gemini_key_exhausted(err):
                    return None, "rate_limit"
                return None, "gemini_error"
            candidates = resp.get("candidates") or []
            if not candidates:
                print(f"❌ Gemini empty response (key {_gemini_key_index + 1}): {resp}")
                if key_attempt < n_keys - 1:
                    _gemini_key_index = (_gemini_key_index + 1) % n_keys
                    continue
                return None, "rate_limit"
            parts = candidates[0].get("content", {}).get("parts") or []
            if not parts or "text" not in parts[0]:
                print(f"❌ Gemini missing text (key {_gemini_key_index + 1}): {resp}")
                if key_attempt < n_keys - 1:
                    _gemini_key_index = (_gemini_key_index + 1) % n_keys
                    continue
                return None, "rate_limit"
            text = parts[0]["text"].strip()
            text = re.sub(r"\(Internal Reasoning\).*?(?=\n\n|\Z)", "", text, flags=re.DOTALL)
            text = re.sub(r"\(Response.*?\):\s*", "", text)
            return text, GEMINI_MODEL
        except requests.Timeout:
            print(f"❌ Gemini timeout (key {_gemini_key_index + 1}/{n_keys})")
            if key_attempt < n_keys - 1:
                _gemini_key_index = (_gemini_key_index + 1) % n_keys
                continue
            return None, "rate_limit"
        except Exception as e:
            print(f"❌ Gemini call exception (key {_gemini_key_index + 1}): {e}")
            return None, "gemini_error"

    return None, "rate_limit"


def _build_gemini_messages(history: list, augmented: str) -> list:
    gemini_messages = [
        {
            "role": "user" if m["role"] == "user" else "model",
            "parts": [{"text": m["content"]}],
        }
        for m in _trim_history_for_gemini(history)
    ]
    gemini_messages.append({"role": "user", "parts": [{"text": augmented}]})
    return gemini_messages


def call_structured_clinical(
    query: str,
    history: list,
    ctx: PatientContext,
    *,
    temporary_override: bool = False,
) -> tuple[Optional[ClinicalPipelineResult], str]:
    """Call Gemini with JSON schema for Phases 1-3."""
    augmented = build_patient_summary(ctx) + "\nسؤال المريض الحالي:\n" + query.strip()
    if temporary_override:
        augmented += "\n\n" + TEMPORARY_TREATMENT_NOTE

    messages = _build_gemini_messages(history, augmented)
    raw, status = call_gemini(
        messages,
        CLINICAL_PIPELINE_PROMPT,
        json_schema=clinical_pipeline_json_schema(),
    )
    if status in ("rate_limit", "no_api_key"):
        return None, status
    if not raw:
        return None, status or "gemini_error"

    result = parse_clinical_pipeline(raw, ctx)
    if not result:
        return None, "parse_error"
    return result, "ok"


def call_intake_conversational(query: str, history: list, ctx: PatientContext) -> tuple[Optional[str], str]:
    """Conversational intake when structured pipeline returns needs_info."""
    augmented = build_patient_summary(ctx) + "\nسؤال المريض الحالي:\n" + query.strip()
    messages = _build_gemini_messages(history, augmented)
    return call_gemini(messages, INTAKE_PROMPT, max_tokens=1024)
