"""Gemini API client with JSON structured output support."""

import os
import re
import time
from datetime import datetime
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

GEMINI_API_KEYS = [
    k for k in [
        os.getenv("GEMINI_API_KEY", ""),
        os.getenv("GEMINI_API_KEY_2", ""),
        os.getenv("GEMINI_API_KEY_3", ""),
    ] if k
]
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_TIMEOUT_SEC = int(os.getenv("GEMINI_TIMEOUT_SEC", "25"))
MIN_INTERVAL = 6.0  # 10 RPM per key (60s / 10)
GEMINI_KEY_EXHAUSTED_UNTIL: list[float] = [0.0] * len(GEMINI_API_KEYS)
_last_call_times: list[float] = [0.0] * len(GEMINI_API_KEYS)


def _trim_history_for_gemini(history: list, max_messages: int = 10) -> list:
    h = history or []
    return h if len(h) <= max_messages else h[-max_messages:]


def get_keys_status() -> list[str]:
    now = time.time()
    statuses: list[str] = []
    for i in range(len(GEMINI_API_KEYS)):
        until = GEMINI_KEY_EXHAUSTED_UNTIL[i] if i < len(GEMINI_KEY_EXHAUSTED_UNTIL) else 0.0
        if now > until:
            statuses.append("ready")
        else:
            t = datetime.fromtimestamp(until).strftime("%H:%M:%S")
            statuses.append(f"cooldown until {t}")
    return statuses


def _first_available_key_index(now: float) -> Optional[int]:
    for i in range(len(GEMINI_API_KEYS)):
        if now > GEMINI_KEY_EXHAUSTED_UNTIL[i]:
            return i
    return None


def _wait_for_next_key() -> int:
    now = time.time()
    key_idx = min(range(len(GEMINI_API_KEYS)), key=lambda i: GEMINI_KEY_EXHAUSTED_UNTIL[i])
    wait = GEMINI_KEY_EXHAUSTED_UNTIL[key_idx] - now
    if wait > 0:
        time.sleep(wait)
    return key_idx


def _another_available_key(current: int) -> Optional[int]:
    now = time.time()
    for offset in range(1, len(GEMINI_API_KEYS)):
        i = (current + offset) % len(GEMINI_API_KEYS)
        if now > GEMINI_KEY_EXHAUSTED_UNTIL[i]:
            return i
    return None


def _pick_key_index() -> int:
    now = time.time()
    key_idx = _first_available_key_index(now)
    if key_idx is None:
        key_idx = _wait_for_next_key()
    return key_idx


def _enforce_rpm(key_idx: int) -> None:
    elapsed = time.time() - _last_call_times[key_idx]
    if elapsed < MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - elapsed)


def _parse_retry_after(response: requests.Response) -> float:
    retry_hdr = response.headers.get("Retry-After") or response.headers.get("retry-after")
    if retry_hdr:
        try:
            return float(retry_hdr)
        except ValueError:
            pass
    return 60.0


def _is_unavailable(response: requests.Response, error: dict) -> bool:
    if response.status_code == 503:
        return True
    return str(error.get("status", "")).upper() == "UNAVAILABLE"


def _is_rate_limit(response: requests.Response, error: dict) -> bool:
    if response.status_code == 429:
        return True
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
    # Picks first key off cooldown (or waits for soonest); on 503/429 marks key exhausted and rotates.
    # Each key enforces its own 6s RPM interval; success returns immediately.
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
    key_idx = _pick_key_index()

    while True:
        _enforce_rpm(key_idx)
        api_key = GEMINI_API_KEYS[key_idx]
        headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
        try:
            _last_call_times[key_idx] = time.time()
            r = requests.post(url, headers=headers, json=payload, timeout=GEMINI_TIMEOUT_SEC)
            resp = r.json()
            if "error" in resp:
                err = resp["error"]
                print(f"❌ Gemini API error (key {key_idx + 1}/{n_keys}): {err}")
                if _is_unavailable(r, err):
                    GEMINI_KEY_EXHAUSTED_UNTIL[key_idx] = time.time() + 60
                    print(f"🔑 Key {key_idx + 1} unavailable — cooldown 60s, rotating")
                    key_idx = _pick_key_index()
                    continue
                if _is_rate_limit(r, err):
                    retry_after = _parse_retry_after(r)
                    GEMINI_KEY_EXHAUSTED_UNTIL[key_idx] = time.time() + retry_after
                    print(f"🔑 Key {key_idx + 1} rate-limited — cooldown {retry_after}s, rotating")
                    key_idx = _pick_key_index()
                    continue
                return None, "gemini_error"
            candidates = resp.get("candidates") or []
            if not candidates:
                print(f"❌ Gemini empty response (key {key_idx + 1}): {resp}")
                other = _another_available_key(key_idx)
                if other is not None:
                    key_idx = other
                    continue
                return None, "rate_limit"
            parts = candidates[0].get("content", {}).get("parts") or []
            if not parts or "text" not in parts[0]:
                print(f"❌ Gemini missing text (key {key_idx + 1}): {resp}")
                other = _another_available_key(key_idx)
                if other is not None:
                    key_idx = other
                    continue
                return None, "rate_limit"
            text = parts[0]["text"].strip()
            text = re.sub(r"\(Internal Reasoning\).*?(?=\n\n|\Z)", "", text, flags=re.DOTALL)
            text = re.sub(r"\(Response.*?\):\s*", "", text)
            return text, GEMINI_MODEL
        except requests.Timeout:
            print(f"❌ Gemini timeout (key {key_idx + 1}/{n_keys})")
            other = _another_available_key(key_idx)
            if other is not None:
                key_idx = other
                continue
            return None, "rate_limit"
        except Exception as e:
            print(f"❌ Gemini call exception (key {key_idx + 1}): {e}")
            return None, "gemini_error"


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
