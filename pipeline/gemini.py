"""Gemini API client with key rotation."""

import os
import re
import time
from datetime import datetime
from typing import Optional

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from pipeline.prompts import SYSTEM_PROMPT

GEMINI_API_KEYS = [
    k for k in [
        os.getenv("GEMINI_API_KEY", ""),
        os.getenv("GEMINI_API_KEY_2", ""),
        os.getenv("GEMINI_API_KEY_3", ""),
    ] if k
]
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_TIMEOUT_SEC = int(os.getenv("GEMINI_TIMEOUT_SEC", "40"))
MIN_INTERVAL = 6.0  # 10 RPM per key (60s / 10)
GEMINI_KEY_EXHAUSTED_UNTIL: list[float] = [0.0] * len(GEMINI_API_KEYS)
_last_call_times: list[float] = [0.0] * len(GEMINI_API_KEYS)

CLINICAL_PLAN_MARKER = "───CLINICAL_PLAN───"


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
    max_tokens: int = 2048,
) -> tuple[Optional[str], str]:
    # Picks first key off cooldown (or waits for soonest); on 503/429 marks key exhausted and rotates.
    # Each key enforces its own 6s RPM interval; success returns immediately.
    if not GEMINI_API_KEYS:
        print("No Gemini API keys configured (set GEMINI_API_KEY or GEMINI API KEY)")
        return None, "no_api_key"

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": messages,
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": max_tokens,
        },
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
                print(f"Gemini API error (key {key_idx + 1}/{n_keys}): {err}")
                if _is_unavailable(r, err):
                    GEMINI_KEY_EXHAUSTED_UNTIL[key_idx] = time.time() + 60
                    key_idx = _pick_key_index()
                    continue
                if _is_rate_limit(r, err):
                    retry_after = _parse_retry_after(r)
                    GEMINI_KEY_EXHAUSTED_UNTIL[key_idx] = time.time() + retry_after
                    key_idx = _pick_key_index()
                    continue
                return None, "gemini_error"
            candidates = resp.get("candidates") or []
            if not candidates:
                other = _another_available_key(key_idx)
                if other is not None:
                    key_idx = other
                    continue
                return None, "rate_limit"
            parts = candidates[0].get("content", {}).get("parts") or []
            if not parts or "text" not in parts[0]:
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
            other = _another_available_key(key_idx)
            if other is not None:
                key_idx = other
                continue
            return None, "rate_limit"
        except Exception as e:
            print(f"Gemini call exception (key {key_idx + 1}): {e}")
            return None, "gemini_error"


def _build_messages(history: list, query: str) -> list:
    gemini_messages = [
        {
            "role": "user" if m["role"] == "user" else "model",
            "parts": [{"text": m["content"]}],
        }
        for m in _trim_history_for_gemini(history)
    ]
    gemini_messages.append({"role": "user", "parts": [{"text": query.strip()}]})
    return gemini_messages


def chat(query: str, history: list) -> tuple[Optional[str], str]:
    """Single conversational Gemini call with SYSTEM_PROMPT."""
    messages = _build_messages(history, query)
    return call_gemini(messages, SYSTEM_PROMPT, max_tokens=2048)
