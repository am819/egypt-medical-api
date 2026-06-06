"""
app.py — Production FastAPI backend for the Egyptian Medical AI Chatbot
========================================================================
Conversational Gemini doctor + Egyptian drug database lookup by active ingredient.

Run with:
    uvicorn app:app --host 0.0.0.0 --port $PORT

Required files next to app.py:
    egypt_drugs_cleaned_utf8.csv
    faiss.index          (optional — generate with build_index.py)
"""

import asyncio
import os
from threading import Lock

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from pipeline import dataset
from pipeline.gemini import GEMINI_API_KEYS, get_keys_status
from pipeline.orchestrator import rag

CHAT_TIMEOUT_SEC = int(os.getenv("CHAT_TIMEOUT_SEC", "60"))
_rag_lock = Lock()

# ── Load drug dataset at startup ─────────────────────────────────────────────
try:
    dataset.load_dataset()
except Exception as e:
    print(f"❌ Startup failed (CSV/FAISS): {e}")
    print("   App starting in degraded mode — check /health")


# ══════════════════════════════════════════════════════════════════════════════
# FASTAPI APPLICATION
# ══════════════════════════════════════════════════════════════════════════════
app = FastAPI(
    title="Egyptian Medical AI API",
    description="محادثة طبية ذكية + قاعدة أدوية مصرية — Gemini 2.5 Flash",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    history: list = []


class ChatResponse(BaseModel):
    response: str


@app.get("/", include_in_schema=False)
def serve_ui():
    return FileResponse("index.html")


@app.post("/chat", response_model=ChatResponse, summary="Send a message and receive an AI response")
async def chat_endpoint(body: ChatRequest):
    if not body.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty")

    def _run_rag():
        with _rag_lock:
            return rag(body.message, body.history)

    try:
        response_text = await asyncio.wait_for(
            asyncio.to_thread(_run_rag),
            timeout=CHAT_TIMEOUT_SEC,
        )
    except asyncio.TimeoutError:
        print("❌ /chat timed out")
        return ChatResponse(
            response="معلش، الطلب أخد وقت طويل — استنى شوية وحاول تاني."
        )
    except Exception as e:
        print(f"❌ /chat unhandled error: {e}")
        return ChatResponse(
            response="عذراً، حدث خطأ مؤقت — حاول تاني بعد شوية."
        )

    return ChatResponse(response=response_text)


@app.get("/health", summary="Health check")
def health_check():
    degraded = dataset.index is None or dataset.df.empty
    return {
        "status": "degraded" if degraded else "ok",
        "keys_loaded": len(GEMINI_API_KEYS),
        "keys_status": get_keys_status(),
        "index_loaded": dataset.index is not None,
        "drug_count": len(dataset.df),
    }


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
