"""Drug dataset and FAISS index loading."""

import gc
import os
from typing import Any, Optional

import faiss
import pandas as pd
from rapidfuzz import process

CSV_PATH = os.getenv("EGYPT_DRUGS_CSV", "egypt_drugs_cleaned_utf8.csv")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "faiss.index")
EMBED_MODEL_NAME = os.getenv(
    "EMBED_MODEL_NAME",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
ENABLE_SEMANTIC_SEARCH = os.getenv(
    "ENABLE_SEMANTIC_SEARCH", "false"
).strip().lower() in ("1", "true", "yes")

index: Optional[faiss.Index] = None
embed_model: Optional[Any] = None
_embed_load_failed: bool = False
INGREDIENT_COL = "active_ingredient"
df = pd.DataFrame()


def load_dataset() -> None:
    global index, df, INGREDIENT_COL, _embed_load_failed
    try:
        df_raw = pd.read_csv(CSV_PATH).fillna("").astype(str)
        INGREDIENT_COL = (
            "ingredient_clean" if "ingredient_clean" in df_raw.columns else "active_ingredient"
        )
        if "combined" not in df_raw.columns:
            df_raw["combined"] = (
                df_raw.get("name_ar", pd.Series([""] * len(df_raw)))
                + " "
                + df_raw.get("name_en", pd.Series([""] * len(df_raw)))
                + " "
                + df_raw.get(INGREDIENT_COL, pd.Series([""] * len(df_raw)))
            )
        df = df_raw.reset_index(drop=True)
        del df_raw
        gc.collect()
        print(f"✅ CSV loaded — {len(df)} rows")

        index = faiss.read_index(FAISS_INDEX_PATH)
        print(f"✅ FAISS index loaded — {index.ntotal} vectors")

        if ENABLE_SEMANTIC_SEARCH:
            print("✅ Startup complete — SentenceTransformer loads on first drug search")
        else:
            print("✅ Startup complete — drug lookup via rapidfuzz (semantic search off)")

    except FileNotFoundError as e:
        print(f"❌ Missing precomputed file: {e}")
        print("   Run build_index.py offline to generate faiss.index")
    except Exception as e:
        print(f"❌ Startup error: {e}")


def get_embed_model() -> Optional[Any]:
    global embed_model, _embed_load_failed
    if not ENABLE_SEMANTIC_SEARCH or _embed_load_failed:
        return None
    if embed_model is not None:
        return embed_model
    try:
        from sentence_transformers import SentenceTransformer
        print(f"🔵 Loading SentenceTransformer ({EMBED_MODEL_NAME})…")
        embed_model = SentenceTransformer(EMBED_MODEL_NAME)
        print("🟢 SentenceTransformer ready")
        return embed_model
    except Exception as e:
        print(f"❌ SentenceTransformer load failed — rapidfuzz fallback: {e}")
        _embed_load_failed = True
        return None


def semantic_candidate_indices(query_text: str, top_k: int = 40) -> list:
    if df.empty:
        return []

    if index is not None and get_embed_model() is not None:
        try:
            q = get_embed_model().encode([query_text]).astype("float32")
            faiss.normalize_L2(q)
            _scores, ids = index.search(q, min(top_k, index.ntotal))
            return [int(i) for i in ids[0] if i >= 0]
        except Exception as exc:
            print(f"⚠️ FAISS search failed ({exc}), falling back to rapidfuzz")

    try:
        hits = process.extract(query_text, df[INGREDIENT_COL].tolist(), limit=top_k)
        return [hit[2] for hit in hits if hit[1] > 0]
    except Exception:
        return list(range(min(len(df), top_k)))
