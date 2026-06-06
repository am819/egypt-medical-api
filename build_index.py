"""
build_index.py — Offline FAISS index builder for the Egyptian drug dataset.

Run once locally (requires ≥2GB RAM for SentenceTransformer):

    python build_index.py

Produces:
    faiss.index  — IndexFlatIP over normalized embeddings of the `combined` column

Commit faiss.index alongside egypt_drugs_cleaned_utf8.csv for Railway deployment.
"""

import os
import sys

import faiss
import numpy as np
import pandas as pd

CSV_PATH = os.getenv("EGYPT_DRUGS_CSV", "egypt_drugs_cleaned_utf8.csv")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "faiss.index")
EMBED_MODEL_NAME = os.getenv(
    "EMBED_MODEL_NAME",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)


def main() -> None:
    if not os.path.exists(CSV_PATH):
        print(f"❌ CSV not found: {CSV_PATH}")
        sys.exit(1)

    from sentence_transformers import SentenceTransformer

    print(f"Loading CSV: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH).fillna("").astype(str)
    ingredient_col = "ingredient_clean" if "ingredient_clean" in df.columns else "active_ingredient"

    if "combined" not in df.columns:
        df["combined"] = (
            df.get("name_ar", pd.Series([""] * len(df)))
            + " "
            + df.get("name_en", pd.Series([""] * len(df)))
            + " "
            + df.get(ingredient_col, pd.Series([""] * len(df)))
        )

    texts = df["combined"].tolist()
    print(f"Encoding {len(texts)} rows with {EMBED_MODEL_NAME}…")
    model = SentenceTransformer(EMBED_MODEL_NAME)
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64).astype("float32")
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, FAISS_INDEX_PATH)
    print(f"✅ Wrote {FAISS_INDEX_PATH} — {index.ntotal} vectors, dim={dim}")


if __name__ == "__main__":
    main()
