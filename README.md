# egypt-medical-api

Egyptian Arabic pharmacy assistant API built with FastAPI, Gemini structured
assessment, and a local Egyptian drug database.

## Clinical Workflow

```text
Safety Intake
-> Symptom Clarification
-> Clinical Assessment
-> Differential Diagnosis
-> Therapeutic Targets
-> Drug Retrieval
-> Final Response
```

The LLM is only responsible for structured clinical assessment, differential
diagnosis, and therapeutic targets. Product retrieval is deterministic and must
return products from the dataset with product name, active ingredient, and row ID.

## Setup

```bash
pip install -r requirements.txt
```

Required environment variables:

- `GEMINI_API_KEY`
- Optional: `GEMINI_API_KEY_2`, `GEMINI_API_KEY_3`
- Optional: `GEMINI_MODEL`, default `gemini-2.5-flash`
- Optional: `CHAT_TIMEOUT_SEC`, default `50`

Required local files:

- `egypt_drugs_cleaned_utf8.csv`
- `faiss.index`

Run the API:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

## Verification

```bash
pytest
python eval/run_benchmark.py
```

The benchmark is deterministic and covers at least 100 Egyptian Arabic patient
cases across safety intake, extraction, symptom clarification, chronic disease
caution, urgent presentations, and retrieval-shape compliance.
