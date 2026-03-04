# Streamlit Front End

This folder contains a Streamlit UI that uses the FastAPI backend to:

- fetch available metrics from the DB
- build scoring profile JSON payloads interactively
- manage scoring profiles and score sets
- run rankings and download XLSX exports

## Run

From project root:

```bash
pip install -r requirements.txt
streamlit run frontend_streamlit/app.py
```

Default API URL in the UI is:

`http://127.0.0.1:8000`

Make sure FastAPI is running first, for example:

```bash
uvicorn api.main:app --reload
```
