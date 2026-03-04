# RankingFactores API

FastAPI application for uploading financial data and computing security rankings.

## Run the API

From the project root:

```bash
uvicorn api.main:app --reload
```

API docs: http://127.0.0.1:8000/docs

## Resource Routers

Each router has exactly 4 endpoints: GET, POST, PUT, DELETE. Industries and Sectors are GET-only.

### Periods Router (`/periods`)

Handles period content (securities with their metrics).

- `GET /periods/{period}` — Retrieve securities with their associated metrics
- `POST /periods` — Create a new period table from imported Excel/CSV. Params: `reader` (excel|csv|auto), `if_period_exists` (replace|append)
- `PUT /periods/{period}` — Edit the period table: upload file to replace, or body with `remove_metrics`, `remove_securities`, `delete_metrics`, `update_values`
- `DELETE /periods/{period}` — Fully eliminate the period table

Create period (replace if exists):

```bash
curl -X POST -F "file=@2023 Q3.xlsx" "http://127.0.0.1:8000/periods?reader=auto&if_period_exists=replace"
```

Create period (append/merge if exists):

```bash
curl -X POST -F "file=@2023 Q3.xlsx" "http://127.0.0.1:8000/periods?if_period_exists=append"
```

Edit period (remove metrics, update values):

```bash
curl -X PUT http://127.0.0.1:8000/periods/2023%20Q3 \
  -H "Content-Type: application/json" \
  -d '{"remove_metrics": [1, 2], "update_values": [{"ticker": "AAPL", "metric_name": "Revenue", "value": 100.5}]}'
```

Get period content:

```bash
curl -X GET http://127.0.0.1:8000/periods/2023%20Q3
```

### Scorings Router (`/scorings`)

Compute period scoring.

- `POST /scorings/{period}` — Compute period scoring

Parameters: `scoring_profile` (required), `industry` (optional), `sector` (optional), `export` (boolean).

```bash
curl -X POST http://127.0.0.1:8000/scorings/2023%20Q3 \
  -H "Content-Type: application/json" \
  -d '{
    "scoring_profile": "Value Test",
    "industry": "",
    "sector": "",
    "export": false
  }'
```

Export to XLSX:

```bash
curl -X POST http://127.0.0.1:8000/scorings/2023%20Q3 \
  -H "Content-Type: application/json" \
  -d '{
    "scoring_profile": "Value Test",
    "industry": "",
    "sector": "",
    "export": true
  }' \
  --output ranking.xlsx
```

### Metrics Router (`/metrics`)

- `GET /metrics` — List metrics; `GET /metrics?metric_id=N` — Get one metric
- `POST /metrics` — Create simple metric or derived metric
- `PUT /metrics/{metric_id}` — Update metric
- `DELETE /metrics/{metric_id}` — Delete metric

Update metric direction:

```bash
curl -X PUT http://127.0.0.1:8000/metrics/2 \
  -H "Content-Type: application/json" \
  -d '{"higher_is_better": false}'
```

Create simple metric:

```bash
curl -X POST http://127.0.0.1:8000/metrics \
  -H "Content-Type: application/json" \
  -d '{"metric_name": "ROE", "higher_is_better": true}'
```

Create derived metric:

```bash
curl -X POST http://127.0.0.1:8000/metrics \
  -H "Content-Type: application/json" \
  -d '{
    "metric_names": ["Debt", "Assets"],
    "operations": ["/"],
    "new_metric_name": "Debt/Assets",
    "higher_is_better": false
  }'
```

### Scoring Profiles Router (`/scoring-profiles`)

- `GET /scoring-profiles` — Get Scoring Profiles; if `profile_name` empty, returns all; otherwise returns single profile
- `POST /scoring-profiles` — Create profile (body: `profile_name`, `profile`)
- `PUT /scoring-profiles/{profile_name}` — Upsert profile
- `DELETE /scoring-profiles/{profile_name}` — Delete profile

### Reference (Periods, Industries & Sectors — GET only)

- `GET /periods` — List available periods
- `GET /sectors` — List sectors
- `GET /industries` — List industries
