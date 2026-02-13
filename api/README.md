# RankingFactores API

FastAPI application for uploading financial data and computing security rankings.

## Run the API

From the project root:

```bash
uvicorn api.main:app --reload
```

API docs: http://127.0.0.1:8000/docs

## Endpoints

### GET /metrics

Return all metrics available in the database.

- **Response**: `{ "metrics": [ { "metric_id", "metric_name", "higher_is_better" }, ... ] }`

```bash
curl http://127.0.0.1:8000/metrics
```

### PATCH /metrics/{metric_id}

Update the `higher_is_better` column for a metric.

- **Body** (JSON): `{ "higher_is_better": true }` or `{ "higher_is_better": false }`
- **Response**: `{ "success": true, "metric_id": 2 }`

```bash
curl -X PATCH http://127.0.0.1:8000/metrics/2 \
  -H "Content-Type: application/json" \
  -d '{"higher_is_better": false}'
```

### POST /upload

Upload an Excel file to import into the database.

- **Body**: `multipart/form-data` with `file` (Excel .xlsx or .xls)
- **Response**: Import summary (period, companies_count, metrics_count, records_count, index_code)

```bash
curl -X POST -F "file=@2023 Q3.xlsx" http://127.0.0.1:8000/upload
```

### POST /ranking

Compute security ranking for a given quarter.

- **Body** (JSON):
  - `quarter` (required): e.g. `"2023 Q3"`
  - `industry` (optional): Filter by industry. Omit or empty for all companies.
  - `method` (required): `"linear"` or `"softplus"`
  - `weights` (required): `{ "metric_name": weight, ... }`

```bash
curl -X POST http://127.0.0.1:8000/ranking \
  -H "Content-Type: application/json" \
  -d '{
    "quarter": "2023 Q3",
    "industry": "",
    "method": "linear",
    "weights": {
      "Current Book to Price": 1.0,
      "5Y Average Book to Price": 0.5
    }
  }'
```
