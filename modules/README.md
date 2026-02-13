# RankingFactores Modules (OOP Structure)

## Architecture Overview

```
modules/
├── config/           # Settings and constants
├── models/           # Dataclasses (ImportResult, RankingParams, RankingResult)
├── db/               # Database layer
├── ingestion/        # File reading and import pipeline
└── analytics/        # Z-scores, ranking, export
```

## Usage

### Import data
```bash
python run_import.py                    # Uses default file (2023 Q3.xlsx)
python run_import.py "2024 Q1.xlsx"      # Specific file
python -m modules.ingestion.importer "2024 Q1.xlsx"
```

### Run ranking CLI
```bash
python run_ranking.py
python -m modules.analytics.cli
```

### Programmatic usage
```python
from modules.ingestion import DataImporter, FileReader
from modules.db import FinancialDatabase
from modules.analytics import ZScoreCalculator, RankingEngine

# Import
importer = DataImporter()
result = importer.import_file("2023 Q3.xlsx")

# Ranking
db = FinancialDatabase()
calc = ZScoreCalculator(engine=db.engine)
df_z, direction_map = calc.compute(period="2023 Q3", metric_ids=[1,2,3])
engine = RankingEngine()
df_scored = engine.compute(df_z, weights={"Metric": 1.0}, direction_map=direction_map)
```

## Key Classes

| Class | Purpose |
|-------|---------|
| `FinancialDatabase` | Repository for persisting fundamentals |
| `FileReader` | Reads Excel/CSV, extracts period and index code |
| `DataImporter` | Orchestrates read → validate → persist |
| `ZScoreCalculator` | Winsorized z-scores with index/industry filters |
| `RankingEngine` | Linear or softplus score combination |
