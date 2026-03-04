"""RankingFactores - Minimal landing. Use the sidebar to navigate to Scoring Profile Wizard or Metrics Operations."""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="RankingFactores", layout="wide")
st.title("RankingFactores")
st.caption("Financial data ranking and scoring profile builder.")

st.markdown("""
Select a page from the **sidebar**:

- **Scoring Profile Wizard** — Build scoring methodologies step-by-step with nested composition boxes.
- **Metrics Operations** — Create derived metrics (e.g. Debt/Assets) from existing metrics across all periods.
- **Scoring Profiles** — Retrieve, edit, and delete saved scoring profiles.
- **Periods** — Create periods from Excel/CSV, view and edit content (editable cells), remove securities/metrics, delete period.
- **Ranking** — Execute ranking for a period and scoring profile, filter by sector/industry, export to Excel.
""")
