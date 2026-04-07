"""RankingFactores - Minimal landing. Use the sidebar to navigate to Scoring Profile Wizard or Metrics Operations."""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit runs this script with cwd/path that may not include the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
""")
