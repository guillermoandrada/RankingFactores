"""RankingFactores - Home. Use the sidebar to navigate."""

from __future__ import annotations

import streamlit as st

from streamlit_app.ui.layout import inject_custom_css, render_page_header

st.set_page_config(page_title="RankingFactores", layout="wide")
inject_custom_css()
render_page_header("RankingFactores", "Financial data ranking and scoring profile builder.")

st.markdown("""
Select a page from the **sidebar** in workflow order:

1. **Periods** — Upload Excel data, create periods, view and edit content.
2. **Metrics** — Create derived metrics (e.g. Debt/Assets) from existing metrics.
3. **Scoring Profile Wizard** — Build scoring methodologies step-by-step.
4. **Scoring Profiles** — Edit and manage saved scoring profiles.
5. **Ranking** — Run rankings for a period and profile, export results.
""")
