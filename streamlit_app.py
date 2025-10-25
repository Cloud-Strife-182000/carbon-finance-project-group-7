# Minimal ESG Dashboard — v0: One Graph
# Goal: keep it ultra-simple and build up later.

import pandas as pd
import streamlit as st
import altair as alt

# Page setup
st.set_page_config(page_title="Carbon Finance Term Project | Group-7", layout="wide")

# Title
st.title("Carbon Finance Term Project | Group-7")
st.caption("Just one chart to start. We’ll add more later.")

# Sidebar uploader
st.sidebar.header("Upload (optional)")
st.sidebar.write(
    "Provide a CSV with two columns: `year`, `green_bonds_inr_cr`. "
    "If not provided, we use a small demo dataset."
)

uploaded = st.sidebar.file_uploader("CSV file", type=["csv"])

# Data loading
if uploaded:
    try:
        df = pd.read_csv(uploaded)
        needed = {"year", "green_bonds_inr_cr"}
        if not needed.issubset(set(df.columns)):
            st.error(f"CSV must have columns: {sorted(list(needed))}")
            st.stop()
        # Ensure numeric types
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
        df["green_bonds_inr_cr"] = pd.to_numeric(df["green_bonds_inr_cr"], errors="coerce")
        df = df.dropna(subset=["year", "green_bonds_inr_cr"])
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()
else:
    # Demo data
    df = pd.DataFrame({
        "year": [2019, 2020, 2021, 2022, 2023, 2024],
        "green_bonds_inr_cr": [800, 650, 1200, 2100, 2600, 3100],
    })

# Chart
st.subheader("Green Bonds by Year (₹ crore)")

chart = (
    alt.Chart(df)
    .mark_line(point=True)
    .encode(
        x=alt.X("year:O", title="Year"),
        y=alt.Y("green_bonds_inr_cr:Q", title="₹ crore"),
        tooltip=["year", "green_bonds_inr_cr"]
    )
    .properties(height=420)
)

st.altair_chart(chart, use_container_width=True)

# Data preview
st.write("Data preview")
st.dataframe(df, width='stretch', hide_index=True)
