# Minimal ESG Dashboard — v0.5: One Graph + Team (local folder images via base64, circular)
# Title: “Carbon Finance Term Project | Group-7”

import os
import base64
import mimetypes
from pathlib import Path
import pandas as pd
import streamlit as st
import altair as alt

# Page setup
st.set_page_config(page_title="Carbon Finance Term Project | Group-7", layout="wide")

# Title
st.title("Carbon Finance Term Project | Group-7")
st.caption("ESG Disclosures & Carbon Finance — Minimal Dashboard (v0.5)")

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
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
        df["green_bonds_inr_cr"] = pd.to_numeric(df["green_bonds_inr_cr"], errors="coerce")
        df = df.dropna(subset=["year", "green_bonds_inr_cr"])
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()
else:
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

st.divider()

# -----------------------------
# Team section (circular photos from folder via base64)
# -----------------------------
st.subheader("Project Team — Group 7")

# Folder containing team photos (place this folder next to your app file)
BASE_DIR = Path(__file__).parent
TEAM_DIR = BASE_DIR / "team_photos"   # e.g., ./team_photos/p24siddharth.JPG

# Define members with corresponding filenames in TEAM_DIR
team = [
    {"name": "Akshat Negi",             "file": "p24akshatnegi.JPG"},
    {"name": "G R Srikanth",            "file": "p24srikanth.JPG"},
    {"name": "Siddharth Kumar Pandey",  "file": "p24siddharth.JPG"},
    {"name": "Vineet Ranjan Maitrey",   "file": "p24vineet.jpg"},
    # Add or remove members as needed:
    # {"name": "Member 5", "file": "member5.jpeg"},
]

# Create folder hint (doesn't create files; just helps users)
if not TEAM_DIR.exists():
    TEAM_DIR.mkdir(parents=True, exist_ok=True)
    st.info(f"Created {TEAM_DIR.as_posix()}. Add your image files there (e.g., p24siddharth.JPG).")

cols_per_row = 5
size_px = 120

# CSS for circular images / placeholders
st.markdown(
    f"""
    <style>
    .team-wrap {{
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
        text-align: center;
        padding: 8px 4px;
    }}
    .team-img, .team-missing {{
        width: {size_px}px;
        height: {size_px}px;
        border-radius: 50%;
        object-fit: cover;
        border: 2px solid #e5e7eb;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }}
    .team-missing {{
        display: flex;
        align-items: center;
        justify-content: center;
        background: #f3f4f6;
        color: #6b7280;
        font-weight: 700;
        font-size: 1.2rem;
    }}
    .team-name {{
        margin-top: 8px;
        font-weight: 600;
        font-size: 0.95rem;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

def get_initials(name: str) -> str:
    """Return 1–2 uppercase initials for placeholder."""
    parts = name.strip().split()
    if not parts:
        return "?"
    if len(parts) == 1:
        return parts[0][0].upper()
    return (parts[0][0] + parts[-1][0]).upper()

def to_data_uri(path: Path):
    """
    Read file bytes and return a data: URI (base64).
    Returns None if file missing or unreadable.
    """
    try:
        mime, _ = mimetypes.guess_type(path.name)
        if mime is None:
            mime = "image/jpeg"
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        return f"data:{mime};base64,{b64}"
    except Exception:
        return None

def img_or_placeholder(path: Path, name: str) -> str:
    data_uri = to_data_uri(path)
    if data_uri:
        return f'<img class="team-img" src="{data_uri}" alt="{name}"/>'
    return f'<div class="team-missing">{get_initials(name)}</div>'

def render_team(members, cols_per_row=5):
    if not members:
        st.info("Add your team members to the `team` list to display them here.")
        return
    rows = (len(members) + cols_per_row - 1) // cols_per_row
    idx = 0
    for _ in range(rows):
        cols = st.columns(cols_per_row)
        for c in cols:
            if idx >= len(members):
                c.empty()
                continue
            m = members[idx]
            photo_path = (TEAM_DIR / m["file"])
            with c:
                st.markdown(
                    f"""
                    <div class="team-wrap">
                        {img_or_placeholder(photo_path, m["name"])}
                        <div class="team-name">{m['name']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            idx += 1

render_team(team, cols_per_row=cols_per_row)
