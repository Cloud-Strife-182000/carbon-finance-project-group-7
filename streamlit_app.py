# Minimal ESG Dashboard — v0.4: One Graph + Team (local folder images, circular)
# Title: “Carbon Finance Term Project | Group-7”

import os
from pathlib import Path
import pandas as pd
import streamlit as st
import altair as alt

# Page setup
st.set_page_config(page_title="Carbon Finance Term Project | Group-7", layout="wide")

# Title
st.title("Carbon Finance Term Project | Group-7")
st.caption("ESG Disclosures & Carbon Finance — Minimal Dashboard (v0.4)")

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
# Team section (circular photos from folder)
# -----------------------------
st.subheader("Project Team — Group 7")

# Folder containing team photos (place this folder next to your app file)
BASE_DIR = Path(__file__).parent
TEAM_DIR = BASE_DIR / "team_photos"   # e.g., ./team_photos/siddharth.jpg

# Define members with corresponding filenames in TEAM_DIR
team = [
    {"name": "Akshat Negi", "file": "p24akshatnegi.JPG"},
    {"name": "G R Srikanth",  "file": "p24srikanth.JPG"},
    {"name": "Siddharth Kumar Pandey", "file": "p24siddharth.JPG"},
    {"name": "Vineet Ranjan Maitrey",    "file": "p24vineet.jpg"}
    # Add or remove members as needed:
    # {"name": "Member 4", "file": "member4.png"},
    # {"name": "Member 5", "file": "member5.jpeg"},
]

# Create folder hint (doesn't create files; just helps users)
if not TEAM_DIR.exists():
    TEAM_DIR.mkdir(parents=True, exist_ok=True)
    st.info(f"Created {TEAM_DIR.as_posix()}. Add your image files there (e.g., siddharth.jpg).")

cols_per_row = 5
size_px = 120

# CSS for circular images
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
    .team-img {{
        width: {size_px}px;
        height: {size_px}px;
        border-radius: 50%;
        object-fit: cover;
        border: 2px solid #e5e7eb;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }}
    .team-name {{
        margin-top: 8px;
        font-weight: 600;
        font-size: 0.95rem;
    }}
    .team-missing {{
        width: {size_px}px;
        height: {size_px}px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        background: #f3f4f6;
        color: #6b7280;
        font-weight: 600;
        border: 2px dashed #e5e7eb;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

def img_tag_or_placeholder(file_path: Path, alt_text: str) -> str:
    """Return an <img> tag if file exists, else a circular placeholder."""
    if file_path.exists():
        # Use POSIX-style path for HTML src
        return f'<img class="team-img" src="{file_path.as_posix()}" alt="{alt_text}"/>'
    # Placeholder circle with initials if file missing
    initials = "".join([p[0] for p in alt_text.split() if p]).upper()[:2]
    return f'<div class="team-missing">{initials}</div>'

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
            photo_path = (TEAM_DIR / m["file"]).resolve()
            with c:
                st.markdown(
                    f"""
                    <div class="team-wrap">
                        {img_tag_or_placeholder(photo_path, m["name"])}
                        <div class="team-name">{m['name']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            idx += 1

render_team(team, cols_per_row=cols_per_row)
