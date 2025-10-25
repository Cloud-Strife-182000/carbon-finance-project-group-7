# Minimal ESG Dashboard — v0.6: One Graph + Team (cropped & downsized local photos via base64)
# Title: “Carbon Finance Term Project | Group-7”

import os
import io
import base64
import mimetypes
from pathlib import Path
import pandas as pd
import streamlit as st
import altair as alt

from PIL import Image, ImageOps  # NEW: for cropping/downsizing

# Page setup
st.set_page_config(page_title="Carbon Finance Term Project | Group-7", layout="wide")

# Title
st.title("Carbon Finance Term Project | Group-7")
st.caption("ESG Disclosures & Carbon Finance — Minimal Dashboard (v0.6)")

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
# Team section (cropped & resized circular photos via base64)
# -----------------------------
st.subheader("Project Team — Group 7")

BASE_DIR = Path(__file__).parent
TEAM_DIR = BASE_DIR / "team_photos"

team = [
    {"name": "Akshat Negi",             "file": "p24akshatnegi.JPG"},
    {"name": "G R Srikanth",            "file": "p24srikanth.JPG"},
    {"name": "Siddharth Kumar Pandey",  "file": "p24siddharth.JPG"},
    {"name": "Vineet Ranjan Maitrey",   "file": "p24vineet.jpg"},
]

if not TEAM_DIR.exists():
    TEAM_DIR.mkdir(parents=True, exist_ok=True)
    st.info(f"Created {TEAM_DIR.as_posix()}. Add your image files there (e.g., p24siddharth.JPG).")

# Visual diameter of the circular avatar (px)
size_px = 110          # adjust smaller/larger as you prefer (e.g., 96, 120, 140)
encode_px = size_px*2  # encode at 2x for crispness on HiDPI; CSS will display at size_px

# CSS
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
        font-size: 1.1rem;
    }}
    .team-name {{
        margin-top: 8px;
        font-weight: 600;
        font-size: 0.95rem;
        text-align: center;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

def get_initials(name: str) -> str:
    parts = name.strip().split()
    if not parts:
        return "?"
    if len(parts) == 1:
        return parts[0][0].upper()
    return (parts[0][0] + parts[-1][0]).upper()

def to_data_uri_cropped(path: Path, target_px: int) -> str | None:
    """
    Open image, auto-rotate from EXIF, center-crop to square,
    resize to target_px, and return base64 data URI (JPEG).
    """
    try:
        img = Image.open(path)
        img = ImageOps.exif_transpose(img)  # respect EXIF rotation

        # Convert to RGB to avoid issues saving as JPEG from RGBA/CMYK
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        # Center-crop to square
        w, h = img.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        img = img.crop((left, top, left + side, top + side))

        # Resize to target (2x for crispness if you want; we pass encode_px)
        img = img.resize((target_px, target_px), Image.Resampling.LANCZOS)

        # Encode to JPEG with reasonable quality
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85, optimize=True, progressive=True)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"
    except Exception:
        return None

def img_or_placeholder(path: Path, name: str) -> str:
    data_uri = to_data_uri_cropped(path, encode_px)
    if data_uri:
        return f'<img class="team-img" src="{data_uri}" alt="{name}"/>'
    return f'<div class="team-missing">{get_initials(name)}</div>'

def render_team(members, cols_per_row=4):
    rows = (len(members) + cols_per_row - 1) // cols_per_row
    idx = 0
    for _ in range(rows):
        cols = st.columns(cols_per_row)
        for c in cols:
            if idx >= len(members):
                c.empty()
                continue
            m = members[idx]
            photo_path = TEAM_DIR / m["file"]
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

render_team(team, cols_per_row=4)  # 4 columns makes each avatar a bit larger
