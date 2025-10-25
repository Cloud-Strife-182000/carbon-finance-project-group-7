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

import io
import base64
import mimetypes
from pathlib import Path
from PIL import Image, ImageOps

BASE_DIR = Path(__file__).parent
TEAM_DIR = BASE_DIR / "team_photos"

# Add optional offset_y / offset_x in [-1.0, 1.0].
# tip: use offset_y ~ -0.25 to shift crop upward so more forehead fits inside the circle.
team = [
    {"name": "Akshat Negi",             "file": "p24akshatnegi.JPG", "offset_y": -0.30},
    {"name": "G R Srikanth",            "file": "p24srikanth.JPG",   "offset_y": -0.30},
    {"name": "Siddharth Kumar Pandey",  "file": "p24siddharth.JPG",   "offset_y": -0.30},
    {"name": "Vineet Ranjan Maitrey",   "file": "p24vineet.jpg"},
]

if not TEAM_DIR.exists():
    TEAM_DIR.mkdir(parents=True, exist_ok=True)
    st.info(f"Created {TEAM_DIR.as_posix()}. Add your image files there (e.g., p24siddharth.JPG).")

# Visual diameter (px). Increase/decrease as needed.
size_px = 110
encode_px = size_px * 2  # render at 2x for HiDPI sharpness

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

def center_square_crop_with_offset(img: Image.Image, offset_x: float = 0.0, offset_y: float = 0.0) -> Image.Image:
    """
    Center-square crop, but allow nudging the crop window by offsets in [-1, 1].
    offset_x: -1 fully left, +1 fully right
    offset_y: -1 fully up,   +1 fully down
    """
    # EXIF-aware orientation
    img = ImageOps.exif_transpose(img)
    # Ensure RGB for JPEG encoding later
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    w, h = img.size
    side = min(w, h)

    # How far we can move the square window
    max_left = w - side
    max_top = h - side

    # Clamp offsets to [-1, 1]
    ox = max(-1.0, min(1.0, float(offset_x)))
    oy = max(-1.0, min(1.0, float(offset_y)))

    # Base (centered) top-left
    left = (w - side) / 2.0
    top = (h - side) / 2.0

    # Apply offsets: scale by half the range so ±1 reaches the extremes
    left = left + ox * (max_left / 2.0)
    top  = top  + oy * (max_top  / 2.0)

    # Final clamp to valid range
    left = max(0, min(left, max_left))
    top  = max(0, min(top,  max_top))

    return img.crop((int(round(left)), int(round(top)), int(round(left + side)), int(round(top + side))))

def to_data_uri_cropped(path: Path, target_px: int, offset_x: float = 0.0, offset_y: float = 0.0) -> str | None:
    """
    Open image, crop square with per-member offsets, resize to target_px,
    and return base64 data URI (JPEG).
    """
    try:
        img = Image.open(path)
        img = center_square_crop_with_offset(img, offset_x=offset_x, offset_y=offset_y)
        img = img.resize((target_px, target_px), Image.Resampling.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85, optimize=True, progressive=True)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"
    except Exception:
        return None

def img_or_placeholder(path: Path, name: str, offset_x: float = 0.0, offset_y: float = 0.0) -> str:
    data_uri = to_data_uri_cropped(path, encode_px, offset_x=offset_x, offset_y=offset_y)
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
            ox = float(m.get("offset_x", 0.0))
            oy = float(m.get("offset_y", 0.0))
            with c:
                st.markdown(
                    f"""
                    <div class="team-wrap">
                        {img_or_placeholder(photo_path, m["name"], offset_x=ox, offset_y=oy)}
                        <div class="team-name">{m['name']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            idx += 1

render_team(team, cols_per_row=4)
