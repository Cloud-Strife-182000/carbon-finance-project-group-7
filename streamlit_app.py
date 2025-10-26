# Carbon Finance Dashboard v0.7.1
# One Graph + Circular Team Photos (JPEG compressed)
# Title: “Carbon Finance Term Project | Group-7”

import io
from pathlib import Path
import pandas as pd
import streamlit as st
import altair as alt
from PIL import Image, ImageOps, ImageDraw

# ---------------------------------------------------------
# PAGE SETUP
# ---------------------------------------------------------
st.set_page_config(page_title="Carbon Finance Term Project | Group-7", layout="wide")

st.title("Carbon Finance Term Project | Group-7")
st.caption("ESG Disclosures & Carbon Finance — Minimal Dashboard (v0.7.1)")

# ---------------------------------------------------------
# DATA
# ---------------------------------------------------------
st.sidebar.header("Upload (optional)")
st.sidebar.write(
    "Provide a CSV with two columns: `year`, `green_bonds_inr_cr`. "
    "If not provided, we use a small demo dataset."
)

uploaded = st.sidebar.file_uploader("CSV file", type=["csv"])

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

# ---------------------------------------------------------
# CHART
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# DATA PREVIEW
# ---------------------------------------------------------
st.write("Data preview")
st.dataframe(df, width='stretch', hide_index=True)

st.divider()

# ---------------------------------------------------------
# TEAM SECTION (JPEG circular images)
# ---------------------------------------------------------
st.subheader("Project Team — Group 7")

BASE_DIR = Path(__file__).parent
TEAM_DIR = BASE_DIR / "team_photos"

# Updated team list
team = [
    {"name": "Akshat Negi",             "file": "p24akshatnegi.JPG", "offset_y": -0.75},
    {"name": "G R Srikanth",            "file": "p24srikanth.JPG",   "offset_y": -0.75},
    {"name": "Siddharth Kumar Pandey",  "file": "p24siddharth.JPG",  "offset_y": -0.75},
    {"name": "Vineet Ranjan Maitrey",   "file": "p24vineet.jpg"},
]

if not TEAM_DIR.exists():
    TEAM_DIR.mkdir(parents=True, exist_ok=True)
    st.info(f"Created {TEAM_DIR.as_posix()}. Add your image files there (e.g., p24siddharth.JPG).")

size_px = 120  # final circle diameter

def make_circular_image(path: Path, size=size_px, offset_y=0.0):
    """Crop, resize, and mask image to a circular shape."""
    try:
        img = Image.open(path)
        img = ImageOps.exif_transpose(img)
        if img.mode != "RGB":
            img = img.convert("RGB")

        w, h = img.size
        side = min(w, h)
        # Offset adjustment: move crop up or down
        oy = int(offset_y * (h - side) / 2)
        top = (h - side) // 2 + oy
        top = max(0, min(top, h - side))
        img = img.crop(((w - side) // 2, top, (w - side) // 2 + side, top + side))
        img = img.resize((size, size), Image.Resampling.LANCZOS)

        # Make circular mask
        mask = Image.new("L", (size, size), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0, size, size), fill=255)
        img.putalpha(mask)

        # Convert to displayable PNG bytes
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return None

cols_per_row = 4
rows = (len(team) + cols_per_row - 1) // cols_per_row
idx = 0

for _ in range(rows):
    cols = st.columns(cols_per_row)
    for c in cols:
        if idx >= len(team):
            c.empty()
            continue
        member = team[idx]
        img_bytes = make_circular_image(
            TEAM_DIR / member["file"],
            size=size_px,
            offset_y=member.get("offset_y", 0.0)
        )
        with c:
            if img_bytes:
                st.image(img_bytes, width=size_px)
            else:
                st.markdown(f"**{member['name']}** (Image missing)")
            st.markdown(f"**{member['name']}**", unsafe_allow_html=True)
        idx += 1