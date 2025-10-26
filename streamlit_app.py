# Carbon Finance Dashboard v1.3.1
# Fixed Score Columns (ESG/Environment/Social/Governance) + Filters + KPIs + Charts + Team (WebP)
# Title: “Carbon Finance Term Project | Group-7”

import io
import math
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
st.caption("ESG Disclosures & Carbon Finance — Score & Rating Distributions (v1.3.1, WebP)")

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "esg_risk_data.csv"

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
if DATA_PATH.exists():
    df = pd.read_csv(DATA_PATH)
else:
    st.error(f"No file found at {DATA_PATH}. Please place it there.")
    st.stop()

# ---------------------------------------------------------
# FIXED SCORE COLUMNS (only these four)
# ---------------------------------------------------------
FIXED_SCORE_COLS = ["ESG Score", "Environment Score", "Social Score", "Governance Score"]
available_scores = [c for c in FIXED_SCORE_COLS if c in df.columns]

if not available_scores:
    st.error("None of the fixed score columns were found: "
             "'ESG Score', 'Environment Score', 'Social Score', 'Governance Score'.")
    st.stop()

st.subheader("Choose Score Metric")
score_col = st.selectbox("Score column to analyze", available_scores)

# numeric enforcement
df = df.copy()
df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
df = df.dropna(subset=[score_col])

# ---------------------------------------------------------
# SECTOR FILTER: explicit 'Sector Classification'
# ---------------------------------------------------------
SECTOR_COL = "Sector Classification"
if SECTOR_COL not in df.columns:
    st.warning(f"Column '{SECTOR_COL}' not found. Sector filter will be hidden.")
    sector_col = None
else:
    sector_col = SECTOR_COL

# ---------------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------------
st.sidebar.header("Filters")

# Sectors multiselect
if sector_col:
    sector_values = sorted(
        df[sector_col].astype("string").fillna("Unknown").unique().tolist(),
        key=lambda x: (x is None, str(x))
    )
    chosen_sectors = st.sidebar.multiselect(
        "Sectors", options=sector_values, default=sector_values,
        help=f"Filtering by '{sector_col}' column",
    )
else:
    chosen_sectors = None

# Minimum score slider (based on selected score_col)
score_min = float(df[score_col].min())
score_max = float(df[score_col].max())
lo = math.floor(score_min / 5.0) * 5
hi = math.ceil(score_max / 5.0) * 5
min_score = st.sidebar.slider(
    "Minimum Score", min_value=float(lo), max_value=float(hi),
    value=float(lo), step=1.0,
)

# ---------------------------------------------------------
# APPLY FILTERS
# ---------------------------------------------------------
df_f = df.copy()
if sector_col and chosen_sectors is not None:
    df_f[sector_col] = df_f[sector_col].astype("string").fillna("Unknown")
    df_f = df_f[df_f[sector_col].isin(chosen_sectors)]
df_f = df_f[df_f[score_col] >= min_score]

if len(df_f) == 0:
    st.warning("No rows match the current filters. Adjust sector selection or minimum score.")
    st.stop()

# ---------------------------------------------------------
# RATING BANDS (from selected score)
# AAA (>=85), AA (75–84), A (65–74), BBB (55–64), BB (<55)
# ---------------------------------------------------------
def map_rating(score):
    if pd.isna(score):
        return "Unknown"
    if score >= 85:
        return "AAA"
    elif score >= 75:
        return "AA"
    elif score >= 65:
        return "A"
    elif score >= 55:
        return "BBB"
    else:
        return "BB"

df_f["ESG_Rating_Band"] = df_f[score_col].apply(map_rating)

# ---------------------------------------------------------
# KPIs
# ---------------------------------------------------------
n_companies = len(df_f)
avg_score = float(df_f[score_col].mean())
med_score = float(df_f[score_col].median())

k1, k2, k3 = st.columns(3)
with k1:
    st.metric("Companies Analyzed", f"{n_companies:,}")
with k2:
    st.metric("Average Score", f"{avg_score:.1f}")
with k3:
    st.metric("Median Score", f"{med_score:.1f}")

st.divider()

# ---------------------------------------------------------
# CHARTS
# ---------------------------------------------------------
left, right = st.columns(2)

with left:
    st.subheader(f"{score_col} Distribution (Histogram)")
    score_hist = (
        alt.Chart(df_f)
        .mark_bar()
        .encode(
            x=alt.X(f"{score_col}:Q", bin=alt.Bin(maxbins=20), title=score_col),
            y=alt.Y("count():Q", title="Count"),
            tooltip=[
                alt.Tooltip(f"{score_col}:Q", title="Score (binned)", bin=alt.Bin(maxbins=20)),
                alt.Tooltip("count():Q", title="Count"),
            ],
        )
        .properties(height=380)
    )
    st.altair_chart(score_hist, use_container_width=True)

with right:
    st.subheader("Rating Distribution (Derived from Selected Score)")
    rating_counts = (
        df_f["ESG_Rating_Band"]
        .astype("string")
        .fillna("Unknown")
        .value_counts(dropna=False)
        .rename_axis("rating")
        .reset_index(name="n")
    )
    order = ["AAA", "AA", "A", "BBB", "BB", "Unknown"]
    present_order = [r for r in order if r in rating_counts["rating"].unique()]
    rating_counts["rating"] = pd.Categorical(
        rating_counts["rating"], categories=present_order, ordered=True
    )
    rating_counts = rating_counts.sort_values("rating")

    pie = (
        alt.Chart(rating_counts)
        .mark_arc()
        .encode(
            theta=alt.Theta("n:Q", title=""),
            color=alt.Color("rating:N", title="Rating", sort=present_order),
            tooltip=[
                alt.Tooltip("rating:N", title="Rating"),
                alt.Tooltip("n:Q", title="Count"),
            ],
        )
        .properties(height=380)
    )
    st.altair_chart(pie, use_container_width=True)

st.divider()

with st.expander("Preview Filtered Data (first 100 rows)"):
    st.dataframe(df_f.head(100), use_container_width=True, hide_index=True)

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