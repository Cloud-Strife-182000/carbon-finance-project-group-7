# Carbon Finance Dashboard v1.1.1
# ESG Score Histogram + Derived Rating Pie + Team (WebP circular)
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
st.caption("ESG Disclosures & Carbon Finance — Score & Rating Distributions (v1.1.1, WebP)")

# Base directory (used for data and team photos)
BASE_DIR = Path(__file__).parent

# ---------------------------------------------------------
# DATA LOADING (ESG risk file from data folder)
# ---------------------------------------------------------
st.sidebar.header("Upload ESG Data (optional)")
st.sidebar.write(
    "Upload a CSV that contains an ESG score column (numeric). "
    "If not provided, the app loads 'data/esg_risk_data.csv' by default."
)

uploaded = st.sidebar.file_uploader("CSV file", type=["csv"])
DATA_PATH = BASE_DIR / "data" / "esg_risk_data.csv"

if uploaded:
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read the uploaded CSV: {e}")
        st.stop()
else:
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
    else:
        st.error(f"No file found at {DATA_PATH}. Please upload a CSV or place it in that folder.")
        st.stop()

# ---------------------------------------------------------
# COLUMN SELECTION (choose ESG score; rating is derived)
# ---------------------------------------------------------
st.subheader("Select ESG Score Column")

# heuristic candidates
score_candidates = [c for c in df.columns if c.lower() in {"esg_score", "score", "esg"}]
score_candidates += [c for c in df.select_dtypes("number").columns if c not in score_candidates]

score_col = st.selectbox("ESG score column (numeric)", score_candidates or list(df.columns))

# type enforcement for score
df = df.copy()
df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
df = df.dropna(subset=[score_col])

# ---------------------------------------------------------
# DERIVE RATING BANDS (your criteria)
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

df["ESG_Rating_Band"] = df[score_col].apply(map_rating)

# ---------------------------------------------------------
# VISUALS
# ---------------------------------------------------------
left, right = st.columns(2)

with left:
    st.subheader("ESG Score Distribution (Histogram)")
    score_hist = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(f"{score_col}:Q", bin=alt.Bin(maxbins=20), title="ESG Score"),
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
    st.subheader("ESG Rating Distribution (Derived from Score)")

    # Build counts with guaranteed column names: ['rating', 'n']
    rating_counts = (
        df["ESG_Rating_Band"]
        .astype("string")
        .fillna("Unknown")
        .value_counts(dropna=False)   # Series indexed by rating
        .rename_axis("rating")        # index name -> 'rating'
        .reset_index(name="n")        # counts in column 'n'
    )

    # Explicit order (only keep those present; Unknown last if present)
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

with st.expander("Preview Data (first 100 rows)"):
    st.dataframe(df.head(100), use_container_width=True, hide_index=True)

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