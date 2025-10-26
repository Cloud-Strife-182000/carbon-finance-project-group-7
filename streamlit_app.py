# Carbon Finance Dashboard v1.3.5
# Filters (Year, Sector, Score Metric in sidebar) + KPIs + Tabs (ESG Overview | Team)
# Green-themed charts + smart pie labels; Title: “Carbon Finance Term Project | Group-7”

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
st.caption("ESG Disclosures & Carbon Finance — Score & Rating Distributions (v1.3.5)")

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
# YEAR PARSING (from 'Last Updated On')
# ---------------------------------------------------------
if "Last Updated On" not in df.columns:
    st.error("Column 'Last Updated On' not found in dataset.")
    st.stop()

df["Last Updated On"] = pd.to_datetime(df["Last Updated On"], errors="coerce")
df["Year"] = df["Last Updated On"].dt.year
available_years = sorted(df["Year"].dropna().unique(), reverse=True)
valid_years = [y for y in available_years if y in [2025, 2024]]
if not valid_years and len(available_years) > 0:
    valid_years = available_years[-2:]  # fallback

# ---------------------------------------------------------
# FIXED SCORE COLUMNS (ESG / E / S / G)
# ---------------------------------------------------------
FIXED_SCORE_COLS = ["ESG Score", "Environment Score", "Social Score", "Governance Score"]
available_scores = [c for c in FIXED_SCORE_COLS if c in df.columns]
if not available_scores:
    st.error("None of the fixed score columns were found.")
    st.stop()

# ---------------------------------------------------------
# FILTERS (SIDEBAR) — Year, Sector, Score Metric, Min Score
# ---------------------------------------------------------
st.sidebar.header("Filters")

# Year selector
year_choice = st.sidebar.selectbox(
    "Year",
    options=valid_years,
    index=0 if (valid_years and 2025 in valid_years) else 0
)
df = df[df["Year"] == year_choice].copy()

# Score metric selector (moved to sidebar)
default_idx = available_scores.index("ESG Score") if "ESG Score" in available_scores else 0
score_col = st.sidebar.selectbox("Score Metric", available_scores, index=default_idx)

# Numeric enforcement for selected score (on the selected year)
df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
df = df.dropna(subset=[score_col])

# Sector filter
SECTOR_COL = "Sector Classification"
if SECTOR_COL in df.columns:
    sector_values = sorted(df[SECTOR_COL].astype("string").fillna("Unknown").unique().tolist())
    chosen_sectors = st.sidebar.multiselect(
        "Sectors", options=sector_values, default=sector_values,
        help=f"Filtering by '{SECTOR_COL}' column",
    )
else:
    st.warning(f"Column '{SECTOR_COL}' not found. Sector filter disabled.")
    chosen_sectors = None

# Minimum score slider (based on year+metric-filtered data)
score_min = float(df[score_col].min())
score_max = float(df[score_col].max())
lo = math.floor(score_min / 5.0) * 5
hi = math.ceil(score_max / 5.0) * 5
min_score = st.sidebar.slider("Minimum Score", min_value=float(lo), max_value=float(hi), value=float(lo), step=1.0)

# ---------------------------------------------------------
# APPLY FILTERS
# ---------------------------------------------------------
df_f = df.copy()
if chosen_sectors:
    df_f[SECTOR_COL] = df_f[SECTOR_COL].astype("string").fillna("Unknown")
    df_f = df_f[df_f[SECTOR_COL].isin(chosen_sectors)]
df_f = df_f[df_f[score_col] >= min_score]

if len(df_f) == 0:
    st.warning("No rows match the current filters. Adjust sector selection or minimum score.")
    st.stop()

# ---------------------------------------------------------
# RATING BANDS (derived from selected score)
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

# ---------------------------------------------------------
# TABS (ESG Overview | Team)
# ---------------------------------------------------------
tab_overview, tab_team = st.tabs(["📊 ESG Overview", "👥 Team"])

with tab_overview:
    st.divider()

    # ---------------- CHARTS — GREEN THEME ----------------
left, right = st.columns(2)

# Histogram with green gradient
with left:
    st.subheader(f"{score_col} Distribution (Histogram)")
    score_hist = (
        alt.Chart(df_f)
        .mark_bar()
        .encode(
            x=alt.X(f"{score_col}:Q", bin=alt.Bin(maxbins=20), title=score_col),
            y=alt.Y("count():Q", title="Count"),
            color=alt.Color("count():Q", scale=alt.Scale(scheme="greens"), legend=None),
            tooltip=[
                alt.Tooltip(f"{score_col}:Q", title="Score (binned)", bin=alt.Bin(maxbins=20)),
                alt.Tooltip("count():Q", title="Count"),
            ],
        )
        .properties(height=380)
        .configure_view(strokeWidth=0)
    )
    st.altair_chart(score_hist, use_container_width=True)

# Donut with smart labels (inside for big slices, outside for small)
with right:
    # Dynamic title from selected score metric
    title_prefix = {
        "ESG Score": "ESG",
        "Environment Score": "Environment",
        "Social Score": "Social",
        "Governance Score": "Governance",
    }.get(score_col, "ESG")
    st.subheader(f"{title_prefix} Rating Distribution")

    # Build counts (explicit order + numeric key to pin label geometry)
    rating_counts = (
        df_f["ESG_Rating_Band"]
        .astype("string").fillna("Unknown")
        .value_counts(dropna=False)
        .rename_axis("rating").reset_index(name="n")
    )

    desired_order = ["AAA", "AA", "A", "BBB", "BB", "Unknown"]
    order_index = {lab: i for i, lab in enumerate(desired_order)}
    rating_counts["sort_key"] = rating_counts["rating"].map(order_index).fillna(999).astype(int)
    rating_counts = rating_counts.sort_values("sort_key").reset_index(drop=True)

    # Percent and labels
    total_n = rating_counts["n"].sum()
    rating_counts["pct"] = (rating_counts["n"] / total_n * 100).round(1)
    rating_counts["label"] = rating_counts["pct"].astype(int).astype(str) + "%"

    # Green palette (dark → light), slice to present categories
    green_palette = ['#00441b', '#006d2c', '#238b45', '#41ae76', '#66c2a4', '#99d8c9']
    palette = green_palette[: len(rating_counts)]

    # Use a single base chart object so both layers inherit identical data immediately
    base = alt.Chart(rating_counts).properties(height=400)

    # Donut wedges
    pie = (
        base
        .mark_arc(outerRadius=150, innerRadius=60, cornerRadius=5)
        .encode(
            theta=alt.Theta("n:Q", stack=True, title=""),
            order=alt.Order("sort_key:Q", sort="ascending"),
            color=alt.Color(
                "rating:N",
                title="Rating",
                scale=alt.Scale(domain=rating_counts["rating"].tolist(), range=palette),
                sort=None,
            ),
            tooltip=[
                alt.Tooltip("rating:N", title="Rating"),
                alt.Tooltip("n:Q", title="Count"),
                alt.Tooltip("pct:Q", title="Share (%)"),
            ],
        )
    )

    # Label logic: big slices inside (white), small slices outside (black)
    MIN_LABEL_PCT = 8  # threshold to "call out" tiny wedges
    inside = rating_counts[rating_counts["pct"] >= MIN_LABEL_PCT]
    outside = rating_counts[rating_counts["pct"] < MIN_LABEL_PCT]

    labels_inside = (
        alt.Chart(inside)
        .mark_text(size=13, color="white", fontWeight="bold")
        .encode(
            theta=alt.Theta("n:Q", stack=True),
            order=alt.Order("sort_key:Q", sort="ascending"),
            text="label:N",
            radius=alt.value(112),   # inside wedge
        )
    )

    labels_outside = (
        alt.Chart(outside)
        .mark_text(size=12, color="black", fontWeight="bold")  # call-out text black
        .encode(
            theta=alt.Theta("n:Q", stack=True),
            order=alt.Order("sort_key:Q", sort="ascending"),
            text="label:N",
            radius=alt.value(175),   # just outside wedge
        )
    )

    layered_pie = (pie + labels_inside + labels_outside).configure_view(strokeWidth=0)
    st.altair_chart(layered_pie, use_container_width=True)

st.divider()
with st.expander("Preview Filtered Data (first 100 rows)"):
    st.dataframe(df_f.head(100), use_container_width=True, hide_index=True)

with tab_team:
    # ---------------------------------------------------------
    # TEAM SECTION (JPEG circular images)
    # ---------------------------------------------------------
    st.subheader("Project Team — Group 7")

    TEAM_DIR = BASE_DIR / "team_photos"

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
            oy = int(offset_y * (h - side) / 2)
            top = (h - side) // 2 + oy
            top = max(0, min(top, h - side))
            img = img.crop(((w - side) // 2, top, (w - side) // 2 + side, top + side))
            img = img.resize((size, size), Image.Resampling.LANCZOS)

            mask = Image.new("L", (size, size), 0)
            draw = ImageDraw.Draw(mask)
            draw.ellipse((0, 0, size, size), fill=255)
            img.putalpha(mask)

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
