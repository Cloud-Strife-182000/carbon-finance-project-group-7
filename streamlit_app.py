# Carbon Finance Dashboard v1.5.0 — No Score Filter + ESG Components Analysis
# Filters (Year, Sector, Min ESG Score) + KPIs + Tabs (ESG Overview | Team)
# Green-themed charts + fullscreen-stable pie + ESG component bar chart & legend

import io
import math
from pathlib import Path
import pandas as pd
import streamlit as st
import altair as alt
import numpy as np
from PIL import Image, ImageOps, ImageDraw

# ---------------------------------------------------------
# PAGE SETUP
# ---------------------------------------------------------
# Page setup
st.set_page_config(
    page_title="ESG Disclosures and Carbon Finance Flows in Indian Capital Markets",
    layout="wide"
)

# Title
st.title("ESG Disclosures and Carbon Finance Flows in Indian Capital Markets")
st.markdown(
    """
    <h3 style='text-align: left; color: #333; margin-top: -5px;'>
        Carbon Finance Term Project | Group-7 | CaF-B
    </h3>
    """,
    unsafe_allow_html=True
)
st.divider()

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
# YEAR PARSING
# ---------------------------------------------------------
if "Last Updated On" not in df.columns:
    st.error("Column 'Last Updated On' not found in dataset.")
    st.stop()

df["Last Updated On"] = pd.to_datetime(df["Last Updated On"], errors="coerce")
df["Year"] = df["Last Updated On"].dt.year
available_years = sorted(df["Year"].dropna().unique(), reverse=True)
valid_years = [y for y in available_years if y in [2025, 2024]]
if not valid_years and len(available_years) > 0:
    valid_years = available_years[-2:]

# ---------------------------------------------------------
# DEDUP SAME COMPANY-WITHIN-YEAR (keep latest)
# ---------------------------------------------------------
if "Company Name" in df.columns and "Last Updated On" in df.columns:
    df["Year_Updated"] = df["Last Updated On"].dt.year
    df = (
        df.sort_values("Last Updated On", ascending=False)
          .drop_duplicates(subset=["Company Name", "Year_Updated"], keep="first")
          .drop(columns=["Year_Updated"])
    )

# ---------------------------------------------------------
# FIXED SCORE COLUMNS
# ---------------------------------------------------------
FIXED_SCORE_COLS = ["ESG Score", "Environment Score", "Social Score", "Governance Score"]
available_scores = [c for c in FIXED_SCORE_COLS if c in df.columns]
if not available_scores:
    st.error("None of the fixed score columns were found.")
    st.stop()

# We no longer let users pick the metric; we analyze by ESG Score
score_col = "ESG Score" if "ESG Score" in df.columns else available_scores[0]

# ---------------------------------------------------------
# SIDEBAR FILTERS (Year, Sector, Min ESG Score)
# ---------------------------------------------------------
st.sidebar.header("ESG Data Filters")

year_choice = st.sidebar.selectbox("Year", options=valid_years, index=0)
df = df[df["Year"] == year_choice].copy()

# Numeric enforcement for ESG Score (used across KPIs/charts/filters)
df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
df = df.dropna(subset=[score_col])

SECTOR_COL = "Sector Classification"
if SECTOR_COL in df.columns:
    sector_values = sorted(df[SECTOR_COL].astype("string").fillna("Unknown").unique().tolist())
    chosen_sectors = st.sidebar.multiselect(
        "Sectors", options=sector_values, default=[],
        help=None,
    )
    # If nothing selected, include all sectors
    if not chosen_sectors:
        chosen_sectors = sector_values
else:
    st.warning(f"Column '{SECTOR_COL}' not found. Sector filter disabled.")
    chosen_sectors = None

score_min = float(df[score_col].min())
score_max = float(df[score_col].max())
lo = math.floor(score_min / 5.0) * 5
hi = math.ceil(score_max / 5.0) * 5
min_score = st.sidebar.slider("Minimum ESG Score", min_value=float(lo), max_value=float(hi), value=float(lo), step=1.0)

# ---------------------------------------------------------
# APPLY FILTERS
# ---------------------------------------------------------
df_f = df.copy()
if chosen_sectors:
    df_f[SECTOR_COL] = df_f[SECTOR_COL].astype("string").fillna("Unknown")
    df_f = df_f[df_f[SECTOR_COL].isin(chosen_sectors)]
df_f = df_f[df_f[score_col] >= min_score]

if len(df_f) == 0:
    st.warning("No rows match the current filters.")
    st.stop()

# ---------------------------------------------------------
# KPIs
# ---------------------------------------------------------

st.markdown("## ESG Data Summary")

n_total = len(df)
n_companies = len(df_f)
avg_score = float(df_f[score_col].mean())
med_score = float(df_f[score_col].median())

k1, k2, k3 = st.columns(3)
with k1:
    st.metric("Companies Analyzed", f"{n_companies:,} / {n_total:,}")
with k2:
    st.metric("Average ESG Score", f"{avg_score:.1f}")
with k3:
    st.metric("Median ESG Score", f"{med_score:.1f}")

# Year + sectors summary
if chosen_sectors:
    n_sectors_selected = len(chosen_sectors)
    total_sectors = len(df["Sector Classification"].astype("string").fillna("Unknown").unique())
else:
    n_sectors_selected = total_sectors = len(df["Sector Classification"].astype("string").fillna("Unknown").unique())

st.markdown(
    f"""
    <div style="margin-top:-10px; margin-bottom:20px; font-size:0.95rem; color:gray;">
        <b>Year:</b> {year_choice} &nbsp;&nbsp;|&nbsp;&nbsp;
        <b>Sectors Chosen:</b> {n_sectors_selected} / {total_sectors}
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------
tab_overview, tab_sector, tab_team = st.tabs(["ESG Overview", "Sector Analysis", "Team"])

with tab_overview:
    st.markdown("## ESG Performance Dashboard")

    # ---------------- CHARTS (Histogram + Pie) ----------------
    left, right = st.columns(2)

    # Histogram (ESG Score) with green gradient
    with left:
        st.markdown(f"##### {score_col} Distribution (Histogram)")
        score_hist = (
            alt.Chart(df_f)
            .mark_bar()
            .encode(
                x=alt.X(f"{score_col}:Q", bin=alt.Bin(maxbins=20), title=score_col),
                y=alt.Y("count():Q", title="Count"),
                color=alt.Color("count():Q", scale=alt.Scale(scheme="greens"), legend=None),
                tooltip=[
                    alt.Tooltip(f"{score_col}:Q", title="ESG Score (binned)", bin=alt.Bin(maxbins=20)),
                    alt.Tooltip("count():Q", title="Count"),
                ],
            )
            .properties(height=380)
            .configure_view(strokeWidth=0)
        )
        st.altair_chart(score_hist, use_container_width=True)

    # Pie (ESG Rating Distribution via Category) — aligned tooltips/labels
    with right:
        st.markdown("##### ESG Rating Distribution")

        if "Category" not in df_f.columns:
            st.warning("Column 'Category' not found — cannot build rating distribution.")
        else:
            # Build counts
            # Build counts (Category-based)
            rating_counts = (
                df_f["Category"]
                .astype("string").fillna("Unknown")
                .value_counts(dropna=False)
                .reset_index()
            )
            rating_counts.columns = ["rating", "n"]

            # Canonical order: sort by count desc (or keep your preferred order)
            rating_counts = rating_counts.sort_values("n", ascending=False).reset_index(drop=True)
            rating_order = rating_counts["rating"].tolist()
            rating_counts["sort_key"] = rating_counts.index.astype(int)

            # Shares + labels
            total_n = float(rating_counts["n"].sum())
            rating_counts["pct"] = (rating_counts["n"] / total_n * 100).round(1)
            rating_counts["label"] = rating_counts["pct"].astype(int).astype(str) + "%"

            # ---------- NEW: compute mid-angles for conditional alignment ----------
            rating_counts["_cum"] = rating_counts["n"].cumsum()
            rating_counts["_start"] = rating_counts["_cum"] - rating_counts["n"]
            rating_counts["angle"] = ((rating_counts["_start"] + rating_counts["_cum"]) / 2.0) / total_n * 2 * np.pi
            rating_counts["cosA"] = np.cos(rating_counts["angle"])
            rating_counts["absCos"] = rating_counts["cosA"].abs()

            # Split for inside/outside labels
            MIN_LABEL_PCT = 8
            inside  = rating_counts[rating_counts["pct"] >= MIN_LABEL_PCT]
            outside = rating_counts[rating_counts["pct"] <  MIN_LABEL_PCT]

            # Base + pie (unchanged)
            green_palette = ['#00441b', '#006d2c', '#238b45', '#41ae76', '#66c2a4', '#99d8c9', '#ccece6']
            palette = green_palette[: len(rating_counts)]
            base = alt.Chart(rating_counts).properties(width=400, height=400)

            pie = (
                base
                .mark_arc(outerRadius=150, innerRadius=60, cornerRadius=5)
                .encode(
                    theta=alt.Theta("n:Q", stack=True, title=""),
                    order=alt.Order("sort_key:Q", sort="ascending"),
                    color=alt.Color(
                        "rating:N",
                        title="Rating",
                        scale=alt.Scale(domain=rating_order, range=palette),
                        sort=None,
                    ),
                    tooltip=[
                        alt.Tooltip("rating:N", title="Category"),
                        alt.Tooltip("n:Q", title="Count"),
                        alt.Tooltip("pct:Q", title="Share (%)"),
                    ],
                )
            )

            labels_inside = (
                alt.Chart(inside)
                .mark_text(size=13, color="white", fontWeight="bold")
                .encode(
                    theta=alt.Theta("n:Q", stack=True),
                    order=alt.Order("sort_key:Q", sort="ascending"),
                    text="label:N",
                    radius=alt.value(112),
                    tooltip=[
                        alt.Tooltip("rating:N", title="Category"),
                        alt.Tooltip("n:Q", title="Count"),
                        alt.Tooltip("pct:Q", title="Share (%)"),
                    ],
                )
            )

            # --- Outside labels split into 3 layers for proper alignment ---
            outside_center = outside[outside["absCos"] < 0.1]
            outside_right  = outside[(outside["absCos"] >= 0.1) & (outside["cosA"] > 0)]
            outside_left   = outside[(outside["absCos"] >= 0.1) & (outside["cosA"] <= 0)]

            labels_outside_center = (
                alt.Chart(outside_center)
                .mark_text(size=12, color="black", fontWeight="bold", align="center", dx=0)
                .encode(
                    theta=alt.Theta("n:Q", stack=True),
                    order=alt.Order("sort_key:Q", sort="ascending"),
                    text="label:N",
                    radius=alt.value(175),
                    tooltip=[
                        alt.Tooltip("rating:N", title="Category"),
                        alt.Tooltip("n:Q", title="Count"),
                        alt.Tooltip("pct:Q", title="Share (%)"),
                    ],
                )
            )

            labels_outside_right = (
                alt.Chart(outside_right)
                .mark_text(size=12, color="black", fontWeight="bold", align="left", dx=-75)
                .encode(
                    theta=alt.Theta("n:Q", stack=True),
                    order=alt.Order("sort_key:Q", sort="ascending"),
                    text="label:N",
                    radius=alt.value(175),
                    tooltip=[
                        alt.Tooltip("rating:N", title="Category"),
                        alt.Tooltip("n:Q", title="Count"),
                        alt.Tooltip("pct:Q", title="Share (%)"),
                    ],
                )
            )

            labels_outside_left = (
                alt.Chart(outside_left)
                .mark_text(size=12, color="black", fontWeight="bold", align="right", dx=75)
                .encode(
                    theta=alt.Theta("n:Q", stack=True),
                    order=alt.Order("sort_key:Q", sort="ascending"),
                    text="label:N",
                    radius=alt.value(175),
                    tooltip=[
                        alt.Tooltip("rating:N", title="Category"),
                        alt.Tooltip("n:Q", title="Count"),
                        alt.Tooltip("pct:Q", title="Share (%)"),
                    ],
                )
            )

            # Layer everything (keep your existing config)
            final_chart = alt.LayerChart(
                layer=[pie, labels_inside, labels_outside_center, labels_outside_right, labels_outside_left],
                config={
                    "view": {"stroke": "transparent"},
                    "legend": {"orient": "right", "titleFontSize": 13, "labelFontSize": 12},
                },
            ).properties(width=400, height=400)

            st.altair_chart(final_chart, use_container_width=True)

            
    # ---------------- ESG COMPONENTS (no section header) ----------------
    st.markdown("##### Average ESG Component Scores")

    # Check components availability
    comp_cols = [c for c in ["Environment Score", "Social Score", "Governance Score"] if c in df_f.columns]
    if len(comp_cols) == 0:
        st.info("Environment, Social, and Governance component columns not found.")
    else:
        # Compute averages for filtered data
        comp_avg = (
            df_f[comp_cols]
            .mean(numeric_only=True)
            .rename(index={
                "Environment Score": "Environment",
                "Social Score": "Social",
                "Governance Score": "Governance",
            })
            .reset_index()
        )
        comp_avg.columns = ["Component", "Average Score"]

        # Enforce component order
        comp_avg["Component"] = pd.Categorical(
            comp_avg["Component"],
            categories=["Environment", "Social", "Governance"],
            ordered=True,
        )
        comp_avg = comp_avg.sort_values("Component")

        # Single chart with integrated legend (no separate c2 column)
        y_scale = alt.Scale(domain=[0, 100])

        bar_chart = (
            alt.Chart(comp_avg)
            .mark_bar()
            .encode(
                x=alt.X("Component:N", title="", sort=["Environment", "Social", "Governance"]),
                y=alt.Y("Average Score:Q", title="Average Score", scale=y_scale),
                color=alt.Color(
                    "Average Score:Q",
                    scale=alt.Scale(scheme="greens", domain=[0, 100]),
                    legend=alt.Legend(
                        title="Average Score",
                        orient="right",
                        gradientLength=180,
                        gradientThickness=16,
                        labelFontSize=12,
                        titleFontSize=13,
                    ),
                ),
                tooltip=[
                    alt.Tooltip("Component:N"),
                    alt.Tooltip("Average Score:Q", format=".1f"),
                ],
            )
            .properties(height=320)
        )

        # Value labels above bars
        labels = (
            alt.Chart(comp_avg)
            .mark_text(align="center", baseline="bottom", dy=-5, fontWeight="bold", color="#222")
            .encode(
                x=alt.X("Component:N", sort=["Environment", "Social", "Governance"]),
                y=alt.Y("Average Score:Q", scale=y_scale),
                text=alt.Text("Average Score:Q", format=".1f"),
            )
        )

        st.altair_chart((bar_chart + labels).configure_view(strokeWidth=0), use_container_width=True)

    # ---------------- TOP PERFORMERS TABLE ----------------
    metric_label = "ESG"  # fixed metric label since we removed metric filter
    table_title = f"Top 10 {metric_label} Performers in Selected Sectors ({int(year_choice)})"

    name_col = "Company Name"
    if name_col not in df_f.columns:
        st.warning("Column 'Company Name' not found in dataset.")
    else:
        display_cols = [name_col, score_col, "Category"]
        col_renames = {
            name_col: "Company Name",
            score_col: f"{metric_label} Score",
            "Category": "Rating",
        }
        if SECTOR_COL in df_f.columns:
            display_cols.insert(1, SECTOR_COL)
            col_renames[SECTOR_COL] = "Sector"

        top10 = (
            df_f.sort_values(score_col, ascending=False)
                [display_cols]
                .head(10)
                .rename(columns=col_renames)
        )
        top10[f"{metric_label} Score"] = top10[f"{metric_label} Score"].round(1)

        st.markdown(f"#### {table_title}")
        st.dataframe(top10, use_container_width=True, hide_index=True)

    # ---------------- DATA PREVIEW ----------------
    with st.expander("Preview Filtered Data (first 100 rows)"):
        st.dataframe(df_f.head(100), use_container_width=True, hide_index=True)


# ---------------------------------------------------------
# SECTOR ANALYSIS TAB
# ---------------------------------------------------------


with tab_sector:
    st.markdown("## Sector-wise Analysis")

    SECTOR_COL = "Sector Classification"
    if SECTOR_COL not in df_f.columns:
        st.info(f"Column '{SECTOR_COL}' not found. Sector-wise analysis is unavailable.")
    else:
        # -------- Component-wise heatmap by sector (legend integrated) --------
        st.markdown("##### Component-wise ESG Scores by Sector")

        comp_cols = ["Environment Score", "Social Score", "Governance Score"]
        valid_comps = [c for c in comp_cols if c in df_f.columns]
        if not valid_comps:
            st.info("Environment, Social, and Governance component columns not found.")
        else:
            comp_sector = (
                df_f.groupby(SECTOR_COL)[valid_comps]
                .mean(numeric_only=True)
                .reset_index()
                .melt(id_vars=SECTOR_COL, var_name="Component", value_name="Average Score")
                .rename(columns={SECTOR_COL: "Sector"})
            )

            # Clean labels + enforce column order
            comp_sector["Component"] = comp_sector["Component"].replace({
                "Environment Score": "Environment",
                "Social Score": "Social",
                "Governance Score": "Governance",
            })
            comp_sector["Component"] = pd.Categorical(
                comp_sector["Component"],
                categories=["Environment", "Social", "Governance"],
                ordered=True,
            )

            # Order sectors (alphabetical; change if you prefer a score-based order)
            sector_order = sorted(comp_sector["Sector"].unique().tolist())

            heat_h = max(260, 26 * len(sector_order))

            # Single heatmap with integrated legend on the right
            comp_heatmap = (
                alt.Chart(comp_sector)
                .mark_rect()
                .encode(
                    x=alt.X(
                        "Component:N",
                        title="Component",
                        sort=["Environment", "Social", "Governance"],
                    ),
                    y=alt.Y("Sector:N", sort=sector_order, title="Sector"),
                    color=alt.Color(
                        "Average Score:Q",
                        scale=alt.Scale(domain=[0, 100], scheme="redyellowgreen"),
                        legend=alt.Legend(
                            title="Average Score",
                            orient="right",
                            gradientLength=240,
                            gradientThickness=20,
                            labelFontSize=12,
                            titleFontSize=13,
                        ),
                    ),
                    tooltip=[
                        alt.Tooltip("Sector:N"),
                        alt.Tooltip("Component:N"),
                        alt.Tooltip("Average Score:Q", format=".1f"),
                    ],
                )
                .properties(height=heat_h)
                .configure_view(strokeWidth=0)
            )

            st.altair_chart(comp_heatmap, use_container_width=True)

# ---------------------------------------------------------
# TEAM TAB
# ---------------------------------------------------------

with tab_team:
    st.subheader("Project Team — Group 7 | CaF-B")

    TEAM_DIR = BASE_DIR / "team_photos"
    team = [
        {"name": "Akshat Negi", "file": "p24akshatnegi.JPG", "offset_y": -0.75},
        {"name": "G R Srikanth", "file": "p24srikanth.JPG", "offset_y": -0.75},
        {"name": "Siddharth Kumar Pandey", "file": "p24siddharth.JPG", "offset_y": -0.75},
        {"name": "Vineet Ranjan Maitrey", "file": "p24vineet.jpg"},
    ]

    if not TEAM_DIR.exists():
        TEAM_DIR.mkdir(parents=True, exist_ok=True)
        st.info(f"Created {TEAM_DIR.as_posix()}. Add your image files there (e.g., p24siddharth.JPG).")

    size_px = 120

    def make_circular_image(path: Path, size=size_px, offset_y=0.0):
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

# ---------------------------------------------------------
# Data Source Footer
# ---------------------------------------------------------

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    """
    <div style='text-align: center; font-size: 0.9rem; color: gray; margin-top: 10px;'>
        <b>Data Source:</b> 
        <a href='https://india360.esgrisk.ai/Accounts/Ratinglist' target='_blank' style='text-decoration: none; color: #2E8B57;'>
            india360.esgrisk.ai
        </a>
    </div>
    """,
    unsafe_allow_html=True
)
