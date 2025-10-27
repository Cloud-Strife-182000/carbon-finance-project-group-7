Here’s a complete **GitHub README.md** draft for your Streamlit-based carbon finance dashboard project. It draws inspiration from the structure and coding conventions in your original `streamlit_app.py` base, adapted for your customized ESG and Carbon Finance use case.

---

# 🌱 ESG Compass | ESG Disclosures and Carbon Finance Flows in Indian Capital Markets

**Carbon Finance Term Project | Group-7 (CaF-B)**
Built with **Streamlit + Altair**, this interactive dashboard explores ESG (Environmental, Social, and Governance) disclosures and carbon finance flows across Indian capital markets.
It allows users to visualize ESG performance metrics, distribution of ESG ratings, and sector-wise sustainability trends for 2024–2025.

---

## 4️⃣ Run the App

You can use it at https://carbon-finance-project-group-7.streamlit.app/

---

## 📊 Overview

The dashboard provides a **dynamic, data-driven interface** that combines ESG data visualization, sectoral filters, and interactive analysis tabs.

### Key Features

* **Year & Sector Filtering**: Toggle between ESG data for 2024 and 2025, filter by sector, and set minimum ESG scores.
* **KPI Cards**: Automatically display overall statistics, such as number of companies analyzed, average and median ESG scores, and selection info (year, sectors chosen).
* **ESG Rating Distribution (Pie Chart)**: Proportional breakdown of companies by ESG category (AAA, AA, etc.) using the `Category` column from data.
* **ESG Score Distribution (Histogram)**: Distribution of ESG or sub-scores (Environmental, Social, or Governance).
* **Average ESG Component Scores (Bar Chart)**: Comparative visualization of mean Environmental, Social, and Governance scores with an inline gradient legend.
* **Sector-wise Analysis (Heatmap)**: Highlights average ESG and component-wise scores across sectors with a red-yellow-green gradient.
* **Team Showcase**: Circular, offset-aligned photos of team members with names rendered from a local `team_photos` folder.

---

## 🧮 Data Source

Data was sourced from [**India360 ESG Risk Platform**](https://india360.esgrisk.ai/Accounts/Ratinglist), which provides ESG ratings and disclosures for Indian companies.

### Example Data Columns

| Column                                                  | Description               |
| ------------------------------------------------------- | ------------------------- |
| `Company Name`                                          | Name of the firm          |
| `Sector Classification`                                 | Industry or sector        |
| `ESG Score`                                             | Overall ESG score (0–100) |
| `Environment Score`, `Social Score`, `Governance Score` | Component scores          |
| `Category`                                              | ESG Rating (AAA–D)        |
| `Last Updated On`                                       | Reporting year            |

---

## 🧠 Dashboard Structure

| Section             | Description                                                         |
| ------------------- | ------------------------------------------------------------------- |
| **Header**          | Displays project title, team name, and data summary                 |
| **Sidebar Filters** | Sector filter and minimum ESG score slider                          |
| **KPIs**            | Companies analyzed, average/median scores, year, and sectors chosen |
| **Tabs**            | `ESG Overview` (main visuals) and `Team Members` (photo grid)       |
| **Visuals**         | Interactive charts using Altair with responsive layouts             |
| **Data Table**      | Top 10 ESG or component performers in selected sectors              |
| **Source Note**     | Mentions ESG data origin (India360 platform)                        |

---

## 🧩 Tech Stack

* **Python 3.10+**
* **Streamlit** – web app framework
* **Altair** – declarative visualization library
* **Pandas / NumPy** – data manipulation
* **Pillow (PIL)** – image cropping & masking for team portraits

---

## 👥 Project Team — Group-7 (CaF-B)

| Member                 |
| ---------------------- |
| Akshat Negi            |
| G R Srikanth           |
| Siddharth Kumar Pandey |
| Vineet Ranjan Maitrey  |

---

## 📈 Example Visuals

1. ESG Score Distribution Histogram
2. ESG Rating Distribution Pie Chart (Category-based)
3. Average ESG Component Scores Bar Chart
4. Sector-wise ESG and Component Heatmaps

---

## 🧾 License

This project is distributed under the **MIT License**.
You’re free to use, modify, and distribute it with attribution.

---
