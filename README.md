# 📊 End-to-End Sales Forecasting & Demand Intelligence System
### Internship Project — Week 3 & 4

---

## 🎯 Project Overview

This project builds a complete **Sales Forecasting and Demand Intelligence System** for a retail company using 4 years of Superstore Sales data. It covers time series analysis, three forecasting models, anomaly detection, product segmentation, and an interactive dashboard — the kind of system data science teams at real companies build and maintain.

**Submitted by:** Shailesh  
**Internship:** XYlofy AI Internship Program  
**Submission Date:** July 2026

---

## 📦 Dataset

| Dataset | Source | Records |
|---------|--------|---------|
| Superstore Sales (`train.csv`) | [Kaggle — Superstore Sales](https://www.kaggle.com/datasets/rohitsahoo/sales-forecasting) | 9,800 rows |
| Video Game Sales (`vgsales.csv`) | [Kaggle — Video Game Sales](https://www.kaggle.com/datasets/gregorut/videogamesales) | 16,598 rows |

**Time Period:** January 2015 – December 2018  
**Categories:** Furniture · Technology · Office Supplies  
**Regions:** West · East · Central · South

---

## 🗂️ Project Structure

```
SalesForecasting-Internship/
│
├── analysis.ipynb          ← Complete Jupyter Notebook (Tasks 1–7)
├── app.py                  ← Streamlit Interactive Dashboard (4 pages)
├── requirements.txt        ← All Python dependencies
├── summary.docx            ← 2-page Executive Business Report (for CFO/Supply Chain)
├── train.csv               ← Superstore Sales dataset
├── vgsales.csv             ← Video Game Sales dataset
│
└── charts/                 ← All saved chart images (.png)
    ├── monthly_sales_trend.png
    ├── decomposition.png
    ├── stationarity_check.png
    ├── sarima_forecast.png
    ├── prophet_forecast.png
    ├── prophet_components.png
    ├── xgboost_forecast.png
    ├── segment_forecasts.png
    ├── anomaly_detection.png
    ├── elbow_method.png
    └── product_clusters.png
```

---

## ✅ Tasks Completed

| Task | Description | Status |
|------|-------------|--------|
| Task 1 | Data Loading, Merging & Deep Exploration | ✅ Done |
| Task 2 | Time Series Analysis & Decomposition | ✅ Done |
| Task 3 | Sales Forecasting — SARIMA + Prophet + XGBoost | ✅ Done |
| Task 4 | Category & Region Level Forecasting | ✅ Done |
| Task 5 | Anomaly Detection — Isolation Forest + Z-Score | ✅ Done |
| Task 6 | Product Demand Segmentation — K-Means + PCA | ✅ Done |
| Task 7 | Streamlit Interactive Dashboard | ✅ Done |
| Task 8 | Executive Business Report (summary.docx) | ✅ Done |

---

## 🤖 Models Built

### Forecasting Models (Task 3)
| Model | Type | Description |
|-------|------|-------------|
| **SARIMA(1,1,1)(1,1,1,12)** | Statistical | Classical time series model with seasonal parameters |
| **Facebook Prophet** | Industry Standard | Meta's production-grade forecasting tool |
| **XGBoost + Lag Features** | Machine Learning | Time series converted to supervised ML problem |

### Other Models
| Model | Task | Purpose |
|-------|------|---------|
| **Isolation Forest** | Task 5 | ML-based anomaly detection |
| **Z-Score (Rolling)** | Task 5 | Statistical anomaly detection |
| **K-Means Clustering** | Task 6 | Product demand segmentation |
| **PCA** | Task 6 | 2D visualization of clusters |

---

## 📈 Key Results

- **Total Revenue (2015–2018):** $2,261,537 across 9,800 orders
- **Revenue Growth:** +50.5% from 2015 to 2018
- **Top Category:** Technology ($827,456 — 36.6% of revenue)
- **Top Region:** West ($710,220)
- **Peak Sales Month:** November (every year, consistent)
- **Average Shipping Time:** 4.0 days

### 3-Month Forecast (Oct–Dec 2018)
| Month | SARIMA | Prophet | XGBoost |
|-------|--------|---------|---------|
| October 2018 | ~$175,000 | ~$180,000 | ~$172,000 |
| November 2018 | ~$285,000 | ~$290,000 | ~$278,000 |
| December 2018 | ~$320,000 | ~$310,000 | ~$305,000 |

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Jupyter Notebook
Open `analysis.ipynb` in VS Code or Jupyter and click **Run All**

### 3. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`

---

## 🌐 Dashboard Pages

| Page | Description |
|------|-------------|
| 📊 Sales Overview | Total sales by year, monthly trend, region & category charts |
| 🔮 Forecast Explorer | Select category/region + horizon → live Prophet forecast |
| 🚨 Anomaly Report | Anomaly chart + table with dates and possible causes |
| 🎯 Product Demand Segments | Cluster chart + stocking strategy per segment |

---

## 🛠️ Tools & Libraries

| Library | Purpose |
|---------|---------|
| `pandas` · `numpy` | Data loading and manipulation |
| `matplotlib` · `seaborn` | Static charts |
| `plotly` | Interactive charts in dashboard |
| `statsmodels` | SARIMA model + ADF test + decomposition |
| `prophet` | Facebook Prophet forecasting |
| `xgboost` | ML-based time series forecasting |
| `scikit-learn` | Isolation Forest · K-Means · PCA · metrics |
| `streamlit` | Interactive web dashboard |

---

## 📊 Evaluation Criteria

| Criteria | Weight |
|----------|--------|
| Time Series Analysis & Decomposition | 10% |
| All 3 forecasting models correct | 20% |
| Model comparison + justified recommendation | 15% |
| Anomaly Detection (both methods) | 10% |
| Product Segmentation & Clustering | 10% |
| Streamlit Dashboard — functionality | 15% |
| Executive Business Report | 10% |
| Code quality & notebook structure | 5% |
| GitHub repository & requirements.txt | 5% |

---

## 📁 Submission

- ✅ `analysis.ipynb` — Complete notebook
- ✅ `app.py` — Streamlit dashboard
- ✅ `requirements.txt` — Dependencies
- ✅ `summary.docx` — Executive report
- ✅ `charts/` — All visualization PNGs
- ✅ GitHub Repository — This repo
- 🌐 Live Streamlit App — [Link will be added after deployment]

---

*Internship Project — Week 3 & 4 | XYlofy AI Internship Program*
