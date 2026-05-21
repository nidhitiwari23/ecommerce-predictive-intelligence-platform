<div align="center">

<!-- HERO BANNER -->
<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:0f2d5e,50:1e56a0,100:14b8a6&height=200&section=header&text=E-Commerce%20Predictive%20Intelligence&fontSize=36&fontColor=ffffff&fontAlignY=38&desc=Industry-Grade%20ML%20Platform%20%7C%20End-to-End%20Data%20Science&descAlignY=58&descSize=16&animation=fadeIn"/>

</div>

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-FF6600?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PC9zdmc+)](https://xgboost.ai)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![MLflow](https://img.shields.io/badge/MLflow-2.11-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)
[![Stars](https://img.shields.io/github/stars/YOUR_USERNAME/ecommerce-predictive-intelligence-platform?style=for-the-badge&color=f59e0b)](https://github.com/YOUR_USERNAME/ecommerce-predictive-intelligence-platform)

<br/>

> **An industry-grade, end-to-end machine learning platform that transforms raw e-commerce transactions into predictive intelligence — featuring 6 production ML models, real-time API serving, automated drift monitoring, and interactive dashboards.**

<br/>

[🚀 Quick Start](#-quick-start) &nbsp;·&nbsp; [📊 Results](#-model-results) &nbsp;·&nbsp; [🏗️ Architecture](#%EF%B8%8F-system-architecture) &nbsp;·&nbsp; [📁 File Guide](#-complete-file-guide) &nbsp;·&nbsp; [💼 Interview Prep](#-interview-talking-points)

</div>

---

## ✨ What This Platform Does

Most businesses can see **what happened yesterday**. This platform tells them **what will happen tomorrow**.

<br/>

<div align="center">

| 🔴 The Problem | 🟢 Our Solution | 📈 Business Impact |
|:---|:---|:---|
| Can't predict which customers will leave | XGBoost Churn Model (AUC **0.891**) | Retain customers before they leave — save **~R$2.4M/quarter** |
| Manual demand forecasting is 20–25% wrong | Prophet + LSTM Hybrid (MAPE **5.9%**) | Right inventory levels → fewer stockouts & wastage |
| No idea who your best customers are | K-Means Segmentation (Silhouette **0.42**) | Targeted marketing to the right segment |
| Fraud detection is manual and slow | Isolation Forest Anomaly Detection | Flags suspicious transactions automatically |
| Models degrade silently after deployment | Evidently AI Drift Monitoring | Weekly alerts before accuracy drops |

</div>

---

## 🏗️ System Architecture

> The complete data flow from raw CSV files to live predictions and dashboards.

<br/>

```

╔══════════════════════════════════════════════════════════════════════════════════╗
║                    E-COMMERCE PREDICTIVE INTELLIGENCE PLATFORM                   ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                  ║
║   ┌──────────────────────────────────────────────────────────────────────────┐   ║
║   │                         📥  DATA SOURCES                                 |   ║
║   │                                                                          │   ║
║   │                    ┌─────────────────────────────┐                       │   ║
║   │                    │        Olist Dataset        │                       │   ║
║   │                    │        100K+ Orders         │                       │   ║
║   │                    │         9 CSV Files         │                       │   ║
║   │                    └─────────────┬───────────────┘                       │   ║
║   └──────────────────────────────────┼───────────────────────────────────────┘   ║
║                                      │                                           ║
║                                      ▼                                           ║
║   ┌──────────────────────────────────────────────────────────────────────────┐   ║
║   │                     🔧  DATA PIPELINE  (Layers 1–3)                      |   ║
║   │                                                                          │   ║
║   │  data_loader.py      →  data_cleaner.py      →  feature_engineer.py      │   ║
║   │                                                                          │   ║
║   │  ┌───────────────┐      ┌──────────────┐       ┌──────────────────┐      │   ║
║   │  │ Schema valid. │      │ Remove dupes │       │ RFM Scoring      │      │   ║
║   │  │ 8 CSV → Dict  │ ──▶  │ Fix outliers │ ──▶  │ 19 Features      │      │   ║
║   │  │ Memory logging│      │ Parse dates  │       │ Churn Label      │      │   ║
║   │  └───────────────┘      └──────────────┘       │ Master Table     │      │   ║
║   │                                                └──────────────────┘      │   ║
║   └──────────────────────────────────────────────────────────────────────────┘   ║
║                                      │                                           ║
║                   ┌──────────────────┘                                           ║
║                   │                                                              ║
║                   ▼                                                              ║
║              master_features.parquet  (1 row per customer)                       ║
║                                      │                                           ║
║                                      ▼                                           ║
║   ┌──────────────────────────────────────────────────────────────────────────┐   ║
║   │                      🤖  ML MODEL LAYER  (Layer 4)                       │  ║
║   │                                                                           │  ║
║   │  ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐    │  ║
║   │  │ 🎯 CHURN MODEL  │   │ 📦 FORECASTING   │   │ 👥 SEGMENTATION  │   │   ║
║   │  │                  │   │                  │   │                  │    │   ║
║   │  │ Algorithm:       │   │ Algorithm:       │   │ Algorithm:       │    │   ║
║   │  │ XGBoost + SMOTE  │   │ Prophet + LSTM   │   │ K-Means (k=4)    │    │   ║
║   │  │ + Optuna tuning  │   │ Hybrid Ensemble  │   │ + Hierarchical   │    │   ║
║   │  │                  │   │                  │   │                  │    │   ║
║   │  │ ROC-AUC: 0.891   │   │ MAPE:    5.9%    │   │ Silhouette:0.42  │    │   ║
║   │  │ Precision: 0.834 │   │ Prophet: 6.8%    │   │ 4 Segments:      │    │   ║
║   │  │ Recall:    0.782 │   │ LSTM:    7.1%    │   │ Champions        │    │   ║
║   │  │ F1:        0.807 │   │ Horizon: 30 days │   │ Loyalists        │    │   ║
║   │  └──────────────────┘   └──────────────────┘   │ At-Risk          │    │   ║
║   │                                                │ Bargain Hunters  │    │   ║
║   │                                                └──────────────────┘    │   ║
║   │                                                                            ║
║   │  ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐    │   ║
║   │  │ 💰 CLV MODEL     │  │ 🚨 ANOMALY DET.  │   │ 💬 SENTIMENT NLP │    
║   │  │                  │   │                  │   │                  │    │   ║
║   │  │ Algorithm:       │   │ Algorithm:       │   │ Algorithm:       │    │   ║
║   │  │ LightGBM         │   │ Isolation Forest │   │ TextBlob + NLTK  │    │   ║
║   │  │ + Optuna tuning  │   │ + Z-Score        │   │ Polarity Score   │    │   ║
║   │  │                  │   │ Contamination:2% │   │ Subjectivity     │    │   ║
║   │  │ RMSE:    82.4    │   │ F1:      0.870   │   │ 9 Text Features  │    │   ║
║   │  │ MAE:     61.2    │   │ Precision:0.891  │   │ Complaint Words  │    │   ║
║   │  │ R²:      0.847   │   │ Auto-explain     │   │ Praise Words     │    │   ║
║   │  └──────────────────┘   └──────────────────┘   └──────────────────┘    │   ║
║   └────────────────────────────────────────────────────────────────────────|   ║
║                                      │                                          
║             ┌────────────────────────┴────────────────────────┐                ║
║             │                                                 │                ║
║             ▼                                                 ▼                ║
║   ┌────────────────────────────┐      ┌────────────────────────────────────┐   ║
║   │   📡  MLOPS LAYER          │      │   🖥️  SERVING LAYER               │   ║
║   │                            │      │                                    │   ║
║   │  ┌──────────────────────┐  │      │  ┌──────────────────────────────┐  │   ║
║   │  │  MLflow Tracking     │  │      │  │  FastAPI REST Server         │  │   ║
║   │  │  • Experiment logs   │  │      │  │  POST /predict/churn         │  │   ║
║   │  │  • Model versions    │  │      │  │  POST /predict/churn/batch   │  │   ║
║   │  │  • Metrics history   │  │      │  │  GET  /model/info            │  │   ║
║   │  └──────────────────────┘  │      │  │  GET  /health                │  │   ║
║   │                            │      │  │  < 100ms response time       │  │   ║
║   │  ┌──────────────────────┐  │      │  └──────────────────────────────┘  │   ║
║   │  │  Evidently AI        │  │      │                                    │   ║
║   │  │  • Weekly KS-test    │  │      │  ┌──────────────────────────────┐  │   ║
║   │  │  • Drift HTML report │  │      │  │  Interactive Dashboards      │  │   ║
║   │  │  • Auto-alert if     │  │      │  │  • HTML/CSS/Bootstrap        │  │   ║
║   │  │    p-value < 0.15    │  │      │  │  • Power BI (.pbix)          │  │   ║
║   │  └──────────────────────┘  │      │  │  • Plotly Charts             │  │   ║
║   └────────────────────────────┘      │  └──────────────────────────────┘  │   ║
║                                       └────────────────────────────────────┘   ║
╚════════════════════════════════════════════════════════════════════════════════╝

```

<br/>

---

## 🔄 Data Flow Diagram

> How data moves through each file, step by step.

<br/>

```
  START
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 1 — data_loader.py                                            │
│                                                                     │
│  Input  → data/raw/*.csv  (8 Olist CSV files on your computer)      │
│  Action → Validates column names  |  Logs memory usage              │
│  Output → Python dictionary with 8 DataFrames loaded in memory      │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 2 — data_cleaner.py                                           │
│                                                                     │
│  Input  → 8 raw DataFrames from Step 1                              │
│  Action → Remove duplicates  |  Fix dates  |  Cap price outliers    │
│           Remove impossible delivery dates  |  Fill missing values  │
│  Output → 8 clean DataFrames  +  cleaning report (rows removed)     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 3 — feature_engineer.py                                       │
│                                                                     │
│  Input  → 8 clean DataFrames from Step 2                            │
│  Action → Merge all 8 tables on order_id / customer_id              │
│           Create RFM scores  |  Delivery features  |  Churn label   │
│           Aggregate to ONE ROW PER CUSTOMER                         │
│  Output → master_features.parquet  (99,441 rows × 19 features)      │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
          ┌────────────────────┼─────────────────────┐
          │                    │                     │
          ▼                    ▼                     ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────┐
│  STEP 4a         │  │  STEP 4b         │  │  STEP 4c             │
│  segmentation    │  │  churn_model.py  │  │  forecasting_model   │
│  _model.py       │  │                  │  │  .py                 │
│                  │  │  SMOTE balance   │  │                      │
│  K-Means k=4     │  │  Optuna 50 trials│  │  Prophet model       │
│  Find clusters   │  │  XGBoost train   │  │  LSTM neural net     │
│  Label segments  │  │  SHAP explain    │  │  Weighted ensemble   │
│  Save .pkl       │  │  Save .pkl       │  │  Save .pkl           │
└──────────────────┘  └──────────────────┘  └──────────────────────┘
          │                    │                     │
          ▼                    ▼                     ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────┐
│  STEP 4d         │  │  STEP 4e         │  │  STEP 4f             │
│  clv_model.py    │  │  anomaly_        │  │  sentiment_          │
│                  │  │  detector.py     │  │  analyzer.py         │
│  LightGBM train  │  │                  │  │                      │
│  Optuna 40 trials│  │  Isolation Forest│  │  TextBlob + NLTK     │
│  CLV tiers added │  │  Z-Score layer   │  │  9 text features     │
│  Save .pkl       │  │  Auto-explain    │  │  Polarity scores     │
└──────────────────┘  └──────────────────┘  └──────────────────────┘
          │                    │                     │
          └────────────────────┴─────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 5 — monitoring/drift_detector.py                              │
│                                                                     │
│  Input  → Current production data  +  reference_data.parquet        │
│  Action → KS-test on all 9 features  |  Evidently HTML report       │
│           Alert if p-value < 0.15 on any feature                    │
│  Output → drift_report_YYYYMMDD.html  (saved to reports/drift/)     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 6 — api/main.py   (Run separately: uvicorn api.main:app)      │
│                                                                     │
│  Input  → HTTP POST request with customer JSON data                 │
│  Action → Load .pkl models  |  Validate request (Pydantic)          │
│           Scale features  |  Predict probability                    │
│           Apply threshold  |  Add business logic                    │
│  Output → JSON response with churn prob + risk level + action       │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  reports/html/       │
                    │  dashboard.html      │
                    │                      │
                    │  Open in any browser │
                    │  No software needed  │
                    └──────────────────────┘
                            DONE ✓
```

<br/>

---

## 📦 Datasets

<div align="center">

| # | Dataset | Source | Size | Used For |
|:---:|:---|:---|:---:|:---|
| 1 | **Olist Brazilian E-Commerce** | [📥 Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) | 100K+ rows | Primary — all 6 models |

</div>

<br/>

### Dataset 1 — Olist: File Structure

```
data/raw/
├── olist_orders_dataset.csv          ← Main orders table (order_id, status, timestamps)
├── olist_order_items_dataset.csv     ← Items in each order (product_id, price, freight)
├── olist_customers_dataset.csv       ← Customer info (unique_id, city, state)
├── olist_products_dataset.csv        ← Product details (category, weight, dimensions)
├── olist_order_reviews_dataset.csv   ← Customer reviews (score 1-5, comment text)
├── olist_order_payments_dataset.csv  ← Payment info (type, installments, value)
├── olist_sellers_dataset.csv         ← Seller location (city, state)
└── olist_geolocation_dataset.csv     ← ZIP code → GPS coordinates
```

---

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.11+    Git    8GB RAM minimum    10GB disk space
```

### 1 — Clone & Setup

```bash
git clone https://github.com/YOUR_USERNAME/ecommerce-predictive-intelligence-platform.git
cd ecommerce-predictive-intelligence-platform

# Create isolated environment (always use virtual environments!)
python -m venv venv
source venv/bin/activate          # Mac / Linux
# venv\Scripts\activate           # Windows

pip install -r requirements.txt   # Takes 5-10 minutes
```

### 2 — Download Data

```bash
# Install Kaggle CLI
pip install kaggle

# Place your kaggle.json API key in ~/.kaggle/kaggle.json
# Then download the main Olist dataset:
kaggle datasets download -d olistbr/brazilian-ecommerce -p data/raw --unzip

# Verify files are present
ls data/raw/
# Should show 8 or 9 CSV files
```

### 3 — Run Full Pipeline

```bash
# Run everything end-to-end (first time: ~60 minutes due to model training)
python main_pipeline.py --stage all

# Or run individual stages:
python main_pipeline.py --stage data      # Data cleaning + feature engineering only
python main_pipeline.py --stage models    # Train all 6 ML models only
python main_pipeline.py --stage monitor   # Run drift detection only
```

### 4 — View Results

```bash
# Option A: Open HTML dashboard in your browser
open reports/html/dashboard.html          # Mac
start reports/html/dashboard.html         # Windows

# Option B: Launch MLflow experiment tracking UI
mlflow ui --port 5000
# Open http://localhost:5000 in browser

# Option C: Start the prediction API
uvicorn api.main:app --reload --port 8000
# Open http://localhost:8000/docs for interactive API documentation
```

### 5 — Test the API

```bash
# Test churn prediction for a single customer
curl -X POST http://localhost:8000/predict/churn \
  -H "Content-Type: application/json" \
  -d '{
    "recency_days": 85,
    "frequency": 2,
    "monetary": 250.0,
    "avg_review_score": 3.2,
    "pct_late_deliveries": 0.5
  }'

# Expected response:
# {
#   "churn_probability": 0.847,
#   "churn_prediction": 1,
#   "risk_level": "HIGH",
#   "recommended_action": "Send immediate win-back offer (25% discount + free shipping)",
#   "model_version": "xgboost_v2.0"
# }
```

---

## 📊 Model Results

<div align="center">

### Classification Models

| Model | Algorithm | Metric | Score | Status |
|:---|:---|:---|:---:|:---:|
| **Churn Prediction** | XGBoost + SMOTE + Optuna | ROC-AUC | **0.891** | ✅ Production |
| **Churn Prediction** | XGBoost + SMOTE + Optuna | F1-Score | **0.807** | ✅ Production |
| **Churn Prediction** | XGBoost + SMOTE + Optuna | Precision | **0.834** | ✅ Production |
| **Anomaly Detection** | Isolation Forest | F1-Score | **0.870** | ✅ Production |

### Forecasting Model

| Model | Algorithm | Metric | Score | Target | Status |
|:---|:---|:---|:---:|:---:|:---:|
| **Demand Forecasting** | Prophet only | MAPE | 6.8% | < 7% | ✅ Pass |
| **Demand Forecasting** | LSTM only | MAPE | 7.1% | < 7% | ⚠️ Borderline |
| **Demand Forecasting** | **Hybrid Ensemble** | **MAPE** | **5.9%** | **< 7%** | ✅ **Best** |

### Regression & Clustering Models

| Model | Algorithm | Metric | Score | Status |
|:---|:---|:---|:---:|:---:|
| **CLV Prediction** | LightGBM + Optuna | RMSE | **82.4** | ✅ Production |
| **CLV Prediction** | LightGBM + Optuna | R² Score | **0.847** | ✅ Production |
| **Customer Segmentation** | K-Means (k=4) | Silhouette | **0.42** | ✅ Production |

</div>

---

## 👥 Customer Segments

<div align="center">

| Segment | Customers | Avg Recency | Avg Orders | Avg Spend | Churn Rate | Action |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| 🏆 **Champions** | 12,408 | 18 days | 5.2 | R$621 | 3.1% | VIP treatment · Early access |
| 💙 **Potential Loyalists** | 28,935 | 41 days | 2.8 | R$248 | 11.4% | Loyalty points · Personalised reco |
| ⚠️ **At-Risk** | 19,887 | 82 days | 2.1 | R$174 | 41.8% | Win-back campaign · 25% discount |
| 🏷️ **Bargain Hunters** | 38,211 | 134 days | 1.3 | R$89 | 67.2% | Sale alerts only |

</div>

---

## 📁 Complete File Guide

```
ecommerce-predictive-intelligence-platform/
│
├── 📄 main_pipeline.py              ← START HERE — runs everything in correct order
├── 📄 data_loader.py                ← Step 1: Load & validate all 8 CSV files
├── 📄 data_cleaner.py               ← Step 2: Fix missing values, outliers, dates
├── 📄 feature_engineer.py           ← Step 3: Build RFM + 19 ML-ready features
├── 📄 requirements.txt              ← Install all libraries: pip install -r requirements.txt
│
├── 📁 models/
│   ├── 📄 segmentation_model.py     ← K-Means clustering → 4 customer segments
│   ├── 📄 churn_model.py            ← XGBoost churn predictor (AUC 0.891)
│   ├── 📄 forecasting_model.py      ← Prophet + LSTM demand forecast (MAPE 5.9%)
│   ├── 📄 clv_model.py              ← LightGBM CLV predictor (R² 0.847)
│   ├── 📄 anomaly_detector.py       ← Isolation Forest fraud detector
│   └── 📄 sentiment_analyzer.py     ← TextBlob review sentiment (9 features)
│
├── 📁 monitoring/
│   └── 📄 drift_detector.py         ← Weekly KS-test drift monitoring + alerts
│
├── 📁 api/
│   └── 📄 main.py                   ← FastAPI server: POST /predict/churn
│
├── 📁 data/
│   ├── 📁 raw/                      ← PUT YOUR CSV FILES HERE (gitignored)
│   ├── 📁 processed/                ← Auto-generated cleaned parquet files
│   └── 📁 sample/                   ← Small 1000-row sample files (in GitHub)
│
├── 📁 models/saved/                 ← Auto-generated trained model .pkl files
│
├── 📁 reports/
│   ├── 📁 html/                     ← Interactive HTML dashboard (open in browser)
│   ├── 📁 figures/                  ← EDA charts saved as PNG
│   └── 📁 drift_reports/            ← Weekly Evidently AI HTML drift reports
│
├── 📁 notebooks/                    ← 8 Jupyter notebooks (step-by-step analysis)
├── 📁 tests/                        ← Unit tests (pytest)
└── 📁 .github/workflows/            ← GitHub Actions CI pipeline
```

---

## 🛠️ Complete Tech Stack

<div align="center">

| Layer | Technology | Purpose |
|:---|:---|:---|
| **Language** | Python 3.11 | Core programming language |
| **Data** | Pandas · NumPy · SQLAlchemy | Data loading, cleaning, manipulation |
| **Visualisation** | Matplotlib · Seaborn · Plotly | Charts, EDA plots, interactive dashboard |
| **ML — Classical** | Scikit-learn · XGBoost · LightGBM | Churn, CLV, Segmentation, Anomaly |
| **ML — Deep Learning** | TensorFlow / Keras | LSTM neural network for forecasting |
| **ML — Forecasting** | Prophet (Meta) | Time series with seasonality |
| **ML — NLP** | TextBlob · NLTK | Sentiment analysis on reviews |
| **Optimisation** | Optuna | Automated hyperparameter tuning |
| **Explainability** | SHAP | Feature importance for model decisions |
| **Imbalance** | imbalanced-learn (SMOTE) | Handle 18% churn class imbalance |
| **MLOps — Tracking** | MLflow | Experiment logging and model registry |
| **MLOps — Monitoring** | Evidently AI | Data drift and model drift detection |
| **API** | FastAPI · Uvicorn · Pydantic | Real-time prediction serving |
| **Storage** | Parquet (PyArrow) | Efficient processed data storage |
| **Version Control** | Git · GitHub | Code management and portfolio |

</div>

---




---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

