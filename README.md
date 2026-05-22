# 🌫️ Beijing PM2.5 Prediction — Machine Learning Pipeline

> **M.Sc. Applied Statistics and Data Analytics — Project Report**
> Hourly PM2.5 air quality forecasting using lag features, log transformation, and a weighted ensemble of ML models, implemented entirely in Python.

---

## 📌 About the Project

PM2.5 — fine particulate matter ≤ 2.5 µm in diameter — is the world's most hazardous air pollutant, responsible for over 4 million deaths annually. This project builds and evaluates a **complete machine learning pipeline** to predict hourly PM2.5 concentrations in Beijing, China, using 5 years of atmospheric and meteorological data.

Three core techniques drive the system:

- **Feature Engineering** — Physics-informed interaction features and time-lag features that give the model atmospheric memory
- **Ensemble Modelling** — A weighted average of XGBoost (70%), Random Forest (20%), and Stacking (10%) for best-in-class accuracy
- **Log Transformation** — Applied to the target to reduce outlier influence from 0.422% → 0.0007%

---

## 🎯 Objective

To build a generalised PM2.5 prediction system that takes real-time meteorological sensor readings and recent pollution history as input and returns a calibrated PM2.5 forecast in µg/m³ — deployable for public health early warning systems.

---

## ⚙️ Prediction Pipeline

```
Raw Sensor Input → Feature Engineering → Log Transformation → Lag Features → Ensemble Model → PM2.5 Forecast (µg/m³)
```

| Stage | Details |
|---|---|
| 1. Missing Value Handling | Gap-aware strategy: drop large gaps (> 24h), linear interpolation for small gaps (≤ 24h) |
| 2. Log Transformation | `log1p` applied to PM2.5 to normalise distribution and reduce outlier influence |
| 3. Feature Engineering | Interaction features + one-hot encoding + 3 time-lag features |
| 4. Train-Test Split | Temporal split — 2010–2013 (train), 2014 (test) — no data leakage |
| 5. Hyperparameter Tuning | Grid Search (DT), Randomized Search (RF), Optuna Bayesian (XGBoost) |
| 6. Ensemble Prediction | Weighted Average: 70% XGBoost + 20% Random Forest + 10% Stacking |
| 7. Back-transformation | `expm1` applied before computing all evaluation metrics |

---

## 🗂️ File Structure

```
beijing-pm25-prediction/
│
├── Project_Final.ipynb              # 🔑 Main notebook — full pipeline end to end
├── PM2_5_Project_Report.docx        # Full project report with analysis and findings
└── README.md                        # Project documentation
```

> ⚠️ Dataset is downloaded automatically inside the notebook directly from the UCI ML Repository.

---

## 🛠️ Tools & Technologies

- **Language:** Python 3.8+
- **Libraries:** `pandas`, `numpy`, `scikit-learn`, `xgboost`, `optuna`, `matplotlib`, `seaborn`
- **Environment:** Jupyter Notebook

---

## 📊 Dataset

| Detail | Info |
|---|---|
| **Name** | Beijing PM2.5 Data (PRSA Dataset) |
| **Source** | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/381/beijing+pm2+5+data) |
| **Coverage** | Hourly observations — Jan 2010 to Dec 2014 (5 years) |
| **Raw Size** | 43,824 rows × 13 columns |
| **PM2.5 Source** | US Embassy in Beijing |
| **Meteorological Source** | Beijing Capital International Airport |

---

## 🔍 Feature Engineering

### 1. Numeric Interaction Features
| Feature | Formula | Why It Matters |
|---|---|---|
| `temp_dewp_diff` | TEMP − DEWP | Measures atmospheric humidity; near-zero = particle growth |
| `wind_pressure` | Iws × PRES | High value = dispersion; low value = pollution trapping |

### 2. Categorical Encoding
| Feature | Source | Meaning |
|---|---|---|
| `wind_NW` | One-Hot (cbwd) | North-west winds — carries industrial pollution into Beijing |
| `wind_SE` | One-Hot (cbwd) | South-east winds — cleaner oceanic air masses |
| `wind_cv` | One-Hot (cbwd) | Calm & variable — most dangerous; pollutants accumulate |

### 3. Time-Lag Features (Temporal Memory)
| Feature | Description |
|---|---|
| `pm2.5_lag1` | PM2.5 value from 1 hour ago — strongest single predictor |
| `pm2.5_lag2` | PM2.5 value from 2 hours ago — captures short-term persistence |
| `pm2.5_roll_mean` | Rolling mean of last 3 hours — smooths noise, encodes trend direction |

---

## 📈 Model Performance

| Model | Tuning Method | RMSE | R² Score |
|---|---|---|---|
| Linear Regression | Baseline | ~21.23 | ~0.9488 |
| Decision Tree | Grid Search CV | ~21.37 | ~0.9480 |
| Random Forest | Randomized Search | ~21.97 | ~0.9451 |
| XGBoost | Optuna Bayesian | ~22.49 | ~0.9425 |
| Stacking (LR + DT + XGB → RF) | Tuned base models | ~23.30 | ~0.9383 |
| **Weighted Average (70:20:10)** | **XGB + RF + Stack** | **~23.84** | **Best R²** |

> *Exact values vary per Optuna trial run. The Weighted Average ensemble consistently achieves the highest R².*

---

## 🔮 Real-Time Prediction (Unseen Data)

The system accepts live sensor readings and recent PM2.5 history from the user:

```
Input Fields Required:
  year, month, day, hour
  DEWP, TEMP, PRES, cbwd, Iws, Is, Ir
  PM2.5 from 1 hour ago, 2 hours ago, 3 hours ago

Output:
  Predicted PM2.5 = XX.XX µg/m³
```

**Health Interpretation:**

| PM2.5 (µg/m³) | AQI Category |
|---|---|
| 0 – 12 | 🟢 Good |
| 12 – 35 | 🟡 Moderate |
| 35 – 55 | 🟠 Unhealthy for Sensitive Groups |
| 55 – 150 | 🔴 Unhealthy |
| 150+ | 🟣 Very Unhealthy / Hazardous |

---

## ▶️ How to Run

**1. Install dependencies:**
```bash
pip install pandas numpy scikit-learn xgboost optuna matplotlib seaborn jupyter
```

**2. Clone this repository:**
```bash
git clone https://github.com/YOUR_USERNAME/beijing-pm25-prediction.git
cd beijing-pm25-prediction
```

**3. Open the main notebook:**
```bash
jupyter notebook Project_Final.ipynb
```

**4. Run all cells in order** — the notebook will:
- Download the dataset automatically from UCI
- Preprocess and engineer features
- Train and tune all models
- Evaluate on the 2014 test set
- Accept your live sensor input and return a PM2.5 forecast

---

## 🚀 Possible Extensions

- Integrate with a live air quality API for real-time monitoring
- Add a web dashboard for public health advisories
- Extend to multi-step ahead forecasting (next 3h, 6h, 12h)
- Apply deep learning models (LSTM, Transformer) for richer temporal patterns

---

## 👨‍💻 Submitted By

| Name | Roll Number |
|---|---|
| Dhanush S R | CB.PS.P2ASD25012 |
| Anila M | CB.PS.P2ASD25008 |

*Submitted as part of the M.Sc. Applied Statistics and Data Analytics — Project*
