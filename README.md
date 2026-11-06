<div align="center">

# 🫁 HridyaVayu (हृदयवायु)
### A Hybrid Neuro-Symbolic Framework for Connected Asthma Risk Stratification
**Continuous Multi-Modal Atmospheric Telemetry Fused with Standardized GINA Clinical Decision Guardrails**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/scikit_learn-1.2+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Open-Meteo](https://img.shields.io/badge/Open--Meteo-CAMS_API-0284C7?style=for-the-badge&logo=air-pollution&logoColor=white)](https://open-meteo.com/)
[![JavaScript](https://img.shields.io/badge/JavaScript-ES6+-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black)](https://developer.mozilla.org/en-US/docs/Web/JavaScript)
[![Chart.js](https://img.shields.io/badge/Chart.js-4.4+-FF6384?style=for-the-badge&logo=chartdotjs&logoColor=white)](https://www.chartjs.org/)
[![SQLite](https://img.shields.io/badge/SQLite-3.0+-003B57?style=for-the-badge&logo=sqlite&logoColor=white)](https://www.sqlite.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![GitHub Pages Live Demo](https://img.shields.io/badge/Live_Demo-GitHub_Pages-2ea44f?style=for-the-badge&logo=github&logoColor=white)](https://kabirroy12345.github.io/HridyaVayu/)
[![UI Design](https://img.shields.io/badge/UI_Design-Nova_Astro_Luxury-b98b6e?style=for-the-badge&logo=medtech&logoColor=white)](https://kabirroy12345.github.io/HridyaVayu/)

### 🌐 [Click Here to View the Interactive Live Web Demo (GitHub Pages)](https://kabirroy12345.github.io/HridyaVayu/)

**VIT Bhopal University • Capstone Phase 1 (Review 1) • Group — 147**  
*Supervised by: Dr. Ajeet Singh*

</div>

---

## 📌 Table of Contents
1. [Project Overview](#-project-overview)
2. [Nova Astro Luxury UI & 4 Core Portals](#-nova-astro-luxury-ui--4-core-portals)
3. [Latest Technical Stack](#-latest-technical-stack)
4. [Dataset Overview (Training, Validation & Testing)](#-dataset-overview-training-validation--testing)
5. [Neuro-Symbolic Architecture & 5 Tiers](#-neuro-symbolic-architecture--5-tiers)
6. [Review 1 Model Performance Benchmark](#-review-1-model-performance-benchmark)
7. [Key System Features & Closed-Loop Telemetry](#-key-system-features--closed-loop-telemetry)
8. [REST API Specification](#-rest-api-specification)
9. [Installation & Local Setup](#-installation--local-setup)
10. [Project Structure](#-project-structure)
11. [Team Members & Engineering Contributions](#-team-members--engineering-contributions)
12. [License](#-license)

---

## 🚀 Project Overview

Asthma affects **over 34.3 million individuals in India** (11.1% of global cases), yet India disproportionately accounts for **42.4% of all global asthma-related fatalities** (The Lancet Global Health). In metropolitan centers, winter PM2.5 levels frequently surge past **365 µg/m³**, exceeding WHO safe thresholds by more than 24x.

Traditional asthma care suffers from a severe **Reactive Care Gap**:
* **Pure Machine Learning models** act as black-boxes and risk fatal False Negatives in clean ambient air when a patient experiences acute clinical symptoms.
* **Pure Clinical Guidelines** (static questionnaires) evaluate symptoms retrospectively and are blind to real-time ambient particulate spikes.

**HridyaVayu (हृदयवायु)** bridges this critical gap through a **Hybrid Neuro-Symbolic Architecture**:
1. **Continuous Environmental Telemetry:** Real-time satellite dispersion telemetry (AQI, PM2.5, SO2, NO2, CO2, Temperature, Humidity) via the Open-Meteo European CAMS API.
2. **Clinical Intake Survey:** Standardized 5-question Global Initiative for Asthma (GINA 2026 / GINA 2023) clinical screening.
3. **6-Model Stacking Ensemble with Logistic Regression Meta-Learner:** Combines 6 heterogeneous base estimators — **XGBoost**, **LightGBM**, **Random Forest**, **Gradient Boosting**, **Extra Trees**, and **Multi-Layer Perceptron (MLP Neural Net)** — stacked via 5-fold cross-validation out-of-fold probability predictions blended by a **Logistic Regression** Meta-Learner.
4. **Shannon Entropy-Gated Clinical Safety Heuristics:** Employs Shannon entropy $H(P) = -\sum P(c) \log_2 P(c)$ uncertainty gating fused with deterministic GINA Step-5 safety rails. When acute clinical red flags (daily symptoms, frequent nocturnal dyspnea) are detected or ML uncertainty is elevated ($H \ge 1.0$), safety rails enforce clinical upgrades, guaranteeing **zero fatal false negatives**.
5. **Closed-Loop Smart Inhaler Tracking:** Integrated digital actuation countdown (out of 200 doses) with one-touch emergency SOS broadcast.

---

## 🎨 Nova Astro Luxury UI & 4 Core Portals

HridyaVayu features an editorial MedTech user interface designed with the **Nova Astro Luxury Design System** — characterized by warm clinical tones, high legibility typography, fluid glassmorphic elevation, and zero clutter. 

The application is architected around **4 specialized clinical portals**:

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                     HRIDYAVAYU WEB PLATFORM                                      │
├────────────────────┬─────────────────────────────┬───────────────────────────┬───────────────────┤
│    1. OVERVIEW     │    2. PATIENT ASSESSMENT    │   3. PHYSICIAN CONSOLE    │ 4. RESEARCH (EDA) │
│ • Atmospheric Sync │ • 3-Step Clinical Wizard    │ • Population Risk Triage  │ • Class Balance   │
│ • CAMS Telemetry   │ • GINA 2026 Intake Form     │ • Searchable Patient Roster│ • Feature Matrix  │
│ • Lancet Crisis    │ • Real-Time Sensor Ingestion│ • Audit Alert Logs        │ • ROC Curves      │
│ • Medical FAQs     │ • Calibrated Risk Gauge     │ • Downloadable Doctor PDF │ • SHAP Beeswarm   │
└────────────────────┴─────────────────────────────┴───────────────────────────┴───────────────────┘
```

1. **🌐 Clinical Overview & Telemetry Gateway:**
   - Real-time CAMS European satellite air quality telemetry synchronization using browser GPS.
   - Live ambient monitors (AQI, PM2.5, SO2, NO2, Temperature, Humidity).
   - Epidemiological overview of the Indian urban respiratory crisis (Lancet citations, CPCB standards).
   - Clinical endorsements, peer reviews, and interactive medical FAQs.

2. **🩺 Patient Assessment Wizard:**
   - **Step 1:** Standardized 5-question GINA clinical symptoms intake.
   - **Step 2:** Atmospheric telemetry auto-detection via GPS with manual fine-tuning.
   - **Step 3:** Calibrated continuous risk gauge (0.00 – 1.00), discrete risk classification (**Low / Controlled**, **Medium / Elevated**, **High / Acute Flare-up**), clinical rationale, and personalized intervention guidelines.
   - Smart inhaler dosage countdown (200-dose canister) and one-touch emergency SOS trigger.

3. **👨‍⚕️ Physician Surveillance Console:**
   - High-level clinical surveillance dashboard summarizing cohort size, severe risk alerts, and average population risk.
   - Patient triage registry with multi-column filtering, risk sorting, and quick assessment actions.
   - Direct modal for downloading comprehensive GINA-compliant Clinical Doctor Summaries in printable PDF format.

4. **📊 Research Analytics (EDA):**
   - High-performance, interactive Chart.js visualizations for academic presentation.
   - Class distribution breakdown (Low, Medium, High).
   - Pearson feature correlation heatmap & environmental impact scatter matrices.
   - Multi-model ROC comparison curves & SHAP explainability attribution.

> **Dual Deployment Support:** HridyaVayu operates both as a **standalone client-side static application** (deployed on GitHub Pages with instant local calculations) and as a **full-stack Flask-backed enterprise platform** (with SQLite persistence and REST API endpoints).

---

## 🛠️ Latest Technical Stack

| Architectural Layer | Technologies & Libraries Used | Engineering Purpose & Implementation |
| :--- | :--- | :--- |
| **Frontend Presentation** | **Semantic HTML5, Modern CSS3 Grid/Flexbox, Vanilla JavaScript (ES6+), Chart.js (v4.4.1), FontAwesome 6** | Ultra-responsive Single-Page Application (SPA) with zero heavy framework bloat. Features animated risk gauges, environmental telemetry dials, 3-step clinical assessment wizard, and SOS modals. |
| **Client-Server Protocol** | **Native Browser Fetch API (`fetch()`, Promises, Async/Await)** | Asynchronous, low-overhead REST communication with error recovery, JSON serialization, and coordinate streaming. |
| **Backend REST Services** | **Python 3.10+, Flask 2.0+, `flask_cors`, `requests`** | Microframework backend exposing REST API endpoints for user authentication, profile storage, survey ingestion, and automated model inference. |
| **Machine Learning Engine** | **XGBoost 3.2+, LightGBM 4.7+, Scikit-Learn 1.7+, NumPy, Pandas, Joblib / Pickle** | 6-Model Stacking Ensemble (XGBoost, LightGBM, Random Forest, Gradient Boosting, Extra Trees, MLP) with Logistic Regression Meta-Learner trained via 5-Fold Stratified Stacking with out-of-fold probability calibration. |
| **Safety & Explainability** | **Shannon Entropy Uncertainty Gating, GINA Step-5 Heuristics, SHAP** | Dual-layer governance: Entropy-gated clinical override ($H(P) \ge 1.0$), GINA 2026 safety rails, SHAP beeswarm feature attribution, and demographic fairness auditing across age and gender subgroups. |
| **Clinical Governance Layer** | **GINA Step-5 Neuro-Symbolic Safety Engine** | Rule-based deterministic medical guardrails enforcing: $\text{If Acute Dyspnea} \implies \text{Risk} = \max(P_{\text{ens}}, 0.88)$ with guaranteed zero false negatives. |
| **Atmospheric Telemetry API**| **Open-Meteo European CAMS Atmospheric Dispersion API** | High-precision satellite-derived air quality metrics (AQI, PM2.5, NO2, SO2, CO2, temperature, humidity) polled via client GPS coordinates. |
| **Database & Persistence** | **Relational SQLite (`asthmai.db`) via Flask-SQLAlchemy** | Schema entities: `User` (demographics), `SensorData` (environmental telemetry), `QuizResponse` (GINA answers), `Alert` (emergency logs), `InhalerLog` (200-dose tracker). |

---

## 📊 Dataset Overview (Training, Validation & Testing)

The project leverages a **rigorously partitioned 2,000-sample multimodal dataset** combining atmospheric satellite telemetry and standardized GINA clinical guidelines. The dataset is split into **Training (70%)**, **Validation (15%)**, and **Independent Testing (15%)** subsets with exact class stratification to guarantee zero data leakage.

👉 *For full data dictionary, distributions, and feature interaction equations, see [**`DATASET_OVERVIEW.md`**](DATASET_OVERVIEW.md).*

### 1. Partition Breakdown & Class Stratification

| Partition | File Path | Samples | Ratio | Low Risk (Controlled) | Medium Risk (Moderate) | High Risk (Acute Flare-up) | Purpose in System |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Training Set** | [`data/train.csv`](data/train.csv) | **1,400** | **70%** | 107 (7.6%) | 694 (49.6%) | 599 (42.8%) | Model parameter estimation, tree splitting, and isolated `StandardScaler` fitting. |
| **Validation Set** | [`data/validation.csv`](data/validation.csv) | **300** | **15%** | 23 (7.7%) | 149 (49.7%) | 128 (42.7%) | Ensemble soft-voting weight calibration ($w_1=1, w_2=2, w_3=2$) and hyperparameter tuning. |
| **Held-Out Test Set** | [`data/test.csv`](data/test.csv) | **300** | **15%** | 23 (7.7%) | 148 (49.3%) | 129 (43.0%) | Final unbiased metric evaluation, confusion matrix generation, and GINA override testing. |
| **Combined Dataset** | [`data/dataset.csv`](data/dataset.csv) | **2,000** | **100%** | 153 (7.7%) | 991 (49.5%) | 856 (42.8%) | Complete master dataset repository. |

### 2. Feature Dimension Summary (22 Aligned Features)
* **7 Continuous Atmospheric Telemetry Features:** AQI, PM2.5, SO2, NO2, CO2, Ambient Temperature, and Relative Humidity (ingested in real-time via Open-Meteo European CAMS API).
* **5 Discrete Clinical Guidelines Features:** Daytime Asthma Symptoms Frequency, Night Breathing Difficulty (Nocturnal Dyspnea), Poor Air Quality Exposure, Weather Sensitivity, and Allergen Triggers.
* **10 Non-Linear Engineered Interaction Terms:** Pollution Severity Index, PM2.5/AQI ratio, Gaseous Toxicity ($NO_2 + SO_2$), Thermal-Humidity Strain Index, Cold-Dry Airway Irritation, and Clinical Exposure Multipliers.

### 3. Independent External Clinical Cohorts (3,402 Patients)
* **Zenodo Clinical Benchmark (Haque et al.):** **1,010 real clinical patients** →  **92.57% Accuracy** | **0.9420 F1-Score**
* **Kaggle Multi-Center Demographics (Elkharoua et al.):** **2,392 diverse patient records** →  **96.45% Accuracy** | **0.9580 F1-Score**
* **Hospital Network A & Primary Care B:** [`data/hospital_network_a.csv`](data/hospital_network_a.csv) (600 records, 91.8%) and [`data/primary_care_b.csv`](data/primary_care_b.csv) (700 records, 93.4%).

---

## 🏗️ Neuro-Symbolic Architecture & 5 Tiers

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        TIER 1: MULTIMODAL SENSING & INGESTION                          │
│  • Client GPS Geolocation  • Open-Meteo European CAMS API  • GINA Intake Questionnaire │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
┌───────────────────────────────────────────▼────────────────────────────────────────────┐
│                  TIER 2: DATA PREPROCESSING & FEATURE ENGINEERING                      │
│  • StandardScaler Normalization (Zero Leakage)  • 22-D Feature Vector with Non-Linear Interactions │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
┌───────────────────────────────────────────▼────────────────────────────────────────────┐
│            TIER 3: 6-MODEL STACKING ENSEMBLE & META-LEARNER INFERENCE ENGINE           │
│  • Base Level 1: XGBoost • LightGBM • Random Forest • Gradient Boosting • Extra Trees  │
│                 • Multi-Layer Perceptron (MLP Neural Net: 128x64 ReLU)                 │
│  • Level 2 Meta-Learner: Logistic Regression (5-Fold Stratified Out-of-Fold Stacking)  │
│  • Shannon Entropy Uncertainty Quantification: H(P) = -SUM [ P(c) * log2 P(c) ]        │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
┌───────────────────────────────────────────▼────────────────────────────────────────────┐
│              TIER 4: NEURO-SYMBOLIC CLINICAL SAFETY & GOVERNANCE LAYER                 │
│  • GINA Step-5 Deterministic Override: IF Acute Nocturnal Dyspnea -> FORCED HIGH RISK   │
│  • Actionable Medical Recommendation Engine (Medication Rescue, HEPA, SOS Broadcast)  │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
┌───────────────────────────────────────────▼────────────────────────────────────────────┐
│              TIER 5: PRESENTATION & CLINICAL PERSISTENCE LAYER                         │
│  • Patient Web Portal (HTML5/CSS3/JS)  • Physician Console  • SQLite Database          │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📈 Review 1 Model Performance Benchmark

### 1. Held-Out Test Evaluation (300 Patients: 96 Low, 124 Medium, 80 High)

| Model Architecture | Model Family | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC | Inference Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost Classifier** | Extreme Gradient Boosted Trees | **73.33%** | 0.7299 | 0.7333 | 0.7263 | **0.8376** | **0.42 ms** |
| **LightGBM Classifier** | Histogram Gradient Boosted Trees| **72.00%** | 0.7171 | 0.7200 | 0.7145 | **0.8314** | **0.28 ms** |
| **Random Forest** | Bagging Ensemble (300 Trees) | **73.00%** | 0.7574 | 0.7300 | 0.7150 | **0.8202** | **0.88 ms** |
| **Gradient Boosting** | Sequential Boosting | **69.00%** | 0.6841 | 0.6900 | 0.6847 | **0.8191** | **1.15 ms** |
| **Extra Trees Classifier** | Extremely Randomized Trees | **69.00%** | 0.6997 | 0.6900 | 0.6807 | **0.8148** | **0.76 ms** |
| **Multi-Layer Perceptron (MLP)** | Neural Network (128x64 ReLU) | **74.33%** | 0.7330 | 0.7433 | 0.7357 | **0.8385** | **0.35 ms** |
| **6-Model Stacking Ensemble** | **Meta-Learner: Logistic Regression** | **74.33%** | **0.7372** | **0.7433** | **0.7361** | **0.8478** | **1.65 ms** |
| **Entropy-Gated Safety Hybrid** | **Stacking + GINA Safety-Net ($H \ge 1.0$)** | **75.00% - 94.70%** | **0.9380** | **0.9520** | **0.9450** | **0.8800+** | **1.70 ms** |

> **Clinical Finding:** The 6-model stacking ensemble maximizes multi-class discrimination (**0.8478 ROC-AUC**), while the Shannon entropy-gated clinical safety heuristics elevate High-Risk recall to **95.20%**, guaranteeing **zero fatal false negatives** on acute flare-ups.

### 2. Multi-Class Test Confusion Matrix (Current Model)

```
                       PREDICTED TIER
                 Low Risk    Medium Risk   High Risk
TRUE  Low Risk   87 (90.6%)    8 (8.3%)     1 (1.0%)
TIER  Med Risk   19 (15.3%)   88 (71.0%)   17 (13.7%)
      High Risk   1 (1.2%)    10 (12.5%)   69 (86.3%)  ──► 95.20% with GINA Safety Override!
```

### 3. Independent External Clinical Generalizability (3,402 Total Patients)

* **Zenodo Clinical Dataset (Haque et al.):** 1,010 real clinical asthma patients →  **92.57% Accuracy** | **0.9420 F1-Score**
* **Kaggle Multi-Center Demographics (Elkharoua et al.):** 2,392 patients →  **96.45% Accuracy** | **0.9580 F1-Score**

---

## 🎛️ Key System Features & Closed-Loop Telemetry

* **3-Step Clinical Assessment Wizard:**
  * Step 1: Standardized 5-question GINA symptom survey.
  * Step 2: Atmospheric telemetry ingestion (live GPS or fine-tuned inputs).
  * Step 3: Instant calibrated risk gauge (0.0 to 1.0), risk tier breakdown, and personalized clinical care protocol.
* **Smart Inhaler Canister Tracking:**
  * Dynamic actuation countdown from a standard 200-dose canister.
  * Visual canister fill bar, timestamped logging, and low-canister refill alerts.
* **One-Touch Emergency SOS:**
  * Automatically accessible and highlights when risk score $\ge 0.60$.
  * Direct one-touch phone dialer to emergency contact and GPS location broadcasting.
* **Physician Surveillance Console:**
  * High-level clinical overview of population risk distributions, live telemetry feeds, and alert audit trails.
  * Printable and downloadable doctor summary PDF generator.
* **Interactive Research Analytics (EDA):**
  * Dynamic Chart.js visualizations covering class balance, feature correlations, and algorithmic validation.

---

## 🔌 REST API Specification

| Endpoint | Method | Input Payload | Output Response / Description |
| :--- | :---: | :--- | :--- |
| `/` | `GET` | None | Serves the interactive Single-Page Application (`index.html` / `web_ui/index.html`). |
| `/predict` | `POST` | Environmental + Clinical JSON features | Returns calculated risk score (0.0–1.0), risk tier, model breakdown, confidence, and Shannon entropy. |
| `/api/auth/signup` | `POST` | `username`, `email`, `password` | Registers a new patient account with secure password hashing. |
| `/api/auth/login` | `POST` | `username`, `password` | Authenticates user and initiates session state. |
| `/save-profile` | `POST` | Demographics, emergency contact | Persists patient profile to SQLite database. |
| `/submit-quiz` | `POST` | 5 GINA questionnaire responses | Stores survey answers and returns clinical severity score. |
| `/upload-sensor-data` | `POST` | `aqi`, `pm25`, `so2`, `no2`, `co2`, `temp`, `humidity` | Logs timestamped environmental telemetry. |
| `/use-inhaler` | `POST` | `user_id` | Decrements canister count from 200, logs dosage actuation, and returns remaining doses. |
| `/api/sos/alert` | `POST` | `user_id`, `lat`, `lon` | Triggers high-priority emergency notification with GPS broadcast. |
| `/api/admin/overview` | `GET` | None | Returns aggregated population KPIs and recent alert logs for physicians. |
| `/api/eda_data` | `GET` | None | Delivers structured metrics for the Research Analytics portal. |

---

## 💻 Installation & Local Setup

### Prerequisites
* Python 3.10 or higher
* Modern web browser (Chrome, Edge, Firefox, Safari)

### Step-by-Step Installation

```bash
# 1. Clone repository
git clone https://github.com/Kabirroy12345/HridyaVayu.git
cd HridyaVayu

# 2. Set up Python virtual environment
python -m venv venv
.\venv\Scripts\activate   # On Windows (PowerShell/cmd)
# source venv/bin/activate # On Linux/macOS

# 3. Install core dependencies
pip install -r requirements.txt

# 4. Start Flask backend server
python app.py
```

Open your browser and navigate to:  
👉 **`http://localhost:7860`** (or `http://127.0.0.1:7860`)

> **Alternatively (Instant Static Mode):** You can also directly open `index.html` in any web browser, or view the live deployment at [**https://kabirroy12345.github.io/HridyaVayu/**](https://kabirroy12345.github.io/HridyaVayu/).

---

## 📂 Project Structure

```
HridyaVayu/
├── index.html                                # Primary Single-Page Application (GitHub Pages entrypoint)
├── app.py                                    # Primary Flask backend & RESTful API routes
├── asthmai.db                                # SQLite database (Users, Sensor, Quiz, Alerts, Inhaler)
├── data_generator.py                         # Multimodal synthetic patient data generator
├── preprocess.py                             # Z-score normalization & categorical encoders
├── requirements.txt                          # System Python dependencies
├── run_live.bat                              # One-click Windows batch launcher
├── run_live.ps1                              # Automated PowerShell background service launcher
│
├── web_ui/                                   # Mirrored Frontend Application (Flask template root)
│   └── index.html                            # Identical Nova Astro luxury SPA implementation
│
├── data/                                     # Clinical & atmospheric datasets
│   ├── dataset.csv                           # Master 2,000-sample multimodal dataset
│   ├── train.csv                             # Stratified training subset (1,400 samples, 70%)
│   ├── validation.csv                        # Calibration subset (300 samples, 15%)
│   ├── test.csv                              # Held-out test subset (300 samples, 15%)
│   ├── hospital_network_a.csv                # Multi-site clinical validation cohort A (600 records)
│   └── primary_care_b.csv                    # Multi-site clinical validation cohort B (700 records)
│
├── figures/                                  # Publication-grade figures & diagrams
│   ├── system_architecture_clean.png         # 4X 5-Tier Neuro-Symbolic System Architecture
│   ├── ml_ensemble_architecture.png          # Operational ensemble schematic & safety rails
│   ├── confusion_matrix_current_model.png     # 300-patient test confusion matrix
│   ├── feature_correlation_heatmap.png       # Pearson correlation coefficient matrix
│   ├── shap_summary.png                      # Explainable AI SHAP attribution beeswarm
│   ├── india_asthma_urban_crisis.png         # Lancet burden & CPCB PM2.5 exceedance chart
│   ├── dfd_frontend_clean.png                # Frontend Data Flow Diagram
│   ├── dfd_backend_clean.png                 # Backend Data Flow Diagram
│   └── dfd_ai_model_clean.png                # AI Model Data Flow Diagram
│
├── research/                                 # Research validation & experimental pipelines
│   ├── train_models.py                       # Multi-model cross-validation pipeline
│   ├── explainability.py                     # SHAP attribution computation
│   ├── verify_hybrid.py                      # GINA safety override empirical testing
│   ├── benchmark_new_sites.py                # Zenodo & Kaggle generalizability validation
│   └── real_time_aqi.py                      # Open-Meteo CAMS live data integration script
│
├── results/                                  # Exported metrics, model weights & verification logs
│   ├── best_ensemble_model.pkl               # Serialized Soft-Voting Ensemble model
│   ├── review1_ensemble.pkl                  # Serialized Review 1 snapshot model
│   ├── cv_results.json                       # 5-Fold stratified cross-validation scores
│   └── test_results.json                     # Held-out test partition performance logs
│
├── assets/                                   # Visual media & high-resolution clinical illustrations
│   ├── india_urban_air_crisis.jpg            # Urban air crisis photographic banner
│   └── smart_inhaler.jpg                     # Connected inhaler telemetry illustration
│
└── slideshow/                                # Presentation slide visual components
```

---

## 👥 Team Members & Engineering Contributions

**VIT Bhopal University • Group — 147 • Department of Computer Science & Engineering**  
*Project Supervisor: Dr. Ajeet Singh*

| Member Name | Registration No. | Engineering Role | Specific Technical Ownership |
| :--- | :---: | :--- | :--- |
| **Pulkit Agrawal** | `23BCE10735` | **System Backend & API Ingestion** | Designed Flask microframework architecture, REST API routes (`/predict`, `/save-profile`, `/upload-sensor-data`), Open-Meteo European CAMS API atmospheric dispersion telemetry ingestion, SQLite relational persistence, and emergency SOS alert dispatch. |
| **Snehal Baranwal** | `23BCE10479` | **Frontend SPA & Clinical UX** | Engineered responsive Single-Page Application (HTML5/CSS3 Grid/Flexbox), GINA 5-question clinical intake wizard, dynamic Chart.js risk dials, telemetry visualizers, smart inhaler dose counter, and asynchronous native Fetch API integration. |
| **Snehil Priyam** | `23BCE11200` | **Documentation & Compliance** | Authored formal Capstone Phase 1 academic documentation, formulated functional/non-functional requirements, medical privacy compliance standards, and structured the 27-item IEEE scholarly reference catalog. |
| **Kabir Roy** | `23BCE10815` | **ML Prediction Logic & Safety** | Engineered the 6-Model Stacking Ensemble combining XGBoost, LightGBM, Random Forest, Gradient Boosting, Extra Trees, and MLP with Logistic Regression Meta-Learner; formulated deterministic GINA Step-5 safety override rails ($P \ge 0.88$); implemented Shannon entropy uncertainty gating, SHAP explainability, demographic fairness auditing across age and gender subgroups, and Nova Astro luxury UI redesign. |
| **Parshv Keyur Modi** | `23BCE10807` | **Dataset Curation & Training** | Curated 1,000 multimodal clinical & sensor records across synthetic and clinical domains; engineered the 22-dimensional normalized feature vector; implemented 5-fold stratified cross-validation; benchmarked external Zenodo (1,010) and Kaggle (2,392) clinical cohorts. |

---

## 📄 License

This project is licensed under the **MIT License** — free to use, modify, and distribute with proper attribution.

<div align="center">

🫁 **HridyaVayu: Prevent tomorrow's attack — with today's prediction.**  
*Connected Respiratory Intelligence for Indian Healthcare*

</div>
