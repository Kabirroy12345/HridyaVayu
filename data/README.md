# 📊 HridyaVayu: Dataset Overview & Data Architecture
### Multimodal Atmospheric Telemetry & GINA Clinical Guidelines Dataset Breakdown
**Department of Computer Science & Engineering, VIT Bhopal University • Group — 147**

---

## 📌 1. Dataset Partitioning Summary

The core dataset contains **2,000 multimodal clinical and atmospheric records** partitioned into strictly stratified **Training (70%)**, **Validation (15%)**, and **Independent Testing (15%)** subsets. This rigorous division ensures zero data leakage while assessing the model's true generalization capacity across low, moderate, and acute asthma exacerbation scenarios.

| Dataset Partition | File Path | Sample Count | Percentage | Class Stratification (Low / Medium / High) | Primary Experimental Role |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Training Set** | [`data/train.csv`](data/train.csv) | **1,400** | **70.0%** | 107 (7.6%) / 694 (49.6%) / 599 (42.8%) | Model parameter estimation, tree splitting, and feature scaler fitting (`StandardScaler`). |
| **Validation Set** | [`data/validation.csv`](data/validation.csv) | **300** | **15.0%** | 23 (7.7%) / 149 (49.7%) / 128 (42.7%) | Out-of-fold probability calibration, entropy threshold tuning ($H \ge 1.0$), and MLP early stopping. |
| **Held-Out Test Set** | [`data/test.csv`](data/test.csv) | **300** | **15.0%** | 23 (7.7%) / 148 (49.3%) / 129 (43.0%) | Final unbiased performance benchmarking, confusion matrix generation, and clinical safety verification. |
| **Full Combined Dataset**| [`data/dataset.csv`](data/dataset.csv) | **2,000** | **100.0%** | 153 (7.7%) / 991 (49.5%) / 856 (42.8%) | Master aggregated repository dataset. |

---

## 🛡️ 2. Data Hygiene & Zero-Leakage Protocol

To ensure publication-grade scientific validity, the preprocessing pipeline follows strict medical data hygiene principles:
1. **Isolated Scaler Fitting:** The feature scaling transformation (`StandardScaler`) is fitted **strictly on the 1,400 training samples**. The derived mean ($\mu$) and standard deviation ($\sigma$) parameters are frozen and applied to the validation and test sets without recalculation.
2. **Zero Information Bleed:** No test or validation statistics (e.g. min, max, quantiles) are ever exposed during training or feature engineering.
3. **Exact Class Stratification:** The 3-class distribution is preserved identically across all three subsets to prevent sampling bias:
   * **Low Risk:** ~7.7%
   * **Medium Risk:** ~49.5%
   * **High Risk:** ~42.8%

---

## 📋 3. Comprehensive Data Dictionary

### A. Continuous Environmental & Atmospheric Telemetry (7 Features)
Captured via real-time satellite dispersion telemetry from the **Open-Meteo European CAMS API** (ECMWF):

| Feature Name | Type | Physical Unit | Mean ± Std | Min – Max | Clinical Relevance & Physiological Impact |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **AQI** | Integer | Index [0–500] | 185.50 ± 69.33 | 41 – 345 | Composite ambient air quality severity index. Values >150 trigger acute airway inflammation. |
| **PM2.5** | Continuous | µg/m³ | 78.04 ± 37.97 | 4.73 – 172.12 | Fine respirable particulate matter penetrating deep alveolar lung tissues; major exacerbation driver. |
| **SO2 level** | Continuous | ppb / µg/m³ | 26.26 ± 13.63 | 0.60 – 57.68 | Sulfur dioxide gas emitted from combustion; causes rapid bronchoconstriction in asthmatics. |
| **NO2 level** | Continuous | ppb / µg/m³ | 54.36 ± 25.03 | 4.28 – 114.03 | Nitrogen dioxide from vehicular emissions; increases bronchial hyper-reactivity to allergens. |
| **CO2 level** | Continuous | ppm | 443.81 ± 83.70 | 254.45 – 670.92 | Carbon dioxide concentration; serves as a proxy for enclosed spaces and stagnant air pockets. |
| **Humidity** | Continuous | % | 58.13 ± 16.28 | 24.06 – 101.67 | Atmospheric relative humidity; high humidity swells mold spores while dry air irritates mucous membranes. |
| **Temperature**| Continuous | °C | 21.94 ± 9.72 | 4.06 – 45.76 | Ambient thermal index; sudden drops induce cold-induced bronchospasm. |

### B. Standardized GINA Clinical Guidelines Features (5 Features)
Collected via standardized 5-question intake screening following **Global Initiative for Asthma (GINA 2023)** protocols:

| Feature Name | Variable Type | Possible Categorical Values & Cohort Frequency | GINA Clinical Rationale |
| :--- | :---: | :--- | :--- |
| **Asthma Symptoms Frequency** | Categorical (Ordinal) | • `Daily` (682 records, 34.1%)<br>• `Less than once a month` (491 records, 24.6%)<br>• `Frequently (Weekly)` (466 records, 23.3%)<br>• `1-2 times a month` (361 records, 18.1%) | Primary measure of daytime asthma control. Daily symptoms denote uncontrolled persistent asthma (GINA Step 4/5). |
| **Night Breathing Difficulty (Nocturnal Dyspnea)** | Categorical (Ordinal) | • `Never` (594 records, 29.7%)<br>• `Rarely` (535 records, 26.8%)<br>• `Frequently` (442 records, 22.1%)<br>• `Occasionally` (429 records, 21.5%) | Critical clinical red flag. Frequent nocturnal awakenings indicate high risk of fatal asthma exacerbation; triggers GINA Safety Override. |
| **Poor Air Quality Exposure** | Categorical | • `Yes, often` (689 records, 34.5%)<br>• `Occasionally` (685 records, 34.3%)<br>• `No` (626 records, 31.3%) | Self-reported environmental risk factor assessing occupational and commuting exposure. |
| **Weather Sensitivity** | Categorical | • `No specific weather-related triggers` (583 records, 29.2%)<br>• `Cold weather` (486 records, 24.3%)<br>• `Windy and dry weather` (478 records, 23.9%)<br>• `Hot and humid weather` (453 records, 22.7%) | Individual respiratory reaction to barometric, thermal, and humidity fluctuations. |
| **Triggers** | Multi-Select Categorical | • `Pollen` (194)<br>• `Humidity` (194)<br>• `Dust` (181)<br>• `Air pollution (smoke, chemicals)` (113)<br>• Combinations (e.g. `Dust + Pollen + Pollution`) | Specific allergen and environmental triggers causing immediate airway hyper-responsiveness. |

### C. Engineered Composite & Interaction Features (15 Features)
To enable the 6-model stacking ensemble to capture non-linear synergistic triggers, **15 engineered features** are computed alongside the 12 raw inputs (7 environmental + 5 clinical), constructing the full **27-dimensional input space ($X \in \mathbb{R}^{27}$)**:

1. **`pollution_index`:** $0.4 \cdot \text{AQI} + 0.3 \cdot \text{PM2.5} + 0.15 \cdot \text{NO}_2 + 0.15 \cdot \text{SO}_2$
2. **`AQI_PM_ratio`:** $\text{AQI} / (\text{PM2.5} + 1)$ — particulate vs. composite air quality dominance.
3. **`gas_pollution`:** $\text{CO}_2 \times \text{NO}_2 \times \text{SO}_2 / 10{,}000$ — multi-gas combustion synergy.
4. **`humidity_pollution`:** $\text{Humidity} \times \text{pollution\_index} / 100$ — humid bio-aerosol retention.
5. **`temp_pollution`:** $\text{Temperature} \times \text{pollution\_index} / 100$ — thermal-pollution interaction.
6. **`AQI_critical`:** Binary indicator $(\text{AQI} > 200)$.
7. **`AQI_unhealthy`:** Binary indicator $(100 < \text{AQI} \le 200)$.
8. **`PM25_high`:** Binary indicator $(\text{PM2.5} > 75\,\mu\text{g/m}^3)$.
9. **`symptom_severity`:** Ordinal GINA daytime symptom severity ($1\text{–}4$).
10. **`exposure_score`:** Ordinal poor-air exposure score ($1\text{–}3$).
11. **`night_score`:** Ordinal nocturnal dyspnea score ($0\text{–}3$).
12. **`trigger_count`:** Count of active patient-reported respiratory triggers.
13. **`clinical_risk_score`:** $0.4 \cdot \text{symptom\_severity} + 0.3 \cdot \text{exposure\_score} + 0.3 \cdot \text{night\_score}$.
14. **`env_risk_score`:** Weighted atmospheric hazard composite across AQI, PM2.5, and normalized `pollution_index`.
15. **`total_risk_interaction`:** $\text{clinical\_risk\_score} \times \text{env\_risk\_score}$ — synergistic clinical-environmental coupling.

---

## 🏥 4. Synthetic Pessimism & Independent External Clinical Cohorts (3,402 Patients)

The 2,000-sample development dataset was expanded from **201 real survey/clinical seed records** using KS-validated Gaussian perturbation ($\epsilon = 0.1\sigma$, all Kolmogorov-Smirnov tests $p > 0.05$) with deliberate boundary noise (**synthetic pessimism**) to harden decision boundaries. The 6-model stacking architecture was further validated on **two independent external clinical datasets totaling 3,402 patient records**:

| External Dataset | Origin & Authors | Cohort Nature | Sample Size | Evaluated Accuracy | F1-Score | Clinical & Fairness Conclusion |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **Zenodo Clinical Asthma Cohort** | Haque et al. (Zenodo Repository) | Diagnosed clinical asthma cohort with symptom & clinical markers. | **1,010** | **92.57%** | **0.9420** | Confirms strong performance on cleaner real-world clinical distributions. |
| **Kaggle Multi-Center Cohort** | Elkharoua et al. | Multi-center demographic, spirometry, and clinical dataset (80/20 stratified split). | **2,392** | **96.45%** | **0.9471** | Audited across Gender (Male 96.3%, Female 96.6%) and Age ($<18$: 93.9%, $18\text{–}65$: 98.0%, $>65$: 93.3%). |

---

## 🎯 5. Ground Truth Target Classification

The target risk classification represents clinical asthma exacerbation threat:

* **🟢 Low Risk (Controlled Asthma):** Patient exhibits infrequent daytime symptoms (<1/month), zero nocturnal dyspnea, and ambient air is within safe thresholds.
* **🟡 Medium Risk (Moderate Concern):** Moderate symptoms (1–2 times/month), intermittent triggers, or elevated urban particulate exposure (AQI 100–200). Patient advised to carry rescue inhaler and limit outdoor exertion.
* **🔴 High Risk (Acute Threat / Severe Flare-up):** Frequent attacks, acute nocturnal dyspnea, or hazardous particulate air pollution (AQI >200, PM2.5 >100 µg/m³). Triggers emergency alert modal, inhaler protocol, and SOS broadcasting.
