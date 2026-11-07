# HridyaVayu (AsthmAI) — Research Results & Empirical Evaluation Report

## Executive Summary

This report summarizes the empirical evaluation of the **HridyaVayu Neuro-Symbolic Asthma Risk Prediction System**. We engineered a **6-Model Stacking Ensemble** blended by a **Logistic Regression Meta-Learner** via **5-fold stratified cross-validation**, governed by **Shannon Entropy uncertainty gating ($H(P) \ge 1.0$)** and **GINA clinical safety heuristics**.

- **Development Dataset**: 2,000 multimodal samples (expanded from 201 clinical seed records via KS-validated synthetic pessimism; 1,400 train / 300 validation / 300 held-out test)
- **Feature Space**: **27 total features** (7 continuous environmental + 5 GINA clinical survey + 15 engineered non-linear interaction features)
- **6-Model Stacking Test Performance ($N=300$)**: **75.00% Accuracy** | **0.7451 Weighted F1** | **0.8488 Multi-Class ROC-AUC**
- **5-Fold Stratified Cross-Validation ($N=1,700$)**: **71.41% ± 2.07% Accuracy**
- **Low-Entropy High-Confidence Subset ($H < 0.5$)**: **96.67% Accuracy**
- **Entropy-Gated GINA Safety Hybrid**: Elevates High-Risk Recall from **79.07% → 82.17%** with **0 High-Risk patients misclassified as Low Risk**
- **Ablation Gain**: **+10.0 percentage points** accuracy gain when adding GINA clinical guidelines & engineered interactions to environmental telemetry ($58.5\% \rightarrow 68.5\%$ in 5-fold CV)
- **External Cohort Benchmarks ($N=3,402$)**:
  - **Zenodo Clinical Cohort (Haque et al., $N=1,010$)**: **92.57% Accuracy** | **0.9420 F1-Score**
  - **Kaggle Multi-Center Cohort (Elkharoua et al., $N=2,392$)**: **96.45% Accuracy** | **0.9471 Weighted F1-Score**

---

## 1. Dataset & Synthetic Pessimism Validation

| Attribute | Value |
| :--- | :--- |
| Seed Records | 201 clinical & survey records |
| Expanded Multimodal Dataset | 2,000 samples (`1,400` train / `300` val / `300` test) |
| Raw Features | 12 (7 environmental telemetry + 5 GINA clinical intake) |
| Engineered Features | 15 (`pollution_index`, `AQI_PM_ratio`, `clinical_risk_score`, `env_risk_score`, `total_risk_interaction`, etc.) |
| Total Model Input Dimension | **27 features** ($X \in \mathbb{R}^{27}$) |
| Class Distribution | Low: `153` (7.7%) • Medium: `991` (49.5%) • High: `856` (42.8%) |

- **Kolmogorov-Smirnov (KS) Test**: Passed across all 7 continuous atmospheric variables ($p > 0.05$).
- **Cohen's $d$ & Wasserstein Distance**: Negligible distributional drift between seed and expanded distributions.
- **Why Development Accuracy (~75%) < External Clinical Accuracy (92.6%–96.5%)**: During synthetic expansion, Gaussian boundary noise ($\epsilon = 0.1\sigma$) was injected (**synthetic pessimism**), creating harder overlapping class boundaries during training so the ensemble generalizes cleanly on lower-noise clinical cohorts.

---

## 2. Held-Out Test Comparison: 6 Base Estimators + Meta-Learner ($N=300$)

| Model | Role | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **XGBoost** | Level-1 Base Estimator | 73.00% | 0.7249 | 0.7300 | 0.7180 | 0.8363 |
| **LightGBM** | Level-1 Base Estimator | 74.00% | 0.7399 | 0.7400 | 0.7351 | 0.8379 |
| **Random Forest** | Level-1 Base Estimator | 72.67% | 0.7523 | 0.7267 | 0.7120 | 0.8222 |
| **Gradient Boosting** | Level-1 Base Estimator | 70.00% | 0.6959 | 0.7000 | 0.6958 | 0.8187 |
| **Extra Trees** | Level-1 Base Estimator | 71.33% | 0.7174 | 0.7133 | 0.7027 | 0.8172 |
| **MLP Neural Net ($128\times 64$)** | Level-1 Base Estimator | 74.33% | 0.7330 | 0.7433 | 0.7357 | 0.8385 |
| **StackingClassifier (Logistic Regression)** | **Level-2 Meta-Learner** | **75.00%** | **0.7438** | **0.7500** | **0.7451** | **0.8488** |

---

## 3. Ablation Study (`results/ablation_results.txt`)

| Feature Configuration | 5-Fold CV Accuracy | Weighted F1 | Takeaway |
| :--- | :---: | :---: | :--- |
| Environmental Features Only (7) | 58.5% ± 1.7% | 0.559 | Ambient air quality alone is insufficient without patient symptom context |
| GINA Clinical Survey Only (5) | 60.1% ± 1.0% | 0.576 | Clinical history alone misses acute environmental pollution spikes |
| Engineered Features Only (15) | 66.8% ± 1.2% | 0.657 | Composite indices (`pollution_index`, `total_risk_interaction`) capture non-linear synergy |
| **Full 27-Feature Multimodal Vector** | **68.5% ± 1.7%** | **0.676** | **+10.0% absolute gain over environmental-only baseline** |

---

## 4. Shannon Entropy Gating & Demographic Fairness Audit

- **Shannon Entropy Triage**: $H(P) = -\sum_{c} P(c) \log_2 P(c) \in [0, 1.585]$ bits.
  - When $H(P) < 0.5$ (high ML certainty), stacking accuracy is **96.67%**.
  - When $H(P) \ge 1.0$ (uncertain borderline predictions), GINA clinical rules govern the decision, increasing High-Risk recall from **79.07% to 82.17%** and ensuring **zero High-Risk patients are misclassified as Low Risk**.
- **Demographic Subgroup Audit (Kaggle $N=2,392$, Test $N=479$)**:
  - **Gender**: Male **96.3%** ($n=241$) vs. Female **96.6%** ($n=238$) — gap **0.3%**, Equalized Odds Ratio **0.942**.
  - **Age**: $<18$: **93.9%** • $18\text{–}45$: **97.7%** • $46\text{–}65$: **98.5%** • $>65$: **93.3%** (max subgroup spread **5.2%** across pediatric/geriatric extremes).
- **Explainability**: TreeSHAP / KernelSHAP feature attribution (`figures/shap_summary.png`, `figures/shap_importance.png`) with optional LIME local tabular explanations supported in `research/explainability.py`.
