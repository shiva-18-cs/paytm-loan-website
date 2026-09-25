# Credence 2.0 — Implementation Notes & Pre-Implementation Audit

This document captures the inspection findings of the existing Credence codebase prior to implementing the 7 requested features.

---

## 1. Existing Architecture
- **Framework**: FastAPI (Python 3.11+) with Uvicorn ASGI server.
- **Frontend**: Single-page application served via Jinja2 (`templates/index.html`) using semantic HTML5, glassmorphic Vanilla CSS, and Vanilla JavaScript (ES6+). Tab-based navigation across Underwriting, Voice Assistant, Loan Simulator, and Audit Log.
- **Explainability**: Integrated in `explainability.py` using `shap.KernelExplainer` for feature attributions and `dice_ml.Dice` for counterfactual recommendations.
- **ML Artifacts**:
  - `Loan_model.joblib`: Pre-trained classifier (KNeighborsClassifier).
  - `Loan_features.joblib`: Ordered feature list: `['person_income', 'person_home_ownership', 'loan_int_rate', 'loan_percent_income', 'previous_loan_defaults_on_file']`.
  - `standard_scaler.joblib`: StandardScaler for `person_income` and `loan_percent_income`.
  - `minmax_scaler.joblib`: MinMaxScaler for `loan_int_rate`.
- **Database**: SQLite3 (`users.db`).
- **Integrations**: Async HTTP POST webhook to external orchestration (`N8N_WEBHOOK_URL`).

---

## 2. Existing Database Schema
- **`users` Table**:
  - `mobile_number TEXT PRIMARY KEY`
  - `name TEXT NOT NULL`
  - `birthdate TEXT NOT NULL`
  - `email TEXT NOT NULL`
- **`applications` / `audit_logs` Table**:
  - `id INTEGER PRIMARY KEY AUTOINCREMENT`
  - `application_id TEXT UNIQUE NOT NULL`
  - `created_at TEXT NOT NULL`
  - `mobile_number TEXT`
  - `name TEXT NOT NULL`
  - `age INTEGER`
  - `email TEXT`
  - `phone TEXT`
  - `income REAL NOT NULL`
  - `home_ownership TEXT NOT NULL`
  - `loan_amount REAL NOT NULL`
  - `loan_int_rate REAL NOT NULL`
  - `loan_percent_income REAL NOT NULL`
  - `previous_default TEXT NOT NULL`
  - `prediction TEXT NOT NULL`
  - `confidence_score REAL NOT NULL`
  - `risk_grade TEXT NOT NULL`
  - `monthly_emi REAL NOT NULL`
  - `debt_ratio REAL NOT NULL`
  - `shap_chart TEXT` (base64 PNG)
  - `counterfactuals_json TEXT` (JSON)
  - `underwriting_notes TEXT` (JSON underwriting memo)
  - `pooled BOOLEAN DEFAULT 0` (to be added via idempotent migration)

---

## 3. Existing ML Input Features
The ML pipeline expects exactly 5 features in this order:
1. `person_income` (float): Annual income in local currency.
2. `person_home_ownership` (int encoded): `RENT` -> 0, `OWN` -> 1, `MORTGAGE` -> 2, `OTHER` -> 3.
3. `loan_int_rate` (float): Annual interest rate percentage.
4. `loan_percent_income` (float): Ratio of loan amount to annual income (e.g. 0.20 = 20%).
5. `previous_loan_defaults_on_file` (int encoded): `NO` -> 0, `YES` -> 1.

---

## 4. Existing `/predict` Flow
1. Receives `LoanFeatures` payload (`name`, `age`, `email`, `phone`, `person_income`, `person_home_ownership`, `loan_int_rate`, `loan_percent_income`, `previous_loan_defaults_on_file`).
2. Encodes categoricals (`home_ownership`, `previous_loan_defaults_on_file`).
3. Normalizes numerical features via `standard_scaler.joblib` and `minmax_scaler.joblib`.
4. Executes model inference via `model.predict()` and `model.predict_proba()`.
5. Calculates financial metrics: loan amount = `person_income * loan_percent_income`, monthly EMI, debt ratio (DTI %).
6. Generates SHAP waterfall chart (base64 PNG) and DiCE counterfactuals.
7. Generates comprehensive AI Underwriting Memorandum.
8. Writes record into database.
9. Dispatches non-blocking webhook payload to `N8N_WEBHOOK_URL`.
10. Returns prediction response to client.

---

## 5. Existing `/simulate` Flow
- Endpoint: `POST /api/simulate` (and alias `POST /simulate`).
- Accepts `SimulationRequest`: `loan_amount`, `loan_int_rate`, `term_months`, `annual_income`.
- Calculates monthly EMI using the standard amortization formula.
- Calculates DTI ratio: `(monthly_emi / (annual_income / 12)) * 100`.
- Categorizes affordability health (`Optimal`, `Moderate`, `Strained`).
- Generates 12-month amortization schedule breakdown.
- Returns financial summary and schedule without writing to the database.

---

## 6. Existing SHAP Implementation
- In `explainability.py`: `shap_explainer = shap.KernelExplainer(predict_approved_proba, X_bg_small)`.
- Generates horizontal bar chart with green (positive) / red (negative) bars.
- Renders to base64 PNG string returned as `shap_chart`.

---

## 7. Existing DiCE Implementation
- In `explainability.py`: `dice_explainer = dice_ml.Dice(d, m, method="random")`.
- `get_counterfactuals(input_df, is_approved)` generates 3 actionable scenarios by varying `person_income`, `loan_int_rate`, and `loan_percent_income` while keeping fixed features intact.

---

## 8. Existing Webhook Dispatcher
- Dispatches outbound HTTP POST to `N8N_WEBHOOK_URL` with a 3-second timeout.
- Catches exceptions gracefully without interrupting the underwriting workflow.

---

## 9. Unavoidable Assumptions
1. `loan_id` in `repayments`, `restructure_offers`, and `grievances` can match either `audit_logs.id` (numeric integer) or `audit_logs.application_id` (string `APP-YYYYMMDD-XXXXXX`). We support lookup by either.
2. In Feature 3 (Co-Applicant), household pooling combines the primary applicant's annual income with the co-applicant's income: `household_income = person_income + co_applicant_income`. Loan amount remains the same, so `loan_percent_income = loan_amount / household_income`.
3. In Feature 7 (Anonymous Simulation), the rate limiter tracks IP addresses via client host headers with an in-memory sliding window / count reset every hour or capped at `ANONYMOUS_SIMULATION_LIMIT=10` per session.
