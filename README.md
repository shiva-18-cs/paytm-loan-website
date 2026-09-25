# 🏦 Credence 2.0 — Next-Gen AI Loan Underwriting & Decision Operating System

An institutional-grade credit decision engine combining machine learning (`KNeighborsClassifier`), Explainable AI (SHAP & DiCE), interactive loan amortization simulation, real-time auditing ledgers, a 3-language conversational voice assistant, and outbound webhook automation.

---

## 🌟 Key Innovations & Features

1. **AI Underwriting Engine & Real-Time Scoring**:
   - Automated approval classification with model probability confidence scores (e.g. 94.2%).
   - Dynamic Risk Tier categorization (`Prime A+`, `Prime A`, `Near-Prime B`, `Subprime C`, `High Risk D`).
   - Automated Application ID generation (`APP-YYYYMMDD-XXXX`).

2. **Explainable AI (XAI)**:
   - **SHAP Feature Attributions**: Generates custom visual bar charts showing how each parameter influenced the approval boundary.
   - **DiCE Counterfactual Scenarios**: Identifies the 3 closest continuous parameter changes required to qualify for approval.

3. **AI Underwriting Memorandum & Counter-Offers**:
   - Automated Executive Credit Summary outlining risk drivers, financial strengths, and compensating factors.
   - **Automated Viable Counter-Offer**: For declined applications, calculates an adjusted loan amount, reduced EMI, and viable DTI ratio.
   - **Print-Ready Institutional Credit Memo**: Print or export formal PDF underwriting memorandums with full applicant details and sign-off blocks.

4. **Interactive Loan Scenario Simulator & Amortization**:
   - Real-time sliders for Loan Principal ($1,000–$250,000), Annual Income, Tenure (12 to 84 months), and Interest Rate.
   - Instant calculation of Monthly EMI, Total Interest, Total Cost, and Debt-to-Income (DTI) ratio.
   - Full 12-month Amortization Schedule table.
   - "Apply to Loan Form" button to transfer simulated parameters directly to the underwriting engine.

5. **Applications Audit Trail & Historical Ledger**:
   - Persistent SQLite storage for every underwriting decision.
   - Real-time searchable and filterable audit dashboard (filter by status, name, phone, or application ID).
   - "View Memo" inspection for past applications.
   - One-click CSV export (`/api/applications/export/csv`).

6. **Conversational 3-Language Voice Wizard**:
   - 12-step guided voice application supporting **English, Hindi, and Marathi**.
   - Speech-to-text recognition and dynamic gTTS / browser synthesis.

---

## 🛠️ Technology Stack

- **Backend**: FastAPI, Uvicorn, Pydantic, Jinja2, Requests, SQLite3
- **Machine Learning**: Scikit-Learn (`KNeighborsClassifier`), Joblib, Pandas, NumPy
- **Explainability**: SHAP (KernelExplainer), DiCE-ML, Matplotlib
- **Speech**: HTML5 Web Speech API, Google Text-to-Speech (gTTS)
- **Frontend**: Semantic HTML5, Vanilla CSS3 (Custom design system with dark/light themes), Vanilla JavaScript
- **DevOps & Containers**: Docker, Docker Compose, Railway (`railway.json`), Render (`render.yaml`)

---

## 🚀 Running Locally

### Option 1: Direct Python Virtual Environment

```bash
# 1. Create and activate virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch FastAPI server with hot-reload
uvicorn main:app --reload --port 8000
```
Open **[http://localhost:8000](http://localhost:8000)** in your browser.

---

### Option 2: Docker & Docker Compose

```bash
# Build and run containerized service
docker-compose up --build
```
Access the application at **[http://localhost:8000](http://localhost:8000)**. Health checks run automatically at `/health`.

---

## 🌐 Cloud Deployment Options

### Deploy to Render
1. Push this project to your new GitHub repository.
2. Log in to [Render](https://render.com) and click **New > Blueprint**.
3. Select your repository — Render will automatically read [`render.yaml`](file:///render.yaml) and configure the build and start commands with health checks.

### Deploy to Railway
1. Push this project to your new GitHub repository.
2. In [Railway](https://railway.app), click **New Project > Deploy from GitHub repo**.
3. Railway automatically detects [`railway.json`](file:///railway.json) and starts the service.

---

## 📦 Connecting to Your New GitHub Repository

To push this upgraded codebase to your own new GitHub repository:

```bash
# 1. Stage all files
git add .

# 2. Commit the new release
git commit -m "feat: Credence 2.0 release with simulator, audit trail, AI memo, and multi-deployment"

# 3. Add your new GitHub repository remote (replace with your repo URL)
git remote add origin https://github.com/<YOUR_USERNAME>/<YOUR_NEW_REPO_NAME>.git

# 4. Push to main branch
git branch -M main
git push -u origin main
```

---

## 🔗 Key API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Web application portal & dashboard |
| `POST` | `/predict` | Primary credit underwriting prediction with SHAP, DiCE & Memo |
| `POST` | `/api/simulate` | Interactive loan simulation & amortization schedule |
| `GET` | `/api/applications` | Query audit trail of submitted loan applications |
| `GET` | `/api/applications/{app_id}` | Retrieve individual application memo & telemetry |
| `GET` | `/api/applications/export/csv` | Download complete applications database as CSV |
| `POST` | `/api/login-check` | Verify mobile number registration |
| `POST` | `/api/register` | Register new underwriter profile |
| `GET` | `/tts` | Multi-lingual audio stream generation |
| `GET` | `/health` | Cloud deployment healthcheck probe |
