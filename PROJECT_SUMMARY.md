# 🏦 Credence Underwriting Portal — Comprehensive Project Specification & Summary

The **Credence Loan Underwriting Portal** is an AI-powered financial platform designed for automated credit decisioning. It integrates a machine learning model, live model interpretability (SHAP & DiCE), a multi-lingual conversational voice assistant wizard, secure database persistence, and outbound automation hooks.

---

## 🛠️ 1. Complete Technical Stack

*   **Frontend Interface & Layout:**
    *   Core structure built with semantic **HTML5** and styled with premium **Vanilla CSS3** (no external framework dependencies like Tailwind or Bootstrap).
    *   Implements a hybrid dark/light dashboard theme using variable-driven HSL color-mapping, floating container wrappers, card elevations, and custom vector icons.
    *   **Typography:** Elegant serif `Lora` for visual accents and headers, and geometric sans-serif `Outfit` for body text and telemetry data.
*   **Backend Server:**
    *   **FastAPI** ASGI web framework providing fast performance and automated OpenAPI/Swagger interactive documentation.
    *   **Uvicorn** ASGI server running with hot-reloading for local development.
*   **Machine Learning Suite:**
    *   **Scikit-Learn:** Core classifier pipeline utilizing `KNeighborsClassifier` to determine loan suitability.
    *   **Joblib:** Serialization tools used to load the pre-trained KNN model (`Loan_model.joblib`), target feature indices (`Loan_features.joblib`), and scaling functions.
    *   **Pandas & NumPy:** In-memory data manipulation, column-mapping, and feature array processing.
*   **Explainability (XAI) Suite:**
    *   **SHAP (Shapley Additive exPlanations):** Computes feature importance using a background-subsampled `KernelExplainer`, rendering a custom feature contribution bar chart detailing how each parameter shifted the decision boundary.
    *   **DiCE (Diverse Counterfactual Explanations):** Generates 3 alternative, close-distance scenarios showing rejected applicants the minimum continuous parameter modifications required to reverse their decision.
*   **Voice Assistant Engine:**
    *   **HTML5 Web Speech API (`webkitSpeechRecognition`):** Translates real-time vocal audio inputs to text inside the browser context.
    *   **gTTS (Google Text-to-Speech):** Python backend integration generating audio files of questions dynamically.
    *   **Speech Synthesis Fallback:** Browser-native TTS API fallback used if the host machine experiences connectivity dropouts.
*   **Database Engine:**
    *   **SQLite:** Zero-configuration, file-based relational database (`users.db`) managing login checks and underwriter registration.
*   **Outbound Automation:**
    *   **n8n:** Open-source automation tool receiving real-time JSON webhooks to run external notification chains (Email, SMS via Twilio, or Google Sheets archiving).

---

## ⚙️ 2. Problems Solved

1.  **Friction in Automated Underwriting:**
    Traditional portals require tedious typing across dozens of fields. Credence solves this by offering a conversational, 12-step guided **Voice Assistant** that speaks questions and listens to responses in three regional languages.
2.  **The Explainability "Black Box":**
    Regulators and customers demand transparency. Credence generates live Shapley plots and counterfactuals so that when an applicant is rejected, they are immediately told *exactly why* (e.g. "Because loan percentage of income was too high") and *how to fix it* (e.g. "Increase annual income by $8,000").
3.  **Real-Time Debt Calculations:**
    Instead of forcing manual calculations, the dashboard runs dynamic amortization equations as the user inputs data. It computes the **Estimated Loan Amount**, **Monthly EMI (5-year term)**, and **Monthly Debt-to-Income (DTI)** ratio, alerting the operator with visual risk levels (Green $\le$ 30%, Orange 30% - 40%, Red > 40%).
4.  **Underwriting System Localization:**
    Full support for **English, Hindi, and Marathi**. Clicking a language toggle instantly translates all labels, metrics, placeholder tags, and spoken questions.
5.  **Disconnected Lead Management:**
    Underwriting data often gets lost in browser sessions. The portal connects predictions to a live webhook listener hosted by **n8n**, piping all user entries directly to external notification integrations.

---

## 📊 3. Machine Learning & Feature Engineering Pipeline

The model runs a **K-Nearest Neighbors (KNN)** classification pipeline trained on a dataset of 45,000 customer loan records (`loan_data__2_.csv`).

### A. Performance & Accuracy Telemetry
*   **Dataset Shape:** 45,000 rows, 14 columns
*   **Target Distribution (`loan_status`):**
    *   `0` (Rejected): 35,000 instances
    *   `1` (Approved): 10,000 instances
*   **Overall Accuracy:** **93.82%** (0.93818)

#### **Classification Report:**
| Target Class | Outcome | Precision | Recall | F1-Score | Support |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **0** | Rejected | 0.96 | 0.97 | 0.96 | 35,000 |
| **1** | Approved | 0.88 | 0.84 | 0.86 | 10,000 |
| **Accuracy** | — | — | — | **0.94** | 45,000 |

#### **Confusion Matrix:**
$$
\begin{pmatrix} 
\text{True Negatives (Correct Rejections): } 33,804 & \text{False Positives (Incorrect Approvals): } 1,196 \\ 
\text{False Negatives (Incorrect Rejections): } 1,586 & \text{True Positives (Correct Approvals): } 8,414 
\end{pmatrix}
$$

### B. Feature Engineering & Mappings
The model expects a 5-element feature vector. Raw inputs undergo mappings and scaling before prediction:

1.  **Categorical Mapping (`person_home_ownership`):**
    Text inputs are ordinal-encoded into integer indexes:
    $$\text{RENT} \rightarrow 0,\quad \text{OWN} \rightarrow 1,\quad \text{MORTGAGE} \rightarrow 2,\quad \text{OTHER} \rightarrow 3$$
2.  **Binary Mapping (`previous_loan_defaults_on_file`):**
    Historical credit records are mapped to binary values:
    $$\text{NO} \rightarrow 0,\quad \text{YES} \rightarrow 1$$
3.  **Continuous Feature Scaling:**
    KNN is distance-sensitive (Euclidean distance). Unscaled features with large ranges (like income) would skew calculations. Scalers are loaded dynamically:
    *   **StandardScaler (`standard_scaler.joblib`):** Applied to `person_income` and `loan_percent_income`.
        Formula:
        $$z = \frac{x - \mu}{\sigma}$$
    *   **MinMaxScaler (`minmax_scaler.joblib`):** Applied to `loan_int_rate` to scale values between `0.0` and `1.0`.
        Formula:
        $$x_{scaled} = \frac{x - x_{min}}{x_{max} - x_{min}}$$

---

## 🎙️ 4. Conversational 3-Language Speech Assistant Engine

The **Voice Assistant Mode** guides users through the underwriting process using a step-by-step wizard.

```
[Start Wizard] ──> [Ask Input Q in EN/HI/MR] ──> [TTS Speaks Q] ──> [STT Record Answer] ──> [Parse Answer] ──> [Next Step]
```

### A. Speech-to-Text Parsing & Mapping Logic
The voice assistant handles spelling variations and accent patterns across three languages:

*   **Numeric Value Parsing:**
    *   **English:** Parsed via natural phrases (e.g. `"fifty thousand"`, `"one hundred thousand"`).
    *   **Hindi:** Handles numeric terms (e.g. `"पाँच लाख"` $\rightarrow 500,000$, `"दोन लाख"` $\rightarrow 200,000$).
    *   **Marathi:** Recognizes local regional words (e.g. `"अकरा"` $\rightarrow 11$, `"तीस हजार"` $\rightarrow 30,000$).
*   **Housing Keyword Parsing:**
    *   Recognizes spoken patterns in English, Hindi, and Marathi:
        *   `RENT` $\leftarrow$ `"किराया"`, `"भाड्याने"`, `"rent"`, `"rented"`.
        *   `OWN` $\leftarrow$ `"own"`, `"owners"`, `"खुद का"`, `"स्वतःचे"`.
        *   `MORTGAGE` $\leftarrow$ `"mortgage"`, `"गहाण"`, `"बंधक"`.
*   **Credit Default Keyword Parsing:**
    *   Maps spoken confirmations:
        *   `YES` $\leftarrow$ `"yes"`, `"हाँ"`, `"होय"`, `"या".`
        *   `NO` $\leftarrow$ `"no"`, `"नहीं"`, `"नाही"`, `"नको"`.

---

## 🔑 5. Relational Database & Session Persistence

The backend uses a local **SQLite** database (`users.db`) to manage credentials and pre-fill form data.

### Database Schema
```sql
CREATE TABLE IF NOT EXISTS users (
    mobile_number TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    birthdate TEXT NOT NULL,
    email TEXT NOT NULL
);
```

### Authentication Lifecycle
1.  On portal access, a login window prompts the operator for a **Mobile Number**.
2.  The frontend issues a `POST` request to `/api/login-check`.
3.  If the record is found, the user profile is returned, and their session details (Name, Email, Phone) are saved in the browser's `localStorage` to persist across refreshes.
4.  If the number is unrecognized, the app displays a registration form. Once completed, the data is saved to SQLite, automatically filling the application fields.

---

## 🔄 6. End-to-End System Integration Flow

```mermaid
sequenceDiagram
    autonumber
    actor U as Underwriter / Applicant
    participant C as Web UI Client (Browser)
    participant B as FastAPI Backend (Uvicorn)
    participant DB as SQLite DB (users.db)
    participant X as Explainability (SHAP & DiCE)
    participant N as n8n Automation Server

    U->>C: Input Mobile Number / Click Login
    C->>B: POST /api/login-check
    B->>DB: Query by mobile_number
    DB-->>B: Return user profile data
    B-->>C: Return user object (Name, Email, etc.)
    Note over C: Session stored in localStorage.<br/>Fills Name, Email, & Phone inputs.
    
    U->>C: Fill Loan Inputs / Converse with Voice Assistant
    Note over C: Web speech synthesizes query;<br/>STT records response & parses values.
    
    C->>B: POST /predict (data payload)
    activate B
    B->>B: Map Ordinal Categories
    B->>B: Standardize & MinMax Scale Features
    B->>B: Run KNN classification
    B->>X: Run SHAP KernelExplainer (calculates feature impact)
    B->>X: Run DiCE counterfactuals (finds optimal parameter path)
    B->>N: POST /webhook/Loan_Application (JSON Lead payload)
    Note over N: n8n triggers outbound alerts<br/>(Gmail notifications / SMS / Twilio)
    B-->>C: Return response (Prediction, SHAP base64 chart, DiCE recommendations, EMI/DTI metrics)
    deactivate B
    C-->>U: Render decision dashboard, XAI metrics, and counterfactuals
```

---

## 🚀 7. Local Server Access

Both servers are active:

*   **FastAPI Underwriting Portal:** **[http://127.0.0.1:8000/](http://127.0.0.1:8000/)**
*   **n8n Workflow Automation Console:** **[http://localhost:5678/](http://localhost:5678/)**

### Active Outbound Webhook Integration:
All loan queries automatically trigger a payload delivery to:
`http://localhost:5678/webhook/Loan_Application`
