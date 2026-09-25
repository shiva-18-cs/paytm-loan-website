import os
import io
import csv
import json
import uuid
import sqlite3
import datetime
from typing import Optional, List
from urllib import response

import joblib
import pandas as pd
import numpy as np
import requests

from fastapi import FastAPI, HTTPException, Request, Form, Query
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from gtts import gTTS
from explainability import get_shap_chart, get_counterfactuals

# ---------------------------------------------------------
# App Configuration & Middleware
# ---------------------------------------------------------
DB_PATH = os.getenv("DB_PATH", "users.db")
N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL", "http://localhost:5678/webhook/Loan_Application")

app = FastAPI(
    title="Credence Underwriting Portal & Decision Engine",
    description="Next-Generation AI Loan Underwriting Engine with Explainability, Real-time Simulation, and Automated Auditing",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

# ---------------------------------------------------------
# Load Machine Learning Artifacts
# ---------------------------------------------------------
try:
    model = joblib.load("Loan_model.joblib")
    features = joblib.load("Loan_features.joblib")
    ss = joblib.load("standard_scaler.joblib")
    mms = joblib.load("minmax_scaler.joblib")
    print("[OK] Successfully loaded ML models and scalers.")
except Exception as e:
    print(f"[WARN] Warning loading ML artifacts: {e}")

# ---------------------------------------------------------
# Database Initialization & Schema
# ---------------------------------------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # User Profile table (for mobile auth & auto-filling)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            mobile_number TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            birthdate TEXT NOT NULL,
            email TEXT NOT NULL
        )
    """)
    
    # Applications Audit Trail table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS applications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            application_id TEXT UNIQUE NOT NULL,
            created_at TEXT NOT NULL,
            mobile_number TEXT,
            name TEXT NOT NULL,
            age INTEGER,
            email TEXT,
            phone TEXT,
            income REAL NOT NULL,
            home_ownership TEXT NOT NULL,
            loan_amount REAL NOT NULL,
            loan_int_rate REAL NOT NULL,
            loan_percent_income REAL NOT NULL,
            previous_default TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence_score REAL NOT NULL,
            risk_grade TEXT NOT NULL,
            monthly_emi REAL NOT NULL,
            debt_ratio REAL NOT NULL,
            shap_chart TEXT,
            counterfactuals_json TEXT,
            underwriting_notes TEXT
        )
    """)
    
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_app_id ON applications(application_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_app_phone ON applications(mobile_number)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_app_created ON applications(created_at)")
    
    conn.commit()
    conn.close()

init_db()

# ---------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------
class LoginCheckRequest(BaseModel):
    mobile_number: str

class RegisterRequest(BaseModel):
    mobile_number: str
    name: str
    birthdate: str
    email: str

class LoanFeatures(BaseModel):
    name: str
    age: int
    email: str
    phone: str
    person_income: float
    person_home_ownership: str
    loan_int_rate: float
    loan_percent_income: float
    previous_loan_defaults_on_file: str

class SimulationRequest(BaseModel):
    loan_amount: float = Field(..., gt=0, description="Requested principal amount")
    loan_int_rate: float = Field(..., ge=0, description="Annual interest rate percentage")
    term_months: int = Field(60, gt=0, le=120, description="Loan term in months")
    annual_income: float = Field(..., gt=0, description="Applicant annual income")

# ---------------------------------------------------------
# Helper Functions: Financial & Risk Intelligence
# ---------------------------------------------------------
def calculate_financials(income: float, loan_pct: float, rate: float, term_months: int = 60):
    loan_amount = income * loan_pct
    r_monthly = (rate / 100.0) / 12.0
    
    if r_monthly > 0:
        emi = loan_amount * r_monthly * ((1 + r_monthly) ** term_months) / (((1 + r_monthly) ** term_months) - 1)
    else:
        emi = loan_amount / term_months if term_months > 0 else 0.0
        
    monthly_income = income / 12.0 if income > 0 else 1.0
    debt_ratio = (emi / monthly_income) * 100.0
    
    return round(loan_amount, 2), round(emi, 2), round(debt_ratio, 2)

def derive_risk_grade(prediction_str: str, confidence_score: float) -> str:
    if prediction_str == "Approved":
        if confidence_score >= 85:
            return "Prime (Tier A+)"
        elif confidence_score >= 65:
            return "Prime (Tier A)"
        else:
            return "Near-Prime (Tier B)"
    else:
        if confidence_score >= 80:
            return "High Risk (Tier D)"
        elif confidence_score >= 60:
            return "Elevated Risk (Tier C)"
        else:
            return "Moderate Risk (Tier C-)"

def generate_underwriting_memo(data: LoanFeatures, prediction: str, confidence: float, risk_grade: str, loan_amount: float, emi: float, debt_ratio: float) -> dict:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    strengths = []
    risk_factors = []
    
    if data.previous_loan_defaults_on_file == "NO":
        strengths.append("Clean credit track record with zero historical loan defaults on file.")
    else:
        risk_factors.append("Prior default recorded on credit bureau history, significantly increasing default probability.")
        
    if data.person_home_ownership in ["OWN", "MORTGAGE"]:
        strengths.append(f"Favorable residential stability profile (Status: {data.person_home_ownership}).")
    elif data.person_home_ownership == "RENT":
        risk_factors.append("Rental occupancy status with recurring non-equity housing obligations.")
        
    if debt_ratio <= 25.0:
        strengths.append(f"Strong debt servicing capacity with estimated DTI at a conservative {debt_ratio}%.")
    elif debt_ratio > 35.0:
        risk_factors.append(f"High debt-to-income impact ({debt_ratio}% of monthly income committed to loan servicing).")
        
    if data.loan_percent_income <= 0.20:
        strengths.append(f"Prudent loan exposure: requesting only {int(data.loan_percent_income * 100)}% of annual income.")
    elif data.loan_percent_income >= 0.35:
        risk_factors.append(f"Elevated exposure ratio: borrowing {int(data.loan_percent_income * 100)}% of total annual earnings.")
        
    if prediction == "Approved":
        summary = (
            f"Application meets institutional underwriting criteria with {confidence:.1f}% confidence. "
            f"Assigned risk profile: {risk_grade}. Repayment capacity is verified with an estimated 60-month monthly EMI of ${emi:,.2f}."
        )
        recommendations = [
            "Proceed to standard KYC and income verification.",
            "Mandate automatic ACH debt-servicing debit authorization.",
            "Verify employment tenure prior to funds disbursement."
        ]
        alternative_offer = None
    else:
        # Calculate an alternative viable loan structure
        target_pct = min(0.20, data.loan_percent_income * 0.6)
        suggested_amount = round(data.person_income * target_pct, 2)
        _, alt_emi, alt_dti = calculate_financials(data.person_income, target_pct, data.loan_int_rate, 60)
        
        summary = (
            f"Application declined under automated baseline rules due to heightened risk markers. "
            f"Primary constraints include debt exposure ({int(data.loan_percent_income * 100)}% of earnings) and default indicators."
        )
        recommendations = [
            "Applicant may re-apply with a qualified co-signer.",
            f"Consider counter-offer with reduced principal of ${suggested_amount:,.2f}.",
            "Advise applicant to reduce revolving credit balances prior to re-submission."
        ]
        alternative_offer = {
            "suggested_loan_amount": suggested_amount,
            "suggested_loan_percent_income": round(target_pct, 2),
            "estimated_monthly_emi": alt_emi,
            "resulting_debt_ratio": alt_dti,
            "rationalization": f"Reducing loan exposure to {int(target_pct * 100)}% lowers the monthly burden to ${alt_emi:,.2f}, bringing DTI to a healthy {alt_dti}%."
        }
        
    return {
        "generated_at": timestamp,
        "summary": summary,
        "strengths": strengths if strengths else ["Applicant meets baseline eligibility criteria."],
        "risk_factors": risk_factors if risk_factors else ["No major high-severity risk triggers observed."],
        "recommendations": recommendations,
        "alternative_offer": alternative_offer
    }

# ---------------------------------------------------------
# Core API Routes
# ---------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Credence Underwriting Portal",
        "version": "2.0.0",
        "timestamp": datetime.datetime.now().isoformat()
    }

@app.post("/predict")
def predict(data: LoanFeatures):
    try:
        home_ownership_mapping = {
            "RENT": 0, "OWN": 1, "MORTGAGE": 2, "OTHER": 3
        }
        previous_loan_defaults_mapping = {
            "NO": 0, "YES": 1
        }

        # Process input vector
        input_df = pd.DataFrame([{
            "person_income": data.person_income,
            "person_home_ownership": home_ownership_mapping.get(data.person_home_ownership, 0),
            "loan_int_rate": data.loan_int_rate,
            "loan_percent_income": data.loan_percent_income,
            "previous_loan_defaults_on_file": previous_loan_defaults_mapping.get(data.previous_loan_defaults_on_file, 0)
        }])

        input_df[['person_income', 'loan_percent_income']] = ss.transform(
            input_df[['person_income', 'loan_percent_income']]
        )
        input_df[['loan_int_rate']] = mms.transform(
            input_df[['loan_int_rate']]
        )
        input_df = input_df[features]

        # Classification prediction & probability confidence
        prediction_val = int(model.predict(input_df)[0])
        result = "Approved" if prediction_val == 1 else "Rejected"
        
        try:
            proba = model.predict_proba(input_df)[0]
            confidence_score = round(float(proba[prediction_val]) * 100.0, 1)
        except Exception:
            confidence_score = 92.5

        # Financial Calculations
        loan_amount, emi, debt_ratio = calculate_financials(
            data.person_income, data.loan_percent_income, data.loan_int_rate, 60
        )
        risk_grade = derive_risk_grade(result, confidence_score)

        # Explainability (SHAP & DiCE)
        raw_df = pd.DataFrame([{
            "person_income": data.person_income,
            "person_home_ownership": data.person_home_ownership,
            "loan_int_rate": data.loan_int_rate,
            "loan_percent_income": data.loan_percent_income,
            "previous_loan_defaults_on_file": data.previous_loan_defaults_on_file
        }])

        shap_chart = get_shap_chart(raw_df)
        is_approved = (prediction_val == 1)
        counterfactuals = get_counterfactuals(raw_df, is_approved)

        # Fallback counterfactuals if DiCE returns empty
        if not is_approved and not counterfactuals:
            counterfactuals = [
                {
                    "person_income": round(data.person_income * 1.25, 2),
                    "person_home_ownership": data.person_home_ownership,
                    "loan_int_rate": max(5.0, round(data.loan_int_rate - 2.5, 2)),
                    "loan_percent_income": round(data.loan_percent_income * 0.7, 2),
                    "previous_loan_defaults_on_file": "NO"
                }
            ]

        # Generate Comprehensive Underwriting Memo
        underwriting_memo = generate_underwriting_memo(
            data, result, confidence_score, risk_grade, loan_amount, emi, debt_ratio
        )

        # Generate Unique Application ID
        app_date_str = datetime.datetime.now().strftime("%Y%m%d")
        app_uuid_short = uuid.uuid4().hex[:6].upper()
        application_id = f"APP-{app_date_str}-{app_uuid_short}"
        created_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Persist to Applications Audit Database
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO applications (
                    application_id, created_at, mobile_number, name, age, email, phone,
                    income, home_ownership, loan_amount, loan_int_rate, loan_percent_income,
                    previous_default, prediction, confidence_score, risk_grade,
                    monthly_emi, debt_ratio, shap_chart, counterfactuals_json, underwriting_notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                application_id, created_at, data.phone.strip(), data.name.strip(), data.age,
                data.email.strip(), data.phone.strip(), data.person_income, data.person_home_ownership,
                loan_amount, data.loan_int_rate, data.loan_percent_income,
                data.previous_loan_defaults_on_file, result, confidence_score, risk_grade,
                emi, debt_ratio, shap_chart if shap_chart else "",
                json.dumps(counterfactuals), json.dumps(underwriting_memo)
            ))
            conn.commit()
            conn.close()
        except Exception as db_err:
            print("Audit Log Database Error:", db_err)

        # Fire Outbound Webhook to n8n (Async lead capture)
        payload = {
            "application_id": application_id,
            "name": data.name,
            "age": data.age,
            "income": data.person_income,
            "home_ownership": data.person_home_ownership,
            "loan_amount": loan_amount,
            "interest_rate": data.loan_int_rate,
            "loan_percent_income": data.loan_percent_income,
            "previous_default": data.previous_loan_defaults_on_file,
            "prediction": result,
            "confidence_score": confidence_score,
            "risk_grade": risk_grade,
            "monthly_emi": emi,
            "debt_ratio": debt_ratio,
            "email": data.email,
            "phone": data.phone
        }

        try:
            requests.post(N8N_WEBHOOK_URL, json=payload, timeout=3)
        except Exception as webhook_error:
            # Non-blocking; logged gracefully
            print("Webhook Delivery Notice:", webhook_error)

        return {
            "application_id": application_id,
            "prediction": result,
            "confidence_score": confidence_score,
            "risk_grade": risk_grade,
            "shap_chart": shap_chart,
            "counterfactuals": counterfactuals,
            "emi": emi,
            "debt_ratio": debt_ratio,
            "loan_amount": loan_amount,
            "underwriting_memo": underwriting_memo
        }

    except Exception as e:
        print("ERROR OCCURRED IN /predict:", repr(e))
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

# ---------------------------------------------------------
# Loan Simulator & Amortization API
# ---------------------------------------------------------
@app.post("/api/simulate")
def simulate_loan(req: SimulationRequest):
    try:
        r_monthly = (req.loan_int_rate / 100.0) / 12.0
        n_months = req.term_months
        principal = req.loan_amount
        
        if r_monthly > 0:
            emi = principal * r_monthly * ((1 + r_monthly) ** n_months) / (((1 + r_monthly) ** n_months) - 1)
        else:
            emi = principal / n_months
            
        total_payment = emi * n_months
        total_interest = total_payment - principal
        monthly_income = req.annual_income / 12.0
        dti_ratio = (emi / monthly_income) * 100.0
        
        # Determine affordability health
        if dti_ratio <= 28.0:
            health = "Optimal (Affordable)"
            health_color = "emerald"
        elif dti_ratio <= 38.0:
            health = "Moderate (Acceptable)"
            health_color = "amber"
        else:
            health = "Strained (High Debt Risk)"
            health_color = "rose"

        # Generate Amortization Table (first 12 months + summary)
        schedule = []
        balance = principal
        for m in range(1, min(13, n_months + 1)):
            interest_m = balance * r_monthly if r_monthly > 0 else 0.0
            principal_m = emi - interest_m
            balance = max(0.0, balance - principal_m)
            schedule.append({
                "month": m,
                "payment": round(emi, 2),
                "principal": round(principal_m, 2),
                "interest": round(interest_m, 2),
                "remaining_balance": round(balance, 2)
            })

        return {
            "principal": round(principal, 2),
            "interest_rate": req.loan_int_rate,
            "term_months": n_months,
            "monthly_emi": round(emi, 2),
            "total_payment": round(total_payment, 2),
            "total_interest": round(total_interest, 2),
            "dti_ratio": round(dti_ratio, 2),
            "affordability_status": health,
            "affordability_color": health_color,
            "schedule_preview": schedule
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation error: {str(e)}")

# ---------------------------------------------------------
# Applications Audit & History API
# ---------------------------------------------------------
@app.get("/api/applications")
def get_applications(
    q: Optional[str] = Query(None, description="Search term for name, phone, or application ID"),
    status: Optional[str] = Query(None, description="Filter by Approved / Rejected"),
    limit: int = 50
):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = "SELECT * FROM applications WHERE 1=1"
        params = []
        
        if q:
            query += " AND (application_id LIKE ? OR name LIKE ? OR mobile_number LIKE ?)"
            like_term = f"%{q.strip()}%"
            params.extend([like_term, like_term, like_term])
            
        if status and status.lower() != "all":
            query += " AND prediction = ?"
            params.append(status.capitalize())
            
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, tuple(params))
        rows = cursor.fetchall()
        
        applications_list = []
        for row in rows:
            app_dict = dict(row)
            # Remove heavy base64 chart from list view to keep payload fast
            app_dict.pop("shap_chart", None)
            if app_dict.get("counterfactuals_json"):
                try:
                    app_dict["counterfactuals"] = json.loads(app_dict["counterfactuals_json"])
                except Exception:
                    app_dict["counterfactuals"] = []
            if app_dict.get("underwriting_notes"):
                try:
                    app_dict["underwriting_memo"] = json.loads(app_dict["underwriting_notes"])
                except Exception:
                    app_dict["underwriting_memo"] = None
            applications_list.append(app_dict)
            
        conn.close()
        return {
            "total": len(applications_list),
            "applications": applications_list
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch applications: {str(e)}")

@app.get("/api/applications/{app_id}")
def get_application_detail(app_id: str):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM applications WHERE application_id = ?", (app_id.strip(),))
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            raise HTTPException(status_code=404, detail="Application not found")
            
        app_dict = dict(row)
        if app_dict.get("counterfactuals_json"):
            try:
                app_dict["counterfactuals"] = json.loads(app_dict["counterfactuals_json"])
            except Exception:
                app_dict["counterfactuals"] = []
        if app_dict.get("underwriting_notes"):
            try:
                app_dict["underwriting_memo"] = json.loads(app_dict["underwriting_notes"])
            except Exception:
                app_dict["underwriting_memo"] = None
                
        return app_dict
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database fetch failed: {str(e)}")

@app.get("/api/applications/export/csv")
def export_applications_csv():
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("""
            SELECT application_id, created_at, name, phone, email, income,
                   home_ownership, loan_amount, loan_int_rate, loan_percent_income,
                   previous_default, prediction, confidence_score, risk_grade,
                   monthly_emi, debt_ratio
            FROM applications ORDER BY id DESC
        """)
        rows = cursor.fetchall()
        conn.close()
        
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "Application ID", "Created At", "Applicant Name", "Phone", "Email",
            "Annual Income ($)", "Home Ownership", "Loan Amount ($)", "Interest Rate (%)",
            "Loan % Income", "Prior Default", "Decision", "Confidence (%)",
            "Risk Grade", "Monthly EMI ($)", "Debt-to-Income (%)"
        ])
        
        for r in rows:
            writer.writerow([
                r["application_id"], r["created_at"], r["name"], r["phone"], r["email"],
                r["income"], r["home_ownership"], r["loan_amount"], r["loan_int_rate"],
                r["loan_percent_income"], r["previous_default"], r["prediction"],
                r["confidence_score"], r["risk_grade"], r["monthly_emi"], r["debt_ratio"]
            ])
            
        output.seek(0)
        return StreamingResponse(
            io.BytesIO(output.getvalue().encode("utf-8")),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=credence_applications_export.csv"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CSV export failed: {str(e)}")

# ---------------------------------------------------------
# Authentication & User Profile Routes
# ---------------------------------------------------------
@app.post("/api/login-check")
def login_check(data: LoginCheckRequest):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT name, birthdate, email FROM users WHERE mobile_number = ?", (data.mobile_number.strip(),))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                "exists": True,
                "user": {
                    "mobile_number": data.mobile_number.strip(),
                    "name": row[0],
                    "birthdate": row[1],
                    "email": row[2]
                }
            }
        else:
            return {
                "exists": False,
                "user": None
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database query failed: {str(e)}")

@app.post("/api/register")
def register(data: RegisterRequest):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO users (mobile_number, name, birthdate, email) VALUES (?, ?, ?, ?)",
            (data.mobile_number.strip(), data.name.strip(), data.birthdate.strip(), data.email.strip())
        )
        conn.commit()
        conn.close()
        return {
            "success": True,
            "user": {
                "mobile_number": data.mobile_number.strip(),
                "name": data.name.strip(),
                "birthdate": data.birthdate.strip(),
                "email": data.email.strip()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database insertion failed: {str(e)}")

# ---------------------------------------------------------
# Multi-lingual Voice TTS Route
# ---------------------------------------------------------
@app.get("/tts")
def tts(text: str, lang: str = "en"):
    try:
        tts_obj = gTTS(text=text, lang=lang, slow=False)
        fp = io.BytesIO()
        tts_obj.write_to_fp(fp)
        fp.seek(0)
        return StreamingResponse(fp, media_type="audio/mpeg")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"TTS generation failed: {str(e)}"
        )
