import os
import json
import uuid
import sqlite3
import datetime
import requests
from typing import Optional, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

import pandas as pd
from explainability import get_counterfactuals

DB_PATH = os.getenv("DB_PATH", "users.db")
N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL", "http://localhost:5678/webhook/Loan_Application")

router = APIRouter(prefix="/restructure", tags=["Restructuring"])

class RestructureRespondRequest(BaseModel):
    offer_id: str = Field(..., description="UUID of the restructuring offer")
    status: Optional[str] = Field(None, description="Must be 'accepted' or 'declined'")
    decision: Optional[str] = Field(None, description="Alias for status")

def get_db():
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    conn.row_factory = sqlite3.Row
    return conn

def init_restructure_tables():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS restructure_offers (
            id TEXT PRIMARY KEY,
            loan_id TEXT NOT NULL,
            offer_json TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            responded_at TEXT
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_restr_loan ON restructure_offers(loan_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_restr_status ON restructure_offers(status)")
    conn.commit()
    conn.close()

init_restructure_tables()

def calculate_amortized_emi(principal: float, annual_rate: float, tenure_months: int) -> float:
    r = (annual_rate / 100.0) / 12.0
    if r > 0 and tenure_months > 0:
        emi = principal * r * ((1 + r) ** tenure_months) / (((1 + r) ** tenure_months) - 1)
    else:
        emi = principal / tenure_months if tenure_months > 0 else 0.0
    return round(emi, 2)

@router.post("/suggest/{loan_id}")
def suggest_restructuring(loan_id: str):
    """
    Evaluates distress on the loan and generates 2-3 mathematically valid restructuring
    options varying tenure and EMI, leveraging the existing DiCE explainability engine.
    """
    conn = get_db()
    cursor = conn.cursor()

    # 1. Fetch latest repayment / distress state
    cursor.execute("""
        SELECT updated_income, updated_dti, distress_flag, emi_due
        FROM repayments
        WHERE loan_id = ?
        ORDER BY cycle_date DESC, id DESC LIMIT 1
    """, (loan_id.strip(),))
    latest_repayment = cursor.fetchone()

    # 2. Fetch loan & applicant profile from audit_logs / applications
    app_data = None
    for tbl in ["audit_logs", "applications"]:
        try:
            cursor.execute(f"""
                SELECT * FROM {tbl}
                WHERE id = ? OR application_id = ?
                LIMIT 1
            """, (loan_id.strip(), loan_id.strip()))
            row = cursor.fetchone()
            if row:
                app_data = dict(row)
                break
        except Exception:
            pass

    if not app_data:
        conn.close()
        raise HTTPException(
            status_code=404,
            detail=f"Loan reference '{loan_id}' not found in audit ledger."
        )

    # Determine baseline financial parameters
    annual_income = float(latest_repayment["updated_income"]) if (latest_repayment and latest_repayment["updated_income"]) else float(app_data.get("income", 50000.0))
    monthly_income = (annual_income / 12.0) if annual_income > 15000 else annual_income
    if monthly_income <= 0:
        monthly_income = 1.0

    current_dti = float(latest_repayment["updated_dti"]) if latest_repayment else float(app_data.get("debt_ratio", 45.0))
    current_emi = float(latest_repayment["emi_due"]) if latest_repayment else float(app_data.get("monthly_emi", 1200.0))
    current_rate = float(app_data.get("loan_int_rate", 10.5))
    loan_amount = float(app_data.get("loan_amount", current_emi * 48))

    # Check distress status
    distress_active = False
    if latest_repayment and latest_repayment["distress_flag"]:
        distress_active = True
    elif current_dti >= 40.0:
        distress_active = True

    # 3. Call existing DiCE engine for counterfactual targets
    # Keeping applicant demographic profile fixed
    profile_df = pd.DataFrame([{
        "person_income": annual_income,
        "person_home_ownership": app_data.get("home_ownership", "RENT"),
        "loan_int_rate": current_rate,
        "loan_percent_income": round(loan_amount / annual_income, 2) if annual_income > 0 else 0.25,
        "previous_loan_defaults_on_file": app_data.get("previous_default", "NO")
    }])

    dice_scenarios = []
    try:
        dice_scenarios = get_counterfactuals(profile_df, is_approved=False)
    except Exception as dice_err:
        print("[WARN] DiCE restructuring guidance error:", dice_err)

    # 4. Generate 2-3 Mathematically Valid Restructuring Options
    options = []

    # Option 1: Extended Tenure (Stretch to 72 months to immediately lower EMI)
    opt1_tenure = 72
    opt1_emi = calculate_amortized_emi(loan_amount, current_rate, opt1_tenure)
    opt1_dti = round((opt1_emi / monthly_income) * 100.0, 2)
    options.append({
        "option": "Extended Tenure Relief",
        "new_tenure": opt1_tenure,
        "new_emi": opt1_emi,
        "resulting_dti": opt1_dti,
        "explanation": f"Extending repayment term to {opt1_tenure} months decreases monthly EMI by ₹{max(0, round(current_emi - opt1_emi)):,.2f}, bringing DTI down to {opt1_dti}%."
    })

    # Option 2: Step-Down Restructure (Moderate extension to 60 months with 1.5% interest concession)
    opt2_rate = max(6.0, round(current_rate - 1.5, 2))
    opt2_tenure = 60
    opt2_emi = calculate_amortized_emi(loan_amount, opt2_rate, opt2_tenure)
    opt2_dti = round((opt2_emi / monthly_income) * 100.0, 2)
    options.append({
        "option": "Rate Concession & Term Realignment",
        "new_tenure": opt2_tenure,
        "new_emi": opt2_emi,
        "resulting_dti": opt2_dti,
        "explanation": f"Lowering annual rate to {opt2_rate}% over {opt2_tenure} months aligns debt servicing with regulatory safe-harbor standards ({opt2_dti}% DTI)."
    })

    # Option 3: DiCE-informed Target Restructure (Optimized for maximum clearance)
    if dice_scenarios:
        best_cf = dice_scenarios[0]
        opt3_rate = float(best_cf.get("loan_int_rate", max(5.5, current_rate - 2.0)))
    else:
        opt3_rate = max(5.0, round(current_rate - 2.5, 2))
        
    opt3_tenure = 84
    opt3_emi = calculate_amortized_emi(loan_amount, opt3_rate, opt3_tenure)
    opt3_dti = round((opt3_emi / monthly_income) * 100.0, 2)
    options.append({
        "option": "Comprehensive Rehabilitation Plan",
        "new_tenure": opt3_tenure,
        "new_emi": opt3_emi,
        "resulting_dti": opt3_dti,
        "explanation": f"Maximizes household cashflow headroom with an 84-month amortization and optimized {opt3_rate}% APR, achieving an ultra-safe {opt3_dti}% DTI."
    })

    # 5. Persist to restructure_offers table
    offer_id = str(uuid.uuid4())
    created_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Enrich options with frontend-compatible aliases
    enriched_options = []
    for opt in options:
        item = dict(opt)
        item["offer_id"] = offer_id
        item["new_monthly_emi"] = opt["new_emi"]
        item["tenure_extension_months"] = opt["new_tenure"]
        item["rationalization"] = opt["explanation"]
        item["moratorium_months"] = 0
        item["interest_rate_concession"] = 1.5 if "Rate Concession" in opt["option"] else 0.0
        enriched_options.append(item)

    cursor.execute("""
        INSERT INTO restructure_offers (
            id, loan_id, offer_json, status, created_at, responded_at
        ) VALUES (?, ?, ?, ?, ?, ?)
    """, (
        offer_id,
        loan_id.strip(),
        json.dumps(enriched_options),
        "pending",
        created_at,
        None
    ))
    conn.commit()
    conn.close()

    return {
        "offer_id": offer_id,
        "loan_id": loan_id,
        "distress_active": distress_active,
        "current_dti": current_dti,
        "current_emi": current_emi,
        "current_monthly_emi": current_emi,
        "options": enriched_options,
        "offers": enriched_options,
        "status": "pending",
        "created_at": created_at
    }

@router.post("/respond")
def respond_restructure(data: RestructureRespondRequest):
    valid_statuses = ["accepted", "declined"]
    raw_status = data.status or data.decision or ""
    chosen_status = raw_status.strip().lower()
    if chosen_status not in valid_statuses:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status '{raw_status}'. Must be one of {valid_statuses}."
        )

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM restructure_offers WHERE id = ?", (data.offer_id.strip(),))
    row = cursor.fetchone()

    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Restructure offer not found.")

    offer_data = dict(row)
    responded_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute("""
        UPDATE restructure_offers
        SET status = ?, responded_at = ?
        WHERE id = ?
    """, (chosen_status, responded_at, data.offer_id.strip()))
    conn.commit()
    conn.close()

    # Reuse existing webhook dispatcher to notify applicant / downstream banking
    webhook_payload = {
        "event": "restructure_response",
        "offer_id": data.offer_id,
        "loan_id": offer_data["loan_id"],
        "status": chosen_status,
        "responded_at": responded_at
    }
    try:
        requests.post(N8N_WEBHOOK_URL, json=webhook_payload, timeout=3)
    except Exception as webhook_err:
        print("[WARN] Outbound webhook dispatch notice:", webhook_err)

    return {
        "offer_id": data.offer_id,
        "loan_id": offer_data["loan_id"],
        "status": chosen_status,
        "responded_at": responded_at,
        "message": f"Restructure offer successfully updated to {chosen_status}."
    }

@router.get("/offers/{loan_id}")
def get_loan_offers(loan_id: str):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, loan_id, offer_json, status, created_at, responded_at
        FROM restructure_offers
        WHERE loan_id = ?
        ORDER BY created_at DESC
    """, (loan_id.strip(),))
    rows = cursor.fetchall()
    conn.close()

    offers = []
    for r in rows:
        parsed_options = []
        try:
            parsed_options = json.loads(r["offer_json"])
        except Exception:
            pass
        offers.append({
            "offer_id": r["id"],
            "loan_id": r["loan_id"],
            "options": parsed_options,
            "status": r["status"],
            "created_at": r["created_at"],
            "responded_at": r["responded_at"]
        })

    return {"loan_id": loan_id, "offers": offers}
