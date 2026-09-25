import os
import sqlite3
import datetime
from typing import Optional, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

DB_PATH = os.getenv("DB_PATH", "users.db")

router = APIRouter(prefix="/repayment", tags=["Repayment Tracker"])

class RepaymentLogRequest(BaseModel):
    loan_id: str = Field(..., description="ID or Application ID of the loan")
    cycle_date: Optional[str] = Field(None, description="Date of repayment cycle YYYY-MM-DD")
    payment_date: Optional[str] = Field(None, description="Alias for cycle_date")
    emi_due: float = Field(..., ge=0, description="Scheduled EMI amount")
    emi_paid: Optional[float] = Field(None, ge=0, description="Actual EMI amount paid")
    amount_paid: Optional[float] = Field(None, ge=0, description="Alias for emi_paid")
    updated_income: Optional[float] = Field(None, ge=0, description="Updated monthly or annual income")
    current_monthly_income: Optional[float] = Field(None, ge=0, description="Alias for updated_income")

def get_db():
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    conn.row_factory = sqlite3.Row
    return conn

def init_repayments_table():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS repayments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            loan_id TEXT NOT NULL,
            cycle_date TEXT NOT NULL,
            emi_due REAL NOT NULL,
            emi_paid REAL NOT NULL,
            updated_income REAL,
            updated_dti REAL NOT NULL,
            distress_flag BOOLEAN NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_repay_loan_id ON repayments(loan_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_repay_cycle ON repayments(cycle_date)")
    conn.commit()
    conn.close()

init_repayments_table()

def get_latest_known_income(loan_id: str) -> float:
    """
    Retrieves latest known income for this loan_id:
    1. From latest repayment record with updated_income
    2. Fallback to audit_logs / applications table
    """
    conn = get_db()
    cursor = conn.cursor()
    
    # 1. Check prior repayments
    cursor.execute("""
        SELECT updated_income FROM repayments
        WHERE loan_id = ? AND updated_income IS NOT NULL AND updated_income > 0
        ORDER BY cycle_date DESC, id DESC LIMIT 1
    """, (loan_id,))
    row = cursor.fetchone()
    if row and row["updated_income"]:
        conn.close()
        return float(row["updated_income"])
        
    # 2. Check audit_logs / applications
    for tbl in ["audit_logs", "applications"]:
        try:
            cursor.execute(f"""
                SELECT income FROM {tbl}
                WHERE id = ? OR application_id = ?
                LIMIT 1
            """, (loan_id, loan_id))
            row = cursor.fetchone()
            if row and row["income"]:
                conn.close()
                return float(row["income"])
        except Exception:
            pass
            
    conn.close()
    return 50000.0  # Safe default if no prior income record found

@router.post("/log")
def log_repayment(data: RepaymentLogRequest):
    try:
        # Determine cycle date
        cycle_date = data.cycle_date or data.payment_date or datetime.date.today().isoformat()
        
        # Determine emi paid
        emi_paid = data.emi_paid if data.emi_paid is not None else (data.amount_paid if data.amount_paid is not None else data.emi_due)

        # Determine income: if updated_income provided, use it; else latest known income
        income_input = data.updated_income if data.updated_income is not None else data.current_monthly_income
        if income_input is not None and income_input > 0:
            effective_income = float(income_input)
        else:
            effective_income = get_latest_known_income(data.loan_id)
            
        # Standardize monthly income
        # In Credence, applicant incomes are annual amounts (e.g. 50,000 to 500,000)
        # If effective_income < 15,000, treat as already monthly, else convert annual to monthly
        monthly_income = (effective_income / 12.0) if effective_income > 15000.0 else effective_income
        if monthly_income <= 0:
            monthly_income = 1.0

        # Recalculate DTI percentage: (emi_due / monthly_income) * 100
        current_dti = round((data.emi_due / monthly_income) * 100.0, 2)

        # Retrieve prior cycles to evaluate consecutive increase condition
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT updated_dti FROM repayments
            WHERE loan_id = ?
            ORDER BY cycle_date ASC, id ASC
        """, (data.loan_id,))
        prior_rows = cursor.fetchall()
        prior_dtis = [float(r["updated_dti"]) for r in prior_rows]
        
        # Distress flag rules:
        # 1. DTI crosses 40% (current_dti > 40.0)
        # OR
        # 2. DTI increases for 2 or more consecutive repayment cycles
        is_above_40 = current_dti > 40.0
        consecutive_increase = False
        if len(prior_dtis) >= 2:
            prev_1 = prior_dtis[-1]
            prev_2 = prior_dtis[-2]
            if prev_2 < prev_1 < current_dti:
                consecutive_increase = True
        elif len(prior_dtis) == 1:
            consecutive_increase = False

        distress_flag = bool(is_above_40 or consecutive_increase)
        created_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        cursor.execute("""
            INSERT INTO repayments (
                loan_id, cycle_date, emi_due, emi_paid, updated_income, updated_dti, distress_flag, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data.loan_id.strip(),
            cycle_date.strip(),
            round(data.emi_due, 2),
            round(emi_paid, 2),
            round(effective_income, 2) if income_input is not None else None,
            current_dti,
            1 if distress_flag else 0,
            created_at
        ))
        conn.commit()
        record_id = cursor.lastrowid
        conn.close()

        return {
            "id": record_id,
            "loan_id": data.loan_id,
            "cycle_date": cycle_date,
            "payment_date": cycle_date,
            "emi_due": data.emi_due,
            "emi_paid": emi_paid,
            "amount_paid": emi_paid,
            "updated_income": income_input,
            "updated_dti": current_dti,
            "dti_ratio": round(current_dti / 100.0, 4),
            "distress_flag": distress_flag,
            "distress_detected": distress_flag,
            "reasons": {
                "dti_above_40": is_above_40,
                "consecutive_dti_increase": consecutive_increase
            },
            "created_at": created_at,
            "message": "Repayment cycle logged successfully."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to log repayment: {str(e)}")

@router.get("/history/{loan_id}")
def get_repayment_history(loan_id: str):
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, cycle_date, emi_due, emi_paid, updated_income, updated_dti, distress_flag, created_at
            FROM repayments
            WHERE loan_id = ?
            ORDER BY cycle_date ASC, id ASC
        """, (loan_id.strip(),))
        rows = cursor.fetchall()
        conn.close()

        history = []
        for r in rows:
            history.append({
                "id": r["id"],
                "loan_id": loan_id,
                "cycle_date": r["cycle_date"],
                "payment_date": r["cycle_date"],
                "emi_due": r["emi_due"],
                "emi_paid": r["emi_paid"],
                "amount_paid": r["emi_paid"],
                "updated_income": r["updated_income"],
                "updated_dti": r["updated_dti"],
                "dti_ratio": round(r["updated_dti"] / 100.0, 4),
                "distress_flag": bool(r["distress_flag"]),
                "distress_detected": bool(r["distress_flag"]),
                "created_at": r["created_at"]
            })

        latest_dti = history[-1]["updated_dti"] if history else None
        prev_dti = history[-2]["updated_dti"] if len(history) >= 2 else None
        current_distress = history[-1]["distress_flag"] if history else False

        return {
            "loan_id": loan_id,
            "total_cycles": len(history),
            "latest_dti": latest_dti,
            "previous_dti": prev_dti,
            "current_distress": current_distress,
            "distress_detected": current_distress,
            "history": history,
            "repayments": list(reversed(history))
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch repayment history: {str(e)}")
