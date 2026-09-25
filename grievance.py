import os
import uuid
import sqlite3
import datetime
import requests
from typing import Optional, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

DB_PATH = os.getenv("DB_PATH", "users.db")
N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL", "http://localhost:5678/webhook/Loan_Application")
GRIEVANCE_ESCALATION_HOURS = int(os.getenv("GRIEVANCE_ESCALATION_HOURS", "24"))

router = APIRouter(prefix="/grievance", tags=["Grievances"])

class GrievanceSubmitRequest(BaseModel):
    loan_id: Optional[str] = Field(None, description="Associated Loan or Application ID")
    applicant_mobile: Optional[str] = Field(None, description="Registered mobile number of applicant")
    mobile_number: Optional[str] = Field(None, description="Alias for applicant_mobile")
    category: str = Field(..., description="Category, e.g. Dispute, EMI Calculation, Foreclosure, Bureau Reporting, General")
    description: str = Field(..., description="Detailed description of the grievance or query")

class GrievanceUpdateStatusRequest(BaseModel):
    ticket_id: str = Field(..., description="UUID of the grievance ticket")
    status: str = Field(..., description="New status: open, acknowledged, escalated, resolved")

def get_db():
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    conn.row_factory = sqlite3.Row
    return conn

def init_grievance_table():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS grievances (
            id TEXT PRIMARY KEY,
            loan_id TEXT,
            applicant_mobile TEXT NOT NULL,
            category TEXT NOT NULL,
            description TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            resolved_at TEXT
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_grievance_status ON grievances(status)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_grievance_phone ON grievances(applicant_mobile)")
    conn.commit()
    conn.close()

init_grievance_table()

def parse_iso_or_str(dt_str: str) -> Optional[datetime.datetime]:
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.datetime.strptime(dt_str.split(".")[0], fmt)
        except Exception:
            pass
    return None

@router.post("/submit")
def submit_grievance(data: GrievanceSubmitRequest):
    try:
        mobile = (data.applicant_mobile or data.mobile_number or "").strip()
        if not mobile:
            raise HTTPException(status_code=422, detail="applicant_mobile or mobile_number is required.")

        ticket_id = str(uuid.uuid4())
        created_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status = "open"

        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO grievances (
                id, loan_id, applicant_mobile, category, description, status, created_at, resolved_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            ticket_id,
            data.loan_id.strip() if data.loan_id else None,
            mobile,
            data.category.strip(),
            data.description.strip(),
            status,
            created_at,
            None
        ))
        conn.commit()
        conn.close()

        return {
            "ticket_id": ticket_id,
            "id": ticket_id,
            "status": status,
            "category": data.category,
            "priority": "Standard (24h SLA)",
            "created_at": created_at,
            "message": "Grievance ticket created successfully. Our team will review within 24 hours."
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit grievance: {str(e)}")

@router.get("/status/{ticket_id}")
def get_grievance_status(ticket_id: str):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM grievances WHERE id = ?", (ticket_id.strip(),))
    row = cursor.fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Grievance ticket not found.")

    return {
        "ticket_id": row["id"],
        "loan_id": row["loan_id"],
        "applicant_mobile": row["applicant_mobile"],
        "category": row["category"],
        "description": row["description"],
        "status": row["status"],
        "created_at": row["created_at"],
        "resolved_at": row["resolved_at"]
    }

@router.post("/check-escalations")
def check_escalations():
    """
    Checks open grievances. If an open grievance is older than GRIEVANCE_ESCALATION_HOURS,
    escalates its status (open -> escalated) and fires the existing webhook dispatcher.
    """
    try:
        now = datetime.datetime.now()
        threshold_delta = datetime.timedelta(hours=GRIEVANCE_ESCALATION_HOURS)

        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM grievances WHERE status = 'open'")
        open_rows = cursor.fetchall()

        escalated_tickets = []
        for r in open_rows:
            dt = parse_iso_or_str(r["created_at"])
            if dt and (now - dt) >= threshold_delta:
                ticket_id = r["id"]
                cursor.execute("""
                    UPDATE grievances SET status = 'escalated' WHERE id = ?
                """, (ticket_id,))
                
                # Trigger outbound webhook to external system
                payload = {
                    "event": "grievance_escalation",
                    "ticket_id": ticket_id,
                    "loan_id": r["loan_id"],
                    "applicant_mobile": r["applicant_mobile"],
                    "category": r["category"],
                    "created_at": r["created_at"],
                    "escalated_at": now.strftime("%Y-%m-%d %H:%M:%S"),
                    "reason": f"Open grievance exceeded {GRIEVANCE_ESCALATION_HOURS} hours SLA."
                }
                try:
                    requests.post(N8N_WEBHOOK_URL, json=payload, timeout=3)
                except Exception as w_err:
                    print("[WARN] Grievance escalation webhook notice:", w_err)

                escalated_tickets.append(ticket_id)

        conn.commit()
        conn.close()

        return {
            "total_open_checked": len(open_rows),
            "escalated_count": len(escalated_tickets),
            "escalated_tickets": escalated_tickets,
            "escalation_sla_hours": GRIEVANCE_ESCALATION_HOURS
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Escalation check failed: {str(e)}")

@router.get("/list")
def list_grievances():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, loan_id, applicant_mobile, category, description, status, created_at, resolved_at
        FROM grievances
        ORDER BY created_at DESC
    """)
    rows = cursor.fetchall()
    conn.close()

    total_resolved = 0
    total_hours_sum = 0.0

    tickets = []
    for r in rows:
        item = dict(r)
        item["ticket_id"] = r["id"]
        item["mobile_number"] = r["applicant_mobile"]
        item["escalation_notified"] = (r["status"] == "escalated")
        tickets.append(item)
        if r["status"] == "resolved" and r["resolved_at"] and r["created_at"]:
            t_created = parse_iso_or_str(r["created_at"])
            t_resolved = parse_iso_or_str(r["resolved_at"])
            if t_created and t_resolved:
                diff_hours = (t_resolved - t_created).total_seconds() / 3600.0
                if diff_hours >= 0:
                    total_hours_sum += diff_hours
                    total_resolved += 1

    avg_resolution_hours = round(total_hours_sum / total_resolved, 1) if total_resolved > 0 else 0.0

    return {
        "total": len(tickets),
        "total_tickets": len(tickets),
        "average_resolution_hours": avg_resolution_hours,
        "total_resolved": total_resolved,
        "grievances": tickets
    }

@router.post("/update-status")
def update_grievance_status(data: GrievanceUpdateStatusRequest):
    allowed = ["open", "acknowledged", "escalated", "resolved"]
    raw_st = data.status.strip().lower()
    if raw_st in ["in_progress", "in progress"]:
        st = "acknowledged"
    else:
        st = raw_st
    if st not in allowed:
        raise HTTPException(status_code=400, detail=f"Status must be one of {allowed}")

    resolved_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") if st == "resolved" else None

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM grievances WHERE id = ?", (data.ticket_id.strip(),))
    if not cursor.fetchone():
        conn.close()
        raise HTTPException(status_code=404, detail="Ticket ID not found")

    if st == "resolved":
        cursor.execute("UPDATE grievances SET status = ?, resolved_at = ? WHERE id = ?", (st, resolved_at, data.ticket_id.strip()))
    else:
        cursor.execute("UPDATE grievances SET status = ? WHERE id = ?", (st, data.ticket_id.strip()))
    conn.commit()
    conn.close()

    return {
        "ticket_id": data.ticket_id,
        "status": st,
        "resolved_at": resolved_at
    }
