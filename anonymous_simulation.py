import os
import time
from typing import Optional, Dict, List
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

ANONYMOUS_SIMULATION_LIMIT = int(os.getenv("ANONYMOUS_SIMULATION_LIMIT", "10"))
WINDOW_SECONDS = 3600  # 1 hour window

router = APIRouter(prefix="/simulate", tags=["Simulation"])

class AnonymousSimulationRequest(BaseModel):
    loan_amount: float = Field(..., gt=0, description="Requested principal amount")
    loan_int_rate: float = Field(..., ge=0, description="Annual interest rate percentage")
    term_months: Optional[int] = Field(None, gt=0, le=120, description="Loan term in months")
    loan_term_months: Optional[int] = Field(None, gt=0, le=120, description="Alias for term_months")
    annual_income: Optional[float] = Field(None, gt=0, description="Annual income")
    person_income: Optional[float] = Field(None, gt=0, description="Alias for annual_income")

# In-memory rate limiting counter per client IP / session
RATE_LIMIT_STORE: Dict[str, List[float]] = {}

def get_client_identifier(request: Request) -> str:
    # Check X-Forwarded-For if behind a proxy
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "anonymous_client"

def check_rate_limit(client_id: str) -> int:
    now = time.time()
    timestamps = RATE_LIMIT_STORE.get(client_id, [])
    # Filter out requests older than the window
    valid_timestamps = [t for t in timestamps if now - t < WINDOW_SECONDS]
    
    if len(valid_timestamps) >= ANONYMOUS_SIMULATION_LIMIT:
        RATE_LIMIT_STORE[client_id] = valid_timestamps
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. You can only perform up to {ANONYMOUS_SIMULATION_LIMIT} anonymous simulations per hour. Please sign in or try again later."
        )
    
    valid_timestamps.append(now)
    RATE_LIMIT_STORE[client_id] = valid_timestamps
    return max(0, ANONYMOUS_SIMULATION_LIMIT - len(valid_timestamps))

def execute_simulation_logic(loan_amount: float, loan_int_rate: float, term_months: int, annual_income: float) -> dict:
    """
    Standardized financial calculation matching /simulate.
    Mathematically determines monthly EMI, total interest, and DTI ratio.
    """
    r_monthly = (loan_int_rate / 100.0) / 12.0
    n_months = term_months
    principal = loan_amount

    if r_monthly > 0:
        emi = principal * r_monthly * ((1 + r_monthly) ** n_months) / (((1 + r_monthly) ** n_months) - 1)
    else:
        emi = principal / n_months if n_months > 0 else 0.0

    total_payment = emi * n_months
    total_interest = total_payment - principal
    monthly_income = (annual_income / 12.0) if annual_income > 0 else 1.0
    dti_ratio = (emi / monthly_income) * 100.0
    loan_pct_income = principal / annual_income if annual_income > 0 else 1.0

    # Determine affordability health & eligibility
    is_favorable = False
    if dti_ratio <= 28.0:
        health = "Optimal (Affordable)"
        health_color = "emerald"
        is_favorable = True
    elif dti_ratio <= 38.0 and loan_pct_income <= 0.40:
        health = "Moderate (Acceptable)"
        health_color = "amber"
        is_favorable = True
    else:
        health = "Strained (High Debt Risk)"
        health_color = "rose"
        is_favorable = False

    # 12-month amortization preview
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

    favorable_message = "Your simulated profile meets the current eligibility criteria." if is_favorable else "Simulated debt burden is elevated. Consider extending term or reducing principal."

    return {
        "principal": round(principal, 2),
        "interest_rate": loan_int_rate,
        "term_months": n_months,
        "annual_income": round(annual_income, 2),
        "monthly_emi": round(emi, 2),
        "emi": round(emi, 2),
        "total_payment": round(total_payment, 2),
        "total_interest": round(total_interest, 2),
        "dti_ratio": round(dti_ratio, 2),
        "dti": round(dti_ratio, 2),
        "affordability_status": health,
        "affordability_color": health_color,
        "is_favorable": is_favorable,
        "eligibility_message": favorable_message,
        "favorable_message": favorable_message,
        "schedule_preview": schedule
    }

@router.post("/anonymous")
def simulate_anonymous(req: AnonymousSimulationRequest, request: Request):
    """
    Feature 7: Anonymous Pre-Application Sandbox.
    Executes identical simulation & eligibility math as /simulate without
    requiring authentication, user records, or database audit logging.
    Strictly rate limited to ANONYMOUS_SIMULATION_LIMIT requests per hour.
    """
    client_id = get_client_identifier(request)
    remaining = check_rate_limit(client_id)

    term = req.term_months or req.loan_term_months or 60
    income = req.annual_income or req.person_income or 50000.0

    res = execute_simulation_logic(
        loan_amount=req.loan_amount,
        loan_int_rate=req.loan_int_rate,
        term_months=term,
        annual_income=income
    )
    res["anonymous"] = True
    res["rate_limit_remaining"] = remaining
    return res
