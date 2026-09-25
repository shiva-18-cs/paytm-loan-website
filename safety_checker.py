from typing import List, Optional, Any
from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(tags=["Safety Checker"])

class SafetyCheckRequest(BaseModel):
    lender_name: Optional[str] = Field(None, description="Reported brand or entity name")
    stated_interest_rate: Optional[Any] = Field(None, description="Stated interest rate or APR percentage")
    stated_apr: Optional[Any] = Field(None, description="Alias for stated_interest_rate")
    fees: Optional[List[Any]] = Field(default=[], description="List of itemized fee descriptions or amounts")
    processing_fee: Optional[Any] = Field(None, description="Alias for fees item")
    tenure: Optional[Any] = Field(None, description="Loan repayment tenure")
    tenure_months: Optional[Any] = Field(None, description="Alias for tenure")
    disclosures_provided: Optional[List[str]] = Field(default=[], description="List of disclosures or documents provided")
    cooling_off_period: Optional[Any] = Field(None, description="Cooling-off / look-up period specified")
    grievance_channel: Optional[str] = Field(None, description="Grievance officer / contact details specified")
    registration_number: Optional[str] = Field(None, description="Regulator license / CIN / NBFC registration number")
    has_kfs: Optional[bool] = Field(None, description="Direct flag for APR / KFS")
    apr_disclosed: Optional[bool] = Field(None, description="Direct flag for APR")
    grievance_officer_named: Optional[bool] = Field(None, description="Direct flag for grievance officer")
    recovery_agent_code: Optional[bool] = Field(None, description="Direct flag for recovery agent code")
    data_privacy_policy: Optional[bool] = Field(None, description="Direct flag for privacy/registration")

@router.post("/safety-check")
def check_lending_safety(data: SafetyCheckRequest):
    """
    Evaluates digital lending transparency and regulatory disclosure completeness.
    RULE-BASED ONLY: Strictly audits statutory transparency disclosures without
    labeling entities as fraudulent, legitimate, safe, or unsafe.
    """
    flags = []
    
    # 1. APR (Annual Percentage Rate) Check
    apr_provided = False
    rate_str = str(data.stated_interest_rate or "").strip().lower()
    if rate_str and rate_str not in ["none", "0", "null", "undefined", "n/a", ""]:
        # If rate has numbers or mentions apr
        apr_provided = True
    # Also check disclosures list
    disclosures_lower = [str(d).lower() for d in (data.disclosures_provided or [])]
    if any("apr" in d or "annual percentage" in d for d in disclosures_lower):
        apr_provided = True
        
    flags.append({
        "item": "Annual Percentage Rate (APR)",
        "status": "provided" if apr_provided else "missing",
        "guidance": "Transparent annual percentage rate including interest and all annualized charges."
    })

    # 2. Itemized Fees Check
    fees_provided = False
    if data.fees and len(data.fees) > 0:
        # Check that fees are not just empty or empty strings
        non_empty_fees = [f for f in data.fees if str(f).strip()]
        if len(non_empty_fees) > 0:
            fees_provided = True
    if any("fee" in d or "schedule of charges" in d or "processing" in d for d in disclosures_lower):
        fees_provided = True
        
    flags.append({
        "item": "Itemized Fees & Processing Charges",
        "status": "provided" if fees_provided else "missing",
        "guidance": "Itemized breakdown of processing fees, penal charges, and upfront deductions."
    })

    # 3. Lender Identity / Registration Information Check
    identity_provided = False
    lender_name_clean = str(data.lender_name or "").strip()
    reg_clean = str(data.registration_number or "").strip()
    if reg_clean and reg_clean.lower() not in ["none", "n/a", "null", ""]:
        identity_provided = True
    elif any("registration" in d or "license" in d or "nbfc" in d or "rbi" in d or "bank" in d for d in disclosures_lower):
        identity_provided = True
    elif lender_name_clean and len(lender_name_clean) > 3 and any(word in lender_name_clean.lower() for word in ["bank", "finance", "ltd", "limited", "nbfc", "capital", "credit"]):
        identity_provided = True

    flags.append({
        "item": "Lender Identity & Official Registration",
        "status": "provided" if identity_provided else "missing",
        "guidance": "Verification of corporate identity, principal NBFC/banking partner, and regulatory license."
    })

    # 4. Grievance / Contact Channel Check
    grievance_provided = False
    contact_clean = str(data.grievance_channel or "").strip().lower()
    if contact_clean and contact_clean not in ["none", "n/a", "null", ""]:
        grievance_provided = True
    elif any("grievance" in d or "nodal officer" in d or "ombudsman" in d or "support email" in d or "helpline" in d for d in disclosures_lower):
        grievance_provided = True

    flags.append({
        "item": "Grievance Redressal & Support Channel",
        "status": "provided" if grievance_provided else "missing",
        "guidance": "Designated Nodal Grievance Officer details, toll-free contact, and escalation matrix."
    })

    # 5. Cooling-off Period Check
    cooling_provided = False
    cooling_clean = str(data.cooling_off_period or "").strip().lower()
    if cooling_clean and cooling_clean not in ["none", "0", "n/a", "null", "no"]:
        cooling_provided = True
    elif any("cooling" in d or "look-up" in d or "exit period" in d or "cancellation" in d for d in disclosures_lower):
        cooling_provided = True

    flags.append({
        "item": "Cooling-Off / Look-Up Exit Period",
        "status": "provided" if cooling_provided else "missing",
        "guidance": "Right to exit digital loan agreement within specified window without penal consequence."
    })

    # Check direct boolean flags if supplied
    if data.apr_disclosed or data.has_kfs:
        flags[0]["status"] = "provided"
    if data.recovery_agent_code or data.processing_fee:
        flags[1]["status"] = "provided"
    if data.data_privacy_policy:
        flags[2]["status"] = "provided"
    if data.grievance_officer_named:
        flags[3]["status"] = "provided"

    # Determine disclosure completeness
    provided_count = sum(1 for f in flags if f["status"] == "provided")
    if provided_count == 5:
        completeness = "complete"
    elif provided_count >= 2:
        completeness = "partial"
    else:
        completeness = "insufficient"

    return {
        "disclosure_completeness": completeness,
        "provided_count": provided_count,
        "score": provided_count,
        "total_required": 5,
        "max_score": 5,
        "flags": flags,
        "disclaimer": "This tool checks disclosure completeness only. It does not determine whether a lender is legitimate, registered, fraudulent, or safe."
    }
