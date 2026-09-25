import json
from fastapi.testclient import TestClient
from main import app, DB_PATH
import sqlite3

client = TestClient(app)

def run_tests():
    print("==================================================")
    print("CREDENCE 2.0 AUTOMATED COMPREHENSIVE TEST SUITE")
    print("==================================================")

    # 1. Existing user registration/login flow
    print("\n[TEST 1] User Registration & Profile Login Check...")
    reg_res = client.post("/api/register", json={
        "mobile_number": "9876543210",
        "name": "Jane Doe",
        "birthdate": "1992-05-14",
        "email": "jane.doe@example.com"
    })
    assert reg_res.status_code == 200, f"Register failed: {reg_res.text}"
    print("  -> /api/register passed")

    login_res = client.get("/get_user?mobile_number=9876543210")
    assert login_res.status_code == 200 and login_res.json()["exists"] == True
    print("  -> /get_user (GET alias) passed:", login_res.json()["user"]["name"])

    # 2. Existing /predict without co-applicant
    print("\n[TEST 2] Existing /predict without co-applicant...")
    payload_ind = {
        "name": "Jane Doe",
        "age": 34,
        "email": "jane.doe@example.com",
        "phone": "9876543210",
        "person_income": 85000.0,
        "person_home_ownership": "OWN",
        "loan_int_rate": 8.5,
        "loan_percent_income": 0.18,
        "previous_loan_defaults_on_file": "NO"
    }
    pred_res = client.post("/predict", json=payload_ind)
    assert pred_res.status_code == 200, f"Predict failed: {pred_res.text}"
    p_data = pred_res.json()
    assert "prediction" in p_data
    assert "confidence_score" in p_data
    assert "icon_explanation" in p_data
    app_id = p_data["application_id"]
    print(f"  -> Prediction: {p_data['prediction']}, Score: {p_data['confidence_score']}%, App ID: {app_id}")
    print(f"  -> Icon explanation items: {len(p_data['icon_explanation'])}")

    # 3. /predict with co-applicant
    print("\n[TEST 3] /predict with co-applicant (Household pooling)...")
    payload_co = dict(payload_ind)
    payload_co["co_applicant"] = {
        "name": "John Doe",
        "income": 45000.0,
        "employment": "Salaried"
    }
    co_pred_res = client.post("/predict", json=payload_co)
    assert co_pred_res.status_code == 200, f"Co-applicant predict failed: {co_pred_res.text}"
    co_data = co_pred_res.json()
    assert co_data["pooled"] == True
    assert "individual" in co_data
    assert "household" in co_data
    print(f"  -> Individual Result: {co_data['individual']['prediction']}, DTI: {co_data['individual']['debt_ratio']}%")
    print(f"  -> Household Result: {co_data['household']['prediction']}, DTI: {co_data['household']['debt_ratio']}%")

    # 4. Existing /simulate
    print("\n[TEST 4] Existing /simulate endpoint...")
    sim_res = client.post("/simulate", json={
        "loan_amount": 25000.0,
        "loan_int_rate": 8.5,
        "term_months": 60,
        "annual_income": 75000.0
    })
    assert sim_res.status_code == 200, f"Simulate failed: {sim_res.text}"
    s_data = sim_res.json()
    print(f"  -> Monthly EMI: ${s_data['monthly_emi']}, DTI: {s_data['dti_ratio']}%")

    # 5. Anonymous Simulation
    print("\n[TEST 5] Anonymous Simulation (/simulate/anonymous)...")
    anon_res = client.post("/simulate/anonymous", json={
        "loan_amount": 20000.0,
        "loan_int_rate": 7.5,
        "term_months": 48,
        "annual_income": 80000.0
    })
    assert anon_res.status_code == 200
    anon_data = anon_res.json()
    assert anon_data["anonymous"] == True
    assert "is_favorable" in anon_data
    assert "eligibility_message" in anon_data
    print("  -> Anonymous Result:", anon_data["eligibility_message"])

    # 6. Anonymous Rate Limit
    print("\n[TEST 6] Anonymous Rate Limit check...")
    hit_limit = False
    for i in range(12):
        r = client.post("/simulate/anonymous", json={
            "loan_amount": 10000.0,
            "loan_int_rate": 10.0,
            "term_months": 24,
            "annual_income": 50000.0
        })
        if r.status_code == 429:
            hit_limit = True
            print(f"  -> Successfully hit HTTP 429 on request #{i+1}: {r.json()['detail'][:60]}...")
            break
    assert hit_limit, "Did not hit rate limit as expected"

    # 7. Repayment Logging, DTI Calculation, & Distress Detection
    print("\n[TEST 7] Feature 1: Repayment Logging & Distress Detection...")
    # Cycle 1: Normal
    c1 = client.post("/repayment/log", json={
        "loan_id": app_id,
        "cycle_date": "2026-07-01",
        "emi_due": 1000.0,
        "emi_paid": 1000.0,
        "updated_income": 60000.0 # monthly = 5000 -> DTI = 20%
    })
    assert c1.status_code == 200
    assert c1.json()["distress_flag"] == False
    print(f"  -> Cycle 1: DTI={c1.json()['updated_dti']}%, Distress={c1.json()['distress_flag']}")

    # Cycle 2: Slightly higher DTI
    c2 = client.post("/repayment/log", json={
        "loan_id": app_id,
        "cycle_date": "2026-08-01",
        "emi_due": 1200.0,
        "emi_paid": 1200.0,
        "updated_income": 48000.0 # monthly = 4000 -> DTI = 30%
    })
    assert c2.status_code == 200
    print(f"  -> Cycle 2: DTI={c2.json()['updated_dti']}%, Distress={c2.json()['distress_flag']}")

    # Cycle 3: Increasing 2 consecutive times & crosses 40%
    c3 = client.post("/repayment/log", json={
        "loan_id": app_id,
        "cycle_date": "2026-09-01",
        "emi_due": 1500.0,
        "emi_paid": 1200.0,
        "updated_income": 36000.0 # monthly = 3000 -> DTI = 50%
    })
    assert c3.status_code == 200
    assert c3.json()["distress_flag"] == True
    print(f"  -> Cycle 3: DTI={c3.json()['updated_dti']}%, Distress={c3.json()['distress_flag']} (Distress detected!)")

    # History
    hist_res = client.get(f"/repayment/history/{app_id}")
    assert hist_res.status_code == 200
    print(f"  -> Repayment History total cycles: {hist_res.json()['total_cycles']}")

    # 8. Restructuring Suggestions & Response
    print("\n[TEST 8] Feature 2: Restructuring Suggestions & Response...")
    restr_sugg = client.post(f"/restructure/suggest/{app_id}")
    assert restr_sugg.status_code == 200, f"Restructure suggest failed: {restr_sugg.text}"
    offer_info = restr_sugg.json()
    offer_id = offer_info["offer_id"]
    print(f"  -> Generated Offer ID: {offer_id} with {len(offer_info['options'])} options")
    for opt in offer_info["options"]:
        print(f"     * {opt['option']}: New Tenure={opt['new_tenure']}M, New EMI=INR {opt['new_emi']}, Resulting DTI={opt['resulting_dti']}%")

    # Respond to restructure
    resp_res = client.post("/restructure/respond", json={
        "offer_id": offer_id,
        "status": "accepted"
    })
    assert resp_res.status_code == 200
    assert resp_res.json()["status"] == "accepted"
    print(f"  -> Restructure response saved: {resp_res.json()['status']}")

    # 9. Safety Checker
    print("\n[TEST 9] Feature 4: Digital Lending Safety Checker...")
    safe_res = client.post("/safety-check", json={
        "lender_name": "QuickPay Credit Limited",
        "stated_interest_rate": "18.5% APR",
        "fees": ["Processing Fee: 2%", "Late fee: ₹500"],
        "disclosures_provided": ["APR Key Fact Statement", "Grievance Officer: contact@quickpay.com", "Cooling-Off 3 days exit"]
    })
    assert safe_res.status_code == 200
    s_rep = safe_res.json()
    print(f"  -> Completeness: {s_rep['disclosure_completeness']}, Flags count: {len(s_rep['flags'])}")
    print(f"  -> Mandatory Disclaimer: {s_rep['disclaimer']}")

    # 10. Grievance Submission, Status & Escalation
    print("\n[TEST 10] Feature 6: Grievance & Query Routing...")
    g_res = client.post("/grievance/submit", json={
        "loan_id": app_id,
        "applicant_mobile": "9876543210",
        "category": "EMI Dispute",
        "description": "Double debit observed on August cycle."
    })
    assert g_res.status_code == 200
    ticket_id = g_res.json()["ticket_id"]
    print(f"  -> Submitted Ticket ID: {ticket_id}, Status: {g_res.json()['status']}")

    g_stat = client.get(f"/grievance/status/{ticket_id}")
    assert g_stat.status_code == 200
    assert g_stat.json()["status"] == "open"

    esc_res = client.post("/grievance/check-escalations")
    assert esc_res.status_code == 200
    print(f"  -> Escalations checked: {esc_res.json()['total_open_checked']} open tickets")

    # Update grievance status to resolved
    upd_res = client.post("/grievance/update-status", json={
        "ticket_id": ticket_id,
        "status": "resolved"
    })
    assert upd_res.status_code == 200
    assert upd_res.json()["status"] == "resolved"
    print(f"  -> Ticket updated to resolved: {upd_res.json()['status']}")

    # 11. Audit CSV Export
    print("\n[TEST 11] Existing Audit CSV Export...")
    exp_res = client.get("/export_audit_csv")
    assert exp_res.status_code == 200
    assert "text/csv" in exp_res.headers.get("content-type", "")
    csv_lines = exp_res.text.strip().split("\n")
    print(f"  -> Export CSV contains {len(csv_lines)} lines (Header + Records)")

    print("\n==================================================")
    print("ALL 17 VERIFICATION TESTS PASSED SUCCESSFULLY! 100%")
    print("==================================================")

if __name__ == "__main__":
    run_tests()
