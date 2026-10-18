"""Comprehensive Clinical Verification & API Test Suite for Polypharmacy Guardian.

Verifies Track 1: Agents That Act on Connected Data.
Strict assertions across:
1. Hero Case 1: CYP2D6 Prodrug Blockade (Fluoxetine + Codeine)
   - FLAGGED verdict
   - Prodrug distinction: verifies therapeutic failure / loss of efficacy messaging
   - Indication-matched candidate alternatives (Tapentadol / Paracetamol)
   - Strict safety check: Clopidogrel is NEVER recommended for Codeine!
2. Hero Case 2: Indian FDC Hidden ACB Stacking (Solvin Cold + Amitriptyline + Oxybutynin)
   - Decomposes Solvin Cold -> Paracetamol + Chlorpheniramine + Phenylephrine
   - Total ACB = 9.0
3. Hero Case 3: CYP2C19 Antiplatelet Failure (Omeprazole + Clopidogrel)
   - Recommends Pantoprazole / Rabeprazole as candidate PPI alternatives
4. Hero Case 4: ISMP LASA Near-Miss Guardrail (Hydralizine + Tramadol)
   - AMBIGUOUS verdict with Tall-Man safety warning
   - AMBIGUOUS entry is strictly EXCLUDED from active calculation
   - Ticket persisted for pharmacist review
5. Hero Case 5: Zero-Trust Unknown Entity (Lecanemab + Fluoxetine)
   - UNKNOWN_COVERAGE verdict
   - Ticket persisted for clinical review queue
6. Negative Control: Clean Regimen (Cetirizine + Paracetamol)
   - NO_FINDINGS_IN_INDEXED_SOURCES
   - Total ACB == 0.0, 0 conflicts, no ticket
7. Unknown-Plus-Active-Conflict (Lecanemab + Fluoxetine + Codeine)
   - Evaluates active CYP2D6 conflict -> FLAGGED
   - Distinctly flags Lecanemab as unmapped entity
8. Weak Inhibitor Gating (Mirabegron + Codeine)
   - Weak CYP2D6 inhibitor does NOT trigger FLAGGED alert
9. Popular Indian Brand Aliases (Dolo 650, Crocin)
   - Dolo 650 and Crocin resolve deterministically to Paracetamol
10. Standard Substrate Toxicity vs. Prodrug Efficacy Messaging (Fluoxetine + Metoprolol)
    - Distinguishes standard substrate clearance/toxicity from prodrug bioactivation failure
11. Idempotent Ticket IDs (SHA-256)
    - Multiple evaluations of the same regimen return the exact same ticket_id
12. Pharmacist Ticket Decision State Mutation
    - POST /api/tickets/<ticket_id>/decision mutates ticket to APPROVED_SWAP
13. Graceful 404 on Non-Existent Ticket Decision
    - Non-existent ticket ID returns HTTP 404 rather than false success
14. Unstructured Free-Text AI Clinical Note Copilot
    - Note extraction via ClinicalAgent loop with grounded synthesis narrative

Exits with code 0 on full pass, non-zero on any failure.
"""

import json
import os
import sys
import urllib.request
import urllib.error


API_BASE = os.getenv("API_BASE", "http://127.0.0.1:5000")


def post_json(path: str, payload: dict, expected_status: int = 200) -> dict:
    url = f"{API_BASE}{path}"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            assert resp.status == expected_status, f"Expected HTTP {expected_status}, got {resp.status}"
            return data
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8")
        try:
            data = json.loads(err_body)
            if e.code == expected_status:
                return data
            raise RuntimeError(f"Expected HTTP {expected_status}, got HTTP {e.code} from {url}: {err_body}")
        except json.JSONDecodeError:
            if e.code == expected_status:
                return {"error": err_body}
            raise RuntimeError(f"HTTP {e.code} from {url}: {err_body}")


def run_all_tests():
    print("=" * 80)
    print("POLYPHARMACY GUARDIAN: CLINICAL VERIFICATION & ASSERTION SUITE")
    print(f"Target Server: {API_BASE}")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # TEST 1: Hero Case 1 — CYP2D6 Prodrug Blockade (Fluoxetine + Codeine)
    # -------------------------------------------------------------------------
    print("\n[TEST 1] Hero Case 1: Fluoxetine + Codeine (CYP2D6 Prodrug Blockade)...")
    res1 = post_json("/api/triage", {"drugs": ["Fluoxetine", "Codeine"]})
    assert res1["verdict"] == "FLAGGED", f"Expected FLAGGED, got {res1['verdict']}"
    assert len(res1["cyp_conflicts"]) >= 1, "Expected at least 1 CYP conflict"
    assert res1["cyp_conflicts"][0]["enzyme"] == "CYP2D6", "Expected CYP2D6 conflict"
    assert res1["ticket_id"] and res1["ticket_id"].startswith("TICK-"), f"Invalid ticket_id: {res1.get('ticket_id')}"

    # Prodrug therapeutic failure messaging check
    narrative_1 = res1.get("clinical_narrative", "").lower() + " ".join(res1.get("reasons", [])).lower()
    assert any(w in narrative_1 for w in ("prodrug", "bioactivation", "therapeutic failure", "loss of efficacy")), \
        "Expected prodrug therapeutic failure messaging for Codeine"

    # Candidate alternatives check
    cands_c1 = res1.get("candidate_alternatives", {}) or res1.get("recommended_swaps", {})
    assert "Codeine" in cands_c1, "Expected candidate alternatives for Codeine"
    codeine_swap_names = [s["recommended_swap"] for s in cands_c1["Codeine"]]

    # CRITICAL CLINICAL SAFETY ASSERTION:
    assert "Clopidogrel" not in codeine_swap_names, "SAFETY VIOLATION: Clopidogrel recommended for Codeine!"
    assert "Omeprazole" not in codeine_swap_names, "SAFETY VIOLATION: Omeprazole recommended for Codeine!"
    assert any(s in ("Tapentadol", "Paracetamol", "Fentanyl", "Aspirin") for s in codeine_swap_names), \
        f"Expected analgesic safe swaps, got: {codeine_swap_names}"

    for s in cands_c1["Codeine"]:
        assert s.get("new_acb") == 0, f"Expected ACB=0 for candidate alternative, got: {s}"
    print("  [PASS] Case 1 Verified: CYP2D6 prodrug failure flagged, Codeine swaps are safe analgesics, Clopidogrel NOT recommended.")

    # -------------------------------------------------------------------------
    # TEST 2: Hero Case 2 — Indian FDC Hidden ACB Stacking (Solvin Cold + Amitriptyline + Oxybutynin)
    # -------------------------------------------------------------------------
    print("\n[TEST 2] Hero Case 2: Solvin Cold + Amitriptyline + Oxybutynin (Indian FDC ACB Stacking)...")
    res2 = post_json("/api/triage", {"drugs": ["Solvin Cold", "Amitriptyline", "Oxybutynin"]})
    assert res2["verdict"] == "FLAGGED", f"Expected FLAGGED, got {res2['verdict']}"
    assert "fdc_decompositions" in res2, "Expected fdc_decompositions in response"
    assert "Solvin Cold" in res2["fdc_decompositions"], "Solvin Cold not decomposed"
    solvin_constituents = res2["fdc_decompositions"]["Solvin Cold"]
    assert "Paracetamol" in solvin_constituents, "Expected Paracetamol in Solvin Cold"
    assert "Chlorpheniramine" in solvin_constituents, "Expected Chlorpheniramine in Solvin Cold"
    assert "Phenylephrine" in solvin_constituents, "Expected Phenylephrine in Solvin Cold"

    # Solvin Cold (Chlorpheniramine ACB=3, Paracetamol ACB=0, Phenylephrine ACB=0) + Amitriptyline (ACB=3) + Oxybutynin (ACB=3) = 9.0
    total_acb_2 = res2["acb_summary"]["total_score"]
    assert total_acb_2 == 9.0, f"Expected Total ACB == 9.0, got {total_acb_2}"
    print(f"  [PASS] Case 2 Verified: Solvin Cold decomposed into {solvin_constituents}; Cumulative ACB = {total_acb_2:.1f}.")

    # -------------------------------------------------------------------------
    # TEST 3: Hero Case 3 — CYP2C19 Antiplatelet Failure (Omeprazole + Clopidogrel)
    # -------------------------------------------------------------------------
    print("\n[TEST 3] Hero Case 3: Omeprazole + Clopidogrel (CYP2C19 Antiplatelet Failure)...")
    res3 = post_json("/api/triage", {"drugs": ["Omeprazole", "Clopidogrel"]})
    assert res3["verdict"] == "FLAGGED", f"Expected FLAGGED, got {res3['verdict']}"
    assert len(res3["cyp_conflicts"]) >= 1, "Expected CYP conflict for Omeprazole + Clopidogrel"
    assert res3["cyp_conflicts"][0]["enzyme"] == "CYP2C19", "Expected CYP2C19 conflict"

    cands_c3 = res3.get("candidate_alternatives", {}) or res3.get("recommended_swaps", {})
    omep_swaps = [s["recommended_swap"] for s in cands_c3.get("Omeprazole", [])]
    assert "Pantoprazole" in omep_swaps or "Rabeprazole" in omep_swaps, \
        f"Expected Pantoprazole/Rabeprazole swap for Omeprazole, got: {omep_swaps}"
    print(f"  [PASS] Case 3 Verified: CYP2C19 failure flagged; Pantoprazole recommended: {omep_swaps}.")

    # -------------------------------------------------------------------------
    # TEST 4: Hero Case 4 — ISMP LASA Near-Miss Guardrail (Hydralizine + Tramadol)
    # -------------------------------------------------------------------------
    print("\n[TEST 4] Hero Case 4: Hydralizine + Tramadol (ISMP LASA Near-Miss Typo)...")
    res4 = post_json("/api/triage", {"drugs": ["Hydralizine", "Tramadol"]})
    assert res4["verdict"] == "AMBIGUOUS", f"Expected AMBIGUOUS, got {res4['verdict']}"
    assert len(res4.get("lasa_warnings", [])) >= 1, "Expected LASA warning for Hydralizine"
    assert any("Hydralizine" in w or "hydrALAZINE" in w for w in res4["lasa_warnings"]), \
        f"Expected Hydralizine / hydrALAZINE warning, got: {res4['lasa_warnings']}"

    # EXCLUDE AMBIGUOUS ENTRIES CHECK:
    # Hydralizine MUST NOT be in the evaluated ACB breakdown
    acb_drugs_evaluated = [item["drug"] for item in res4["acb_summary"]["breakdown"]]
    assert "Hydralizine" not in acb_drugs_evaluated, "SAFETY VIOLATION: Ambiguous drug was evaluated in ACB!"
    assert "Hydralazine" not in acb_drugs_evaluated, "SAFETY VIOLATION: Ambiguous drug was converted and evaluated in ACB!"
    # Ticket must be created for pharmacist verification
    assert res4.get("ticket_id") is not None, "Expected review ticket for AMBIGUOUS regimen"
    print("  [PASS] Case 4 Verified: Near-miss typo 'Hydralizine' triggered AMBIGUOUS guardrail and was excluded from calculation.")

    # -------------------------------------------------------------------------
    # TEST 5: Hero Case 5 — Zero-Trust Unknown Entity (Lecanemab + Fluoxetine)
    # -------------------------------------------------------------------------
    print("\n[TEST 5] Hero Case 5: Lecanemab + Fluoxetine (Zero-Trust Unknown Entity)...")
    res5 = post_json("/api/triage", {"drugs": ["Lecanemab", "Fluoxetine"]})
    assert res5["verdict"] == "UNKNOWN_COVERAGE", f"Expected UNKNOWN_COVERAGE, got {res5['verdict']}"
    lecanemab_res = next((r for r in res5.get("resolutions", []) if r["query"] == "Lecanemab"), None)
    assert lecanemab_res and lecanemab_res["status"] == "UNKNOWN_COVERAGE", "Lecanemab not flagged as UNKNOWN_COVERAGE"
    assert res5.get("ticket_id") is not None, "Expected review ticket for UNKNOWN_COVERAGE"
    print("  [PASS] Case 5 Verified: Unindexed mAb 'Lecanemab' triggered UNKNOWN_COVERAGE and review ticket.")

    # -------------------------------------------------------------------------
    # TEST 6: Negative Control (Clean Regimen) — Cetirizine + Paracetamol
    # -------------------------------------------------------------------------
    print("\n[TEST 6] Negative Control: Cetirizine + Paracetamol (Clean Regimen)...")
    res6 = post_json("/api/triage", {"drugs": ["Cetirizine", "Paracetamol"]})
    assert res6["verdict"] == "NO_FINDINGS_IN_INDEXED_SOURCES", \
        f"Expected NO_FINDINGS_IN_INDEXED_SOURCES, got {res6['verdict']}"
    assert res6["acb_summary"]["total_score"] == 0.0, f"Expected ACB=0, got {res6['acb_summary']['total_score']}"
    assert len(res6["cyp_conflicts"]) == 0, f"Expected 0 conflicts, got {len(res6['cyp_conflicts'])}"
    assert res6.get("ticket_id") is None, f"Expected no ticket for clean regimen, got: {res6.get('ticket_id')}"
    print("  [PASS] Case 6 Verified: Negative control cleared cleanly with 0 ACB and 0 conflicts.")

    # -------------------------------------------------------------------------
    # TEST 7: Unknown-Plus-Active-Conflict (Lecanemab + Fluoxetine + Codeine)
    # -------------------------------------------------------------------------
    print("\n[TEST 7] Unknown + Active Conflict: Lecanemab + Fluoxetine + Codeine...")
    res7 = post_json("/api/triage", {"drugs": ["Lecanemab", "Fluoxetine", "Codeine"]})
    assert res7["verdict"] == "FLAGGED", f"Expected FLAGGED for active CYP conflict, got {res7['verdict']}"
    assert len(res7["cyp_conflicts"]) >= 1, "Expected CYP conflict between Fluoxetine and Codeine"
    assert any("Lecanemab" in r for r in res7["reasons"]), "Expected notice of unmapped entity Lecanemab in reasons"
    print("  [PASS] Case 7 Verified: Active CYP conflict evaluated and flagged despite presence of unmapped entity.")

    # -------------------------------------------------------------------------
    # TEST 8: Weak Inhibitor Gate (Mirabegron + Codeine)
    # -------------------------------------------------------------------------
    print("\n[TEST 8] Weak Inhibitor Gate: Mirabegron + Codeine...")
    res8 = post_json("/api/triage", {"drugs": ["Mirabegron", "Codeine"]})
    # Mirabegron is a weak CYP2D6 inhibitor; weak inhibitors MUST NOT trigger FLAGGED
    assert res8["verdict"] != "FLAGGED", f"Weak inhibitor should not cause FLAGGED, got {res8['verdict']}"
    assert res8["verdict"] == "NO_FINDINGS_IN_INDEXED_SOURCES", f"Expected NO_FINDINGS, got {res8['verdict']}"
    print("  [PASS] Case 8 Verified: Weak CYP2D6 inhibitor correctly gated from causing high-risk FLAGGED alert.")

    # -------------------------------------------------------------------------
    # TEST 9: Popular Indian Brand Aliases (Dolo 650 & Crocin)
    # -------------------------------------------------------------------------
    print("\n[TEST 9] Popular Indian Brand Aliases (Dolo 650 & Crocin -> Paracetamol)...")
    res9 = post_json("/api/triage", {"drugs": ["Dolo 650", "Crocin"]})
    dolo_res = next((r for r in res9.get("resolutions", []) if r["query"] == "Dolo 650"), None)
    crocin_res = next((r for r in res9.get("resolutions", []) if r["query"] == "Crocin"), None)
    assert dolo_res and dolo_res["canonical_name"] == "Paracetamol", f"Expected Paracetamol for Dolo 650, got: {dolo_res}"
    assert crocin_res and crocin_res["canonical_name"] == "Paracetamol", f"Expected Paracetamol for Crocin, got: {crocin_res}"
    print("  [PASS] Case 9 Verified: Indian brand aliases Dolo 650 and Crocin resolved deterministically to Paracetamol.")

    # -------------------------------------------------------------------------
    # TEST 10: Substrate Toxicity vs. Prodrug Failure Messaging (Fluoxetine + Metoprolol)
    # -------------------------------------------------------------------------
    print("\n[TEST 10] Substrate Toxicity Messaging: Fluoxetine + Metoprolol...")
    res10 = post_json("/api/triage", {"drugs": ["Fluoxetine", "Metoprolol"]})
    assert res10["verdict"] == "FLAGGED", f"Expected FLAGGED for Metoprolol elevation, got {res10['verdict']}"
    narrative_10 = res10.get("clinical_narrative", "").lower() + " ".join(res10.get("reasons", [])).lower()
    assert any(w in narrative_10 for w in ("clearance", "toxicity", "plasma levels")), \
        "Expected decreased clearance/toxicity messaging for non-prodrug substrate Metoprolol"
    print("  [PASS] Case 10 Verified: Substrate clearance/toxicity distinguished from prodrug bioactivation failure.")

    # -------------------------------------------------------------------------
    # TEST 11: Deterministic Idempotent Pharmacist Ticket IDs (SHA-256)
    # -------------------------------------------------------------------------
    print("\n[TEST 11] Idempotent Ticket ID Hashing...")
    res11_a = post_json("/api/triage", {"drugs": ["Fluoxetine", "Codeine"]})
    res11_b = post_json("/api/triage", {"drugs": ["Codeine", "Fluoxetine"]})
    ticket_a = res11_a["ticket_id"]
    ticket_b = res11_b["ticket_id"]
    assert ticket_a == ticket_b, f"Ticket IDs must be deterministic regardless of drug order! {ticket_a} != {ticket_b}"
    print(f"  [PASS] Case 11 Verified: Idempotent ticket hash generated: {ticket_a} == {ticket_b}.")

    # -------------------------------------------------------------------------
    # TEST 12: Ticket Decision State Mutation (/api/tickets/<id>/decision)
    # -------------------------------------------------------------------------
    print("\n[TEST 12] Pharmacist Ticket Decision State Mutation...")
    decision_payload = {
        "status": "APPROVED_SWAP",
        "note": "Pharmacist reviewed CYP2D6 blockade; approved Tapentadol 50mg swap.",
    }
    dec_res = post_json(f"/api/tickets/{ticket_a}/decision", decision_payload)
    assert dec_res.get("updated") is True, f"Expected updated=True, got: {dec_res}"
    assert dec_res.get("status") == "APPROVED_SWAP", f"Expected APPROVED_SWAP, got: {dec_res.get('status')}"
    print(f"  [PASS] Case 12 Verified: Ticket {ticket_a} state mutated to APPROVED_SWAP in FalkorDB.")

    # -------------------------------------------------------------------------
    # TEST 13: Ticket Decision on Non-Existent ID (Graceful 404)
    # -------------------------------------------------------------------------
    print("\n[TEST 13] Non-Existent Ticket Decision (Graceful HTTP 404)...")
    bad_res = post_json("/api/tickets/TICK-NONEXISTENT/decision", {"status": "APPROVED_SWAP"}, expected_status=404)
    assert "error" in bad_res, f"Expected error key in 404 response, got: {bad_res}"
    print("  [PASS] Case 13 Verified: Non-existent ticket ID returned graceful HTTP 404 instead of false success.")

    # -------------------------------------------------------------------------
    # TEST 14: AI Clinical Note Copilot (Free-Text Natural Language Extraction)
    # -------------------------------------------------------------------------
    print("\n[TEST 14] AI Clinical Note Copilot (Unstructured Free-Text Input)...")
    sample_note = "72yo male prescribed Solvin Cold for rhinitis along with Amitriptyline 25mg and Oxybutynin 5mg"
    res14 = post_json("/api/triage", {"clinical_note": sample_note})
    assert res14["verdict"] == "FLAGGED", f"Expected FLAGGED from note, got {res14['verdict']}"
    assert res14["acb_summary"]["total_score"] == 9.0, f"Expected Total ACB == 9.0, got {res14['acb_summary']['total_score']}"
    assert len(res14.get("agent_trace", [])) >= 4, "Expected multi-step agent reasoning trace"
    assert len(res14.get("clinical_narrative", "")) > 50, "Expected grounded clinical synthesis narrative"
    print("  [PASS] Case 14 Verified: Unstructured clinical note processed by Agent loop with grounded synthesis.")

    print("\n" + "=" * 80)
    print("ALL 14 CLINICAL VERIFICATION TESTS PASSED SUCCESSFULLY! (EXIT 0)")
    print("=" * 80)


if __name__ == "__main__":
    try:
        run_all_tests()
        sys.exit(0)
    except Exception as e:
        print(f"\n[FAIL] TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
