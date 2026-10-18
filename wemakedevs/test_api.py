"""Comprehensive Clinical Verification & API Test Suite for Polypharmacy Guardian.

Verifies Track 1: Agents That Act on Connected Data.
Strict assertions across:
1. Hero Case 1: CYP2D6 Prodrug Blockade (Fluoxetine + Codeine)
   - FLAGGED verdict
   - Indication-matched safe swaps (Tapentadol / Paracetamol)
   - Strict safety check: Clopidogrel is NEVER recommended for Codeine!
2. Hero Case 2: Indian FDC Hidden ACB Stacking (Solvin Cold + Amitriptyline + Oxybutynin)
   - Decomposes Solvin Cold -> Paracetamol + Chlorpheniramine (ACB=3)
   - Total ACB = 9.0
3. Hero Case 3: CYP2C19 Antiplatelet Failure (Omeprazole + Clopidogrel)
   - Recommends Pantoprazole as safe PPI alternative
4. Hero Case 4: ISMP LASA Near-Miss Guardrail (Hydralizine + Tramadol)
   - AMBIGUOUS verdict
   - Tall-Man safety warning
5. Hero Case 5: Zero-Trust Unknown Entity (Lecanemab + Fluoxetine)
   - UNKNOWN_COVERAGE verdict
6. Negative Control: Clean Regimen (Cetirizine + Paracetamol)
   - NO_FINDINGS_IN_INDEXED_SOURCES
   - Total ACB == 0.0
7. Unknown-Plus-Active-Conflict (Lecanemab + Fluoxetine + Codeine)
   - Evaluates known conflict (CYP2D6 blockade) -> FLAGGED
   - Distinctly flags Lecanemab as unmapped
8. Idempotent Ticket IDs (SHA-256)
   - Multiple evaluations of the same regimen return the exact same ticket_id
9. Pharmacist Ticket Decision State Mutation
   - POST /api/tickets/<ticket_id>/decision mutates ticket to APPROVED_SWAP in FalkorDB
10. Unstructured Free-Text AI Clinical Note Copilot
    - Note extraction via ClinicalAgent loop with narrative & agent trace

Exits with code 0 on full pass, non-zero on any failure.
"""

import json
import os
import sys
import urllib.request
import urllib.error


API_BASE = os.getenv("API_BASE", "http://localhost:5000")


def post_json(path: str, payload: dict) -> dict:
    url = f"{API_BASE}{path}"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8")
        try:
            return json.loads(err_body)
        except Exception:
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

    swaps_c1 = res1.get("recommended_swaps", {})
    assert "Codeine" in swaps_c1, "Expected constraint swaps for Codeine"
    codeine_swap_names = [s["recommended_swap"] for s in swaps_c1["Codeine"]]

    # CRITICAL CLINICAL SAFETY ASSERTION:
    assert "Clopidogrel" not in codeine_swap_names, "SAFETY VIOLATION: Clopidogrel recommended for Codeine!"
    assert "Omeprazole" not in codeine_swap_names, "SAFETY VIOLATION: Omeprazole recommended for Codeine!"
    assert any(s in ("Tapentadol", "Paracetamol", "Fentanyl", "Aspirin") for s in codeine_swap_names), \
        f"Expected analgesic safe swaps, got: {codeine_swap_names}"

    # Verify all recommended swaps have ACB == 0
    for s in swaps_c1["Codeine"]:
        assert s.get("new_acb") == 0, f"Expected ACB=0 for safe swap, got: {s}"
    print("  [PASS] Case 1 Verified: CYP2D6 blockade flagged, Codeine swaps are safe analgesics, Clopidogrel NOT recommended.")

    # -------------------------------------------------------------------------
    # TEST 2: Hero Case 2 — Indian FDC Hidden ACB Stacking (Solvin Cold + Amitriptyline + Oxybutynin)
    # -------------------------------------------------------------------------
    print("\n[TEST 2] Hero Case 2: Solvin Cold + Amitriptyline + Oxybutynin (Indian FDC ACB Stacking)...")
    res2 = post_json("/api/triage", {"drugs": ["Solvin Cold", "Amitriptyline", "Oxybutynin"]})
    assert res2["verdict"] == "FLAGGED", f"Expected FLAGGED, got {res2['verdict']}"
    assert "fdc_decompositions" in res2, "Expected fdc_decompositions in response"
    assert "Solvin Cold" in res2["fdc_decompositions"], "Solvin Cold not decomposed"
    solvin_constituents = res2["fdc_decompositions"]["Solvin Cold"]
    assert "Paracetamol" in solvin_constituents and "Chlorpheniramine" in solvin_constituents, \
        f"Unexpected constituents: {solvin_constituents}"

    # Solvin Cold (Chlorpheniramine ACB=3) + Amitriptyline (ACB=3) + Oxybutynin (ACB=3) = 9.0
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

    omep_swaps = [s["recommended_swap"] for s in res3.get("recommended_swaps", {}).get("Omeprazole", [])]
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
    print("  [PASS] Case 4 Verified: Near-miss typo 'Hydralizine' triggered AMBIGUOUS guardrail with Tall-Man warning.")

    # -------------------------------------------------------------------------
    # TEST 5: Hero Case 5 — Zero-Trust Unknown Entity (Lecanemab + Fluoxetine)
    # -------------------------------------------------------------------------
    print("\n[TEST 5] Hero Case 5: Lecanemab + Fluoxetine (Zero-Trust Unknown Entity)...")
    res5 = post_json("/api/triage", {"drugs": ["Lecanemab", "Fluoxetine"]})
    assert res5["verdict"] == "UNKNOWN_COVERAGE", f"Expected UNKNOWN_COVERAGE, got {res5['verdict']}"
    lecanemab_res = next((r for r in res5.get("resolutions", []) if r["query"] == "Lecanemab"), None)
    assert lecanemab_res and lecanemab_res["status"] == "UNKNOWN_COVERAGE", "Lecanemab not flagged as UNKNOWN_COVERAGE"
    print("  [PASS] Case 5 Verified: Unindexed mAb 'Lecanemab' cleanly triggered UNKNOWN_COVERAGE.")

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
    # Must flag the dangerous known conflict (CYP2D6 blockade) while also noting the unknown entity
    assert res7["verdict"] == "FLAGGED", f"Expected FLAGGED for active CYP conflict, got {res7['verdict']}"
    assert len(res7["cyp_conflicts"]) >= 1, "Expected CYP conflict between Fluoxetine and Codeine"
    assert any("Lecanemab" in r for r in res7["reasons"]), "Expected notice of unmapped entity Lecanemab in reasons"
    print("  [PASS] Case 7 Verified: Active CYP conflict evaluated and flagged despite presence of unmapped entity.")

    # -------------------------------------------------------------------------
    # TEST 8: Deterministic Idempotent Pharmacist Ticket IDs (SHA-256)
    # -------------------------------------------------------------------------
    print("\n[TEST 8] Idempotent Ticket ID Hashing...")
    res8_a = post_json("/api/triage", {"drugs": ["Fluoxetine", "Codeine"]})
    res8_b = post_json("/api/triage", {"drugs": ["Codeine", "Fluoxetine"]})  # reverse order
    ticket_a = res8_a["ticket_id"]
    ticket_b = res8_b["ticket_id"]
    assert ticket_a == ticket_b, f"Ticket IDs must be deterministic regardless of drug order! {ticket_a} != {ticket_b}"
    print(f"  [PASS] Case 8 Verified: Idempotent ticket hash generated: {ticket_a} == {ticket_b}.")

    # -------------------------------------------------------------------------
    # TEST 9: Ticket Decision State Mutation (/api/tickets/<id>/decision)
    # -------------------------------------------------------------------------
    print("\n[TEST 9] Pharmacist Ticket Decision State Mutation...")
    decision_payload = {
        "status": "APPROVED_SWAP",
        "note": "Pharmacist reviewed CYP2D6 blockade; approved Tapentadol 50mg swap.",
    }
    dec_res = post_json(f"/api/tickets/{ticket_a}/decision", decision_payload)
    assert dec_res.get("updated") is True, f"Expected updated=True, got: {dec_res}"
    assert dec_res.get("status") == "APPROVED_SWAP", f"Expected APPROVED_SWAP, got: {dec_res.get('status')}"
    print(f"  [PASS] Case 9 Verified: Ticket {ticket_a} state mutated to APPROVED_SWAP in FalkorDB.")

    # -------------------------------------------------------------------------
    # TEST 10: AI Clinical Note Copilot (Free-Text Natural Language Extraction)
    # -------------------------------------------------------------------------
    print("\n[TEST 10] AI Clinical Note Copilot (Unstructured Free-Text Input)...")
    sample_note = "72yo male prescribed Solvin Cold for rhinitis along with Amitriptyline 25mg and Oxybutynin 5mg"
    res10 = post_json("/api/triage", {"clinical_note": sample_note})
    assert res10["verdict"] == "FLAGGED", f"Expected FLAGGED from note, got {res10['verdict']}"
    assert res10["acb_summary"]["total_score"] == 9.0, f"Expected Total ACB == 9.0, got {res10['acb_summary']['total_score']}"
    assert len(res10.get("agent_trace", [])) >= 4, "Expected multi-step agent reasoning trace"
    assert len(res10.get("clinical_narrative", "")) > 50, "Expected grounded clinical synthesis narrative"
    print("  [PASS] Case 10 Verified: Unstructured clinical note processed by Agent loop with grounded synthesis.")

    print("\n" + "=" * 80)
    print("ALL 10 CLINICAL VERIFICATION TESTS PASSED SUCCESSFULLY! (EXIT 0)")
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
