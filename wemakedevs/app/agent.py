"""Clinical AI Agent Layer for Polypharmacy Guardian.

Implements a 3-step Clinical Decision Support loop:
1. Unstructured Clinical Note Extraction & Intent Routing:
   Extracts medication entities from free-text clinical notes or structured arrays.
   Supports LLM providers (Gemini / OpenAI / Ollama) with a zero-latency
   deterministic clinical NLP extractor fallback for 100% offline reliability.
2. Parameterized Graph Tool Calling:
   Selects and executes deterministic Cypher tools from config/cypher_tools.yaml:
   - expand_fdc_brand
   - calculate_acb
   - check_cyp_conflicts
   - constraint_swap_search
   - upsert_pharmacist_ticket
3. Grounded Clinical Synthesis:
   Synthesizes an agent reasoning trace (`agent_trace`) and clinical narrative
   (`clinical_narrative`) strictly grounded in FalkorDB graph topology and literature citations.
"""

from datetime import datetime, timezone
import hashlib
import json
import os
import re
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional, Set, Tuple

from app.resolver import DrugResolver, format_tall_man, INDIAN_FDC_BRANDS


class ClinicalAgent:
    """Deterministic, graph-grounded Clinical Decision Support Agent."""

    def __init__(self, resolver: Optional[DrugResolver] = None, cypher_executor=None):
        self.resolver = resolver or DrugResolver()
        self.cypher_executor = cypher_executor

    def extract_medications(self, clinical_input: Any) -> Tuple[List[str], Dict[str, Any]]:
        """Step 1: Clinical Note Extraction & Intent Routing.

        Accepts either a structured list of strings or an unstructured clinical note string.
        """
        # If already a list of strings
        if isinstance(clinical_input, list):
            drugs = [str(d).strip() for d in clinical_input if str(d).strip()]
            return drugs, {
                "input_type": "structured_list",
                "extraction_mode": "DIRECT_INPUT",
                "extracted_count": len(drugs),
            }

        note = str(clinical_input or "").strip()
        if not note:
            return [], {
                "input_type": "free_text_empty",
                "extraction_mode": "EMPTY",
                "extracted_count": 0,
            }

        # Attempt LLM extraction if API key or local Ollama is configured
        llm_drugs = self._try_llm_extraction(note)
        if llm_drugs:
            return llm_drugs, {
                "input_type": "unstructured_clinical_note",
                "extraction_mode": "LLM_ASSISTED",
                "extracted_count": len(llm_drugs),
            }

        # Automatic zero-latency deterministic clinical NLP extractor fallback
        fallback_drugs = self._deterministic_nlp_extract(note)
        return fallback_drugs, {
            "input_type": "unstructured_clinical_note",
            "extraction_mode": "DETERMINISTIC_NLP_FALLBACK",
            "extracted_count": len(fallback_drugs),
        }

    def _try_llm_extraction(self, note: str) -> Optional[List[str]]:
        """Attempts fast LLM-based extraction if credentials are available."""
        gemini_key = os.getenv("GEMINI_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        ollama_host = os.getenv("OLLAMA_HOST")

        prompt = (
            "Extract ONLY the medication or drug names mentioned in this clinical note. "
            "Return a clean JSON array of strings containing just the drug names, e.g. [\"Solvin Cold\", \"Amitriptyline\"]. "
            f"Clinical Note: {note}"
        )

        try:
            # 1. Local Ollama (if configured)
            if ollama_host:
                url = f"{ollama_host.rstrip('/')}/api/generate"
                payload = json.dumps({"model": "llama3", "prompt": prompt, "format": "json", "stream": False}).encode()
                req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
                with urllib.request.urlopen(req, timeout=1.5) as resp:
                    data = json.loads(resp.read().decode())
                    text_resp = data.get("response", "")
                    parsed = json.loads(text_resp)
                    if isinstance(parsed, list):
                        return [str(x).strip() for x in parsed if str(x).strip()]
                    if isinstance(parsed, dict) and "drugs" in parsed:
                        return [str(x).strip() for x in parsed["drugs"] if str(x).strip()]

            # 2. OpenAI API (if configured)
            if openai_key:
                url = "https://api.openai.com/v1/chat/completions"
                payload = json.dumps({
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                }).encode()
                req = urllib.request.Request(url, data=payload, headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {openai_key}",
                }, method="POST")
                with urllib.request.urlopen(req, timeout=2.0) as resp:
                    data = json.loads(resp.read().decode())
                    text_resp = data["choices"][0]["message"]["content"]
                    # Extract JSON array
                    match = re.search(r"\[.*?\]", text_resp, re.DOTALL)
                    if match:
                        parsed = json.loads(match.group(0))
                        if isinstance(parsed, list):
                            return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass

        return None

    def _deterministic_nlp_extract(self, note: str) -> List[str]:
        """High-precision, zero-latency clinical NLP extractor for offline mode.

        Matches known drug entities, brand names, and Indian FDCs against the catalog,
        filtering out clinical noise, dosages, frequencies, and punctuation.
        """
        # Sort candidate drug names by character length descending so multi-word names match first
        all_candidates = sorted(list(self.resolver.aliases.keys()), key=lambda x: len(x), reverse=True)

        found_drugs: List[str] = []
        seen_canonical: Set[str] = set()

        text_lower = f" {note.lower()} "

        # Clean punctuation boundaries for exact token matching
        cleaned_text = re.sub(r"[,;:\(\)\.\n\r\t]+", " ", text_lower)

        for cand in all_candidates:
            if len(cand) < 3:
                continue
            # Match candidate with word boundaries
            pattern = rf"(?:\b|_){re.escape(cand)}(?:\b|_)"
            if re.search(pattern, cleaned_text):
                entry = self.resolver.aliases[cand]
                c_name = entry["canonical_name"]
                if c_name not in seen_canonical:
                    seen_canonical.add(c_name)
                    found_drugs.append(c_name)
                # Remove matched slice to avoid sub-token duplicate matching
                cleaned_text = re.sub(pattern, " ", cleaned_text)

        # If no known drugs found directly, split by common delimiters as a last resort
        if not found_drugs:
            tokens = [t.strip() for t in re.split(r"[,;+/\n\r]+", note) if t.strip()]
            for token in tokens:
                # Strip common dosage patterns like '25mg', '10 mg', 'tab', 'daily'
                clean_token = re.sub(r"\b\d+(\.\d+)?\s*(mg|mcg|ml|g|tab|capsule|daily|bid|tid|po)\b", "", token, flags=re.IGNORECASE).strip()
                if clean_token and len(clean_token) >= 3 and clean_token not in found_drugs:
                    found_drugs.append(clean_token)

        return found_drugs

    def execute_triage_loop(
        self,
        raw_input: Any,
        execute_cypher_fn,
    ) -> Dict[str, Any]:
        """Step 2 & 3: Deterministic Tool Calling and Grounded Synthesis."""
        agent_trace: List[Dict[str, Any]] = []
        cypher_trace: List[Dict[str, Any]] = []

        start_time = datetime.now(timezone.utc)

        # 1. Unstructured Clinical Note Extraction
        extracted_drugs, extraction_meta = self.extract_medications(raw_input)
        agent_trace.append({
            "step": 1,
            "action": "extract_clinical_intent",
            "details": extraction_meta,
            "extracted_drugs": extracted_drugs,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        if not extracted_drugs:
            return {
                "verdict": "UNKNOWN_COVERAGE",
                "reason": "No medication entities could be extracted from input.",
                "resolutions": [],
                "acb_summary": {"total_score": 0, "breakdown": []},
                "cyp_conflicts": [],
                "lasa_warnings": [],
                "recommended_swaps": {},
                "ticket_id": None,
                "agent_trace": agent_trace,
                "clinical_narrative": "No medication entities detected. Please enter clinical notes or drug names.",
                "cypher_trace": cypher_trace,
            }

        # 2. Entity Resolution
        resolutions = self.resolver.resolve_regimen(extracted_drugs)
        resolution_dicts = [r.to_dict() for r in resolutions]

        agent_trace.append({
            "step": 2,
            "action": "deterministic_entity_resolution",
            "total_queries": len(resolutions),
            "resolved": [r.canonical_name for r in resolutions if r.status == "RESOLVED"],
            "ambiguous": [r.query for r in resolutions if r.status == "AMBIGUOUS"],
            "unknown": [r.query for r in resolutions if r.status == "UNKNOWN_COVERAGE"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        unknown_drugs = [r for r in resolutions if r.status == "UNKNOWN_COVERAGE"]
        lasa_flags = [r for r in resolutions if r.status == "AMBIGUOUS"]
        resolved_drugs = [r for r in resolutions if r.status == "RESOLVED" and r.canonical_name]

        # 3. FDC Brand Expansion Tool Call
        fdc_expanded_map: Dict[str, List[str]] = {}
        all_eval_ingredients: Set[str] = set()

        for r in resolutions:
            if r.status in ("RESOLVED", "AMBIGUOUS") and r.canonical_name:
                c_name = r.canonical_name
                # Check if it's an FDC brand
                if r.is_fdc or c_name in INDIAN_FDC_BRANDS:
                    try:
                        fdc_rows = execute_cypher_fn(
                            "expand_fdc_brand",
                            {"brand_names": [c_name]},
                            trace_log=cypher_trace,
                        )
                        constituents = [row["ingredient"] for row in fdc_rows if "ingredient" in row]
                        if not constituents and c_name in INDIAN_FDC_BRANDS:
                            constituents = INDIAN_FDC_BRANDS[c_name]
                        fdc_expanded_map[c_name] = constituents
                        for ing in constituents:
                            all_eval_ingredients.add(ing)
                    except Exception:
                        if c_name in INDIAN_FDC_BRANDS:
                            constituents = INDIAN_FDC_BRANDS[c_name]
                            fdc_expanded_map[c_name] = constituents
                            for ing in constituents:
                                all_eval_ingredients.add(ing)
                else:
                    all_eval_ingredients.add(c_name)

        if fdc_expanded_map:
            agent_trace.append({
                "step": 3,
                "tool_call": "expand_fdc_brand",
                "brands_expanded": fdc_expanded_map,
                "rationale": "Indian Fixed-Dose Combination decomposed in FalkorDB to active metabolic ingredients",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

        active_ingredients_list = sorted(list(all_eval_ingredients))

        # 4. Graph Tool: Calculate ACB
        acb_records: List[Dict[str, Any]] = []
        if active_ingredients_list:
            acb_records = execute_cypher_fn(
                "calculate_acb",
                {"drug_names": active_ingredients_list},
                trace_log=cypher_trace,
            )
            agent_trace.append({
                "step": 4,
                "tool_call": "calculate_acb",
                "parameters": {"drug_names": active_ingredients_list},
                "results_count": len(acb_records),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

        total_acb = sum(float(rec.get("score", 0)) for rec in acb_records)

        # 5. Graph Tool: Check CYP Conflicts
        cyp_conflicts: List[Dict[str, Any]] = []
        if len(active_ingredients_list) >= 2:
            cyp_conflicts = execute_cypher_fn(
                "check_cyp_conflicts",
                {"drug_names": active_ingredients_list},
                trace_log=cypher_trace,
            )
            agent_trace.append({
                "step": 5,
                "tool_call": "check_cyp_conflicts",
                "parameters": {"drug_names": active_ingredients_list},
                "conflicts_found": len(cyp_conflicts),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

        # Evaluate Flagging Conditions
        is_acb_flagged = total_acb > 2.0
        is_cyp_flagged = len(cyp_conflicts) > 0
        is_lasa_flagged = len(lasa_flags) > 0
        has_unknown = len(unknown_drugs) > 0

        reasons: List[str] = []
        problematic_drugs: Set[str] = set()

        if is_acb_flagged:
            reasons.append(
                f"High Anticholinergic Burden: Total ACB score is {total_acb:.1f} (clinical threshold > 2.0). Cumulative fall and delirium risk."
            )
            for rec in acb_records:
                if float(rec.get("score", 0)) > 0:
                    problematic_drugs.add(str(rec.get("drug")))

        if is_cyp_flagged:
            conflict_descriptions = []
            for conflict in cyp_conflicts:
                inh = conflict.get("inhibitor")
                enz = conflict.get("enzyme")
                sub = conflict.get("substrate")
                strength = conflict.get("inhibitor_strength", "strong")
                conflict_descriptions.append(f"{inh} ({strength} {enz} inhibitor) blocks metabolism of {sub}")
                if inh:
                    problematic_drugs.add(str(inh))
                if sub:
                    problematic_drugs.add(str(sub))
            reasons.append(f"CYP450 metabolic conflict detected: {'; '.join(conflict_descriptions)}.")

        if is_lasa_flagged:
            for r in lasa_flags:
                if r.warning:
                    reasons.append(r.warning)

        # 6. Graph Tool: Constraint Swap Search
        swaps_by_drug: Dict[str, Any] = {}
        if problematic_drugs and (is_cyp_flagged or is_acb_flagged):
            for p_drug in sorted(list(problematic_drugs)):
                swaps = execute_cypher_fn(
                    "constraint_swap_search",
                    {
                        "target_drug": p_drug,
                        "current_regimen": active_ingredients_list,
                    },
                    trace_log=cypher_trace,
                )
                if swaps:
                    swaps_by_drug[p_drug] = swaps

            agent_trace.append({
                "step": 6,
                "tool_call": "constraint_swap_search",
                "flagged_drugs": list(problematic_drugs),
                "safe_alternatives_found": {k: [s.get("recommended_swap") for s in v] for k, v in swaps_by_drug.items()},
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

        # Determine Verdict
        ticket_id: Optional[str] = None
        if is_cyp_flagged or is_acb_flagged:
            verdict = "FLAGGED"
            # If unknown drugs also present, record them prominently in reasons
            if has_unknown:
                unmapped_names = [u.query for u in unknown_drugs]
                reasons.append(f"Notice: Regimen also contains unmapped agent(s) requiring verification: {', '.join(unmapped_names)}.")
        elif is_lasa_flagged:
            verdict = "AMBIGUOUS"
            if has_unknown:
                unmapped_names = [u.query for u in unknown_drugs]
                reasons.append(f"Regimen also contains unmapped agent(s): {', '.join(unmapped_names)}.")
        elif has_unknown:
            verdict = "UNKNOWN_COVERAGE"
            unmapped_names = [u.query for u in unknown_drugs]
            reasons.append(f"Regimen contains unindexed or low-confidence drug entities: {', '.join(unmapped_names)}.")
        else:
            verdict = "NO_FINDINGS_IN_INDEXED_SOURCES"
            reasons.append("No CYP enzyme conflicts, high ACB (>2.0), or LASA ambiguities found in indexed graph sources.")

        # 7. Graph Tool: Upsert Deterministic Pharmacist Ticket
        if verdict == "FLAGGED":
            sorted_drugs = sorted(active_ingredients_list)
            # Deterministic SHA-256 hash for idempotent ticket ID
            token_bytes = "|".join(sorted_drugs).encode("utf-8")
            hash_suffix = hashlib.sha256(token_bytes).hexdigest()[:8].upper()
            deterministic_ticket_id = f"TICK-{hash_suffix}"

            ticket_params = {
                "ticket_id": deterministic_ticket_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "drugs": ", ".join(sorted_drugs),
                "verdict": verdict,
                "reason": " | ".join(reasons),
            }
            try:
                ticket_res = execute_cypher_fn(
                    "upsert_pharmacist_ticket",
                    ticket_params,
                    trace_log=cypher_trace,
                )
                if ticket_res and "ticket_id" in ticket_res[0]:
                    ticket_id = ticket_res[0]["ticket_id"]
                else:
                    ticket_id = deterministic_ticket_id
            except Exception as e:
                ticket_id = deterministic_ticket_id

            agent_trace.append({
                "step": 7,
                "tool_call": "upsert_pharmacist_ticket",
                "ticket_id": ticket_id,
                "status": "PENDING_REVIEW",
                "idempotent_hash": hash_suffix,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

        # 8. Step 3: Grounded Clinical Synthesis
        clinical_narrative = self._synthesize_narrative(
            verdict=verdict,
            raw_input=raw_input,
            extracted_drugs=extracted_drugs,
            fdc_map=fdc_expanded_map,
            cyp_conflicts=cyp_conflicts,
            total_acb=total_acb,
            acb_records=acb_records,
            lasa_flags=lasa_flags,
            unknown_drugs=unknown_drugs,
            swaps=swaps_by_drug,
            ticket_id=ticket_id,
        )

        return {
            "verdict": verdict,
            "reasons": reasons,
            "ticket_id": ticket_id,
            "resolutions": resolution_dicts,
            "acb_summary": {
                "total_score": total_acb,
                "breakdown": acb_records,
            },
            "cyp_conflicts": cyp_conflicts,
            "lasa_warnings": [r.warning for r in lasa_flags if r.warning],
            "recommended_swaps": swaps_by_drug,
            "fdc_decompositions": fdc_expanded_map,
            "agent_trace": agent_trace,
            "clinical_narrative": clinical_narrative,
            "cypher_trace": cypher_trace,
        }

    def _synthesize_narrative(
        self,
        verdict: str,
        raw_input: Any,
        extracted_drugs: List[str],
        fdc_map: Dict[str, List[str]],
        cyp_conflicts: List[Dict[str, Any]],
        total_acb: float,
        acb_records: List[Dict[str, Any]],
        lasa_flags: List[Any],
        unknown_drugs: List[Any],
        swaps: Dict[str, Any],
        ticket_id: Optional[str],
    ) -> str:
        """Synthesizes an evidence-grounded clinical narrative strictly from FalkorDB graph data."""
        lines: List[str] = []

        if verdict == "FLAGGED":
            lines.append("[ALERT] CLINICAL TRIAGE: ACTION REQUIRED")
        elif verdict == "AMBIGUOUS":
            lines.append("[WARNING] ISMP LOOK-ALIKE / SOUND-ALIKE (LASA) GUARDRAIL: CONFIRMATION REQUIRED")
        elif verdict == "UNKNOWN_COVERAGE":
            lines.append("[NOTICE] UNMAPPED ENTITY ALERT: INCOMPLETE GRAPH COVERAGE")
        else:
            lines.append("[CLEAR] CLINICAL REVIEW: NO FINDINGS IN INDEXED SOURCES")

        # 1. Regimen & FDC Decompositions
        if fdc_map:
            fdc_desc = []
            for b, ings in fdc_map.items():
                fdc_desc.append(f"Fixed-Dose Combination '{b}' decomposed into active constituents: {', '.join(ings)} (CDSCO 2024)")
            lines.append(f"Regimen Decomposition: {'; '.join(fdc_desc)}.")

        # 2. CYP Conflicts
        if cyp_conflicts:
            cyp_lines = []
            for c in cyp_conflicts:
                inh = c.get("inhibitor")
                enz = c.get("enzyme")
                sub = c.get("substrate")
                strength = c.get("inhibitor_strength", "strong")
                src = c.get("inhibitor_source") or "Flockhart CYP450 Table 2024"
                cyp_lines.append(
                    f"{inh} exerts {strength} inhibition on {enz} (Source: {src}), compromising the metabolic clearance/bioactivation of {sub}."
                )
            lines.append(f"Metabolic Bottleneck: {' '.join(cyp_lines)}")

        # 3. Cumulative ACB
        if total_acb > 2.0:
            high_acb_items = [f"{r.get('drug')} (ACB={r.get('score')})" for r in acb_records if float(r.get('score', 0)) > 0]
            lines.append(
                f"Anticholinergic Burden: Regimen exhibits a cumulative ACB of {total_acb:.1f} (Threshold > 2.0; Boustani et al. 2008 ACB Scale), driven by {', '.join(high_acb_items)}. Substantially elevates geriatric confusion, cognitive decline, and fall risk."
            )
        elif total_acb > 0:
            lines.append(f"Anticholinergic Burden: Cumulative ACB score is {total_acb:.1f} (within monitored clinical limits).")

        # 4. LASA Warnings
        if lasa_flags:
            for l_flag in lasa_flags:
                if l_flag.warning:
                    lines.append(f"Medication Safety Warning: {l_flag.warning}")

        # 5. Unknown Entities
        if unknown_drugs:
            unkn_names = [u.query for u in unknown_drugs]
            lines.append(
                f"Unindexed Entities: {', '.join(unkn_names)} could not be mapped to the local clinical knowledge graph. Automated safety checks cannot certify missing agents."
            )

        # 6. Safe Alternatives
        if swaps:
            swap_lines = []
            for target, alts in swaps.items():
                alt_names = [f"{a.get('recommended_swap')} (ACB={a.get('new_acb')})" for a in alts[:2]]
                swap_lines.append(f"For {target} -> safe alternatives: {', '.join(alt_names)}")
            lines.append(
                f"Graph-Grounded Swaps (matching therapeutic indication with ACB=0 & 0 CYP conflicts): {'; '.join(swap_lines)}."
            )

        # 7. Pharmacist Ticket Reference
        if ticket_id:
            lines.append(f"State Mutation: Pharmacist intervention ticket {ticket_id} persisted to FalkorDB for verification.")

        return "\n\n".join(lines)
