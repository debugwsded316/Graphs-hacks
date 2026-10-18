"""Core Clinical Engine & Flask API Server for Polypharmacy Guardian.

Implements Track 1: Context for AI Agents (Agents That Act on Connected Data).
Adheres strictly to architectural and clinical guardrails:
1. NO DYNAMIC CYPHER: Queries are loaded directly from config/cypher_tools.yaml.
   All executions use parameterized bindings.
2. DETERMINISTIC RESOLVER & AI AGENT LAYER:
   Uses DrugResolver backed by RapidFuzz and local CSV seed.
   Agent handles free-text clinical notes or structured regimens with zero-latency
   deterministic clinical NLP fallback.
3. CLINICAL TRIAGE RULES:
   - Output must strictly be one of:
     * 'FLAGGED'
     * 'NO_FINDINGS_IN_INDEXED_SOURCES'
     * 'AMBIGUOUS'
     * 'UNKNOWN_COVERAGE'
   - Never output "Approved" or "Safe".
   - Exclude AMBIGUOUS entries from active calculation.
   - If interaction detected, ACB > 2.0, AMBIGUOUS, or UNKNOWN_COVERAGE, upsert pharmacist ticket.
   - Decomposes Indian Fixed-Dose Combinations (:BrandFDC) into active ingredients.
   - Candidate alternatives for pharmacist review with clinical caution notes.
   - Ticket decision state mutation: /api/tickets/<ticket_id>/decision.
"""

from datetime import datetime, timezone
import hashlib
import os
import re
from typing import Any, Dict, List, Optional, Set

from falkordb import FalkorDB
from flask import Flask, jsonify, request, send_from_directory
import yaml

from app.agent import ClinicalAgent
from app.resolver import DrugResolver, INDIAN_FDC_BRANDS


# Configuration Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "static")
CONFIG_PATH = os.getenv("CYPHER_CONFIG_PATH", os.path.join(BASE_DIR, "config", "cypher_tools.yaml"))
CSV_PATH = os.getenv("DRUGS_CSV_PATH", os.path.join(BASE_DIR, "data", "drugs_seed.csv"))
FALKORDB_HOST = os.getenv("FALKORDB_HOST", "localhost")
FALKORDB_PORT = int(os.getenv("FALKORDB_PORT", "6379"))
FALKORDB_GRAPH = os.getenv("FALKORDB_GRAPH", "polypharmacy_guardian")

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="/static")
resolver = DrugResolver(csv_path=CSV_PATH)


def load_cypher_queries(config_path: str = CONFIG_PATH) -> Dict[str, Dict[str, str]]:
    """Loads decoupled Cypher query definitions from YAML configuration."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Cypher tools config not found at: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid Cypher configuration format in: {config_path}")
    return data


CYPHER_QUERIES = load_cypher_queries()


def get_graph():
    """Returns a connected FalkorDB graph instance."""
    db = FalkorDB(host=FALKORDB_HOST, port=FALKORDB_PORT)
    return db.select_graph(FALKORDB_GRAPH)


def execute_cypher(
    query_name: str,
    params: Dict[str, Any],
    trace_log: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Executes a decoupled parameterized Cypher query and returns list of record dictionaries."""
    if query_name not in CYPHER_QUERIES:
        raise KeyError(f"Cypher query '{query_name}' not defined in {CONFIG_PATH}")

    query_str = CYPHER_QUERIES[query_name]["query"]
    graph = get_graph()
    result = graph.query(query_str, params)

    records: List[Dict[str, Any]] = []
    if hasattr(result, "result_set") and result.result_set:
        header_names = []
        if hasattr(result, "header") and result.header:
            for col in result.header:
                if isinstance(col, (list, tuple)) and len(col) > 1:
                    header_names.append(str(col[1]))
                else:
                    header_names.append(str(col))

        for row in result.result_set:
            if header_names and len(header_names) == len(row):
                records.append(dict(zip(header_names, row)))
            else:
                records.append({"data": row})

    if trace_log is not None:
        trace_log.append({
            "tool_name": query_name,
            "cypher_query": query_str.strip(),
            "parameters": params,
            "returned_rows": len(records),
        })

    return records


agent = ClinicalAgent(resolver=resolver, cypher_executor=execute_cypher)


def build_visjs_subgraph(
    canonical_names: List[str],
    ticket_id: Optional[str] = None,
    cyp_conflicts: Optional[List[Dict[str, Any]]] = None,
    acb_records: Optional[List[Dict[str, Any]]] = None,
    problematic_drugs: Optional[Set[str]] = None,
    swaps_by_drug: Optional[Dict[str, Any]] = None,
    resolutions: Optional[List[Any]] = None,
    fdc_map: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Builds a Vis.js compatible graph (nodes and edges) representing the evaluated clinical regimen."""
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    seen_nodes: Set[str] = set()
    seen_edges: Set[str] = set()

    problematic_set = problematic_drugs or set()
    acb_recs = acb_records or []
    swaps = swaps_by_drug or {}
    fdc_dict = fdc_map or {}

    # 1. Unmapped (UNKNOWN_COVERAGE) and Ambiguous (AMBIGUOUS) Warning Nodes
    if resolutions:
        for r in resolutions:
            status = getattr(r, "status", None) or (r.get("status") if isinstance(r, dict) else None)
            query_name = getattr(r, "query", None) or (r.get("query") if isinstance(r, dict) else "")
            conf = getattr(r, "confidence", 0.0) or (r.get("confidence", 0.0) if isinstance(r, dict) else 0.0)
            warning = getattr(r, "warning", None) or (r.get("warning") if isinstance(r, dict) else None)

            if status == "UNKNOWN_COVERAGE":
                node_id = f"unmapped_{query_name}"
                if node_id not in seen_nodes:
                    seen_nodes.add(node_id)
                    nodes.append({
                        "id": node_id,
                        "label": f"⛔ {query_name}\n[UNKNOWN COVERAGE]\nConf: {conf:.2f}",
                        "group": "unmapped",
                        "shape": "box",
                        "borderWidth": 2,
                        "borderWidthSelected": 3,
                        "shapeProperties": {"borderDashes": [4, 4]},
                        "color": {
                            "background": "#1e1b4b",
                            "border": "#a855f7",
                            "highlight": {"background": "#312e81", "border": "#c084fc"},
                        },
                        "font": {"color": "#f3e8ff", "face": "Inter, sans-serif", "size": 12, "bold": True},
                        "title": f"<b>{query_name}</b><br>Status: UNKNOWN_COVERAGE<br>Confidence: {conf:.2f}<br>{warning or 'Unindexed in graph'}",
                        "metadata": {
                            "type": "Unmapped Drug",
                            "name": query_name,
                            "confidence": conf,
                            "status": "UNKNOWN_COVERAGE",
                            "warning": warning,
                        },
                    })
            elif status == "AMBIGUOUS":
                node_id = f"ambiguous_{query_name}"
                if node_id not in seen_nodes:
                    seen_nodes.add(node_id)
                    nodes.append({
                        "id": node_id,
                        "label": f"⚡ {query_name}\n[AMBIGUOUS / LASA]\nConf: {conf:.2f}",
                        "group": "ambiguous",
                        "shape": "box",
                        "borderWidth": 2,
                        "borderWidthSelected": 3,
                        "shapeProperties": {"borderDashes": [4, 4]},
                        "color": {
                            "background": "#451a03",
                            "border": "#f59e0b",
                            "highlight": {"background": "#78350f", "border": "#fbbf24"},
                        },
                        "font": {"color": "#fef3c7", "face": "Inter, sans-serif", "size": 12, "bold": True},
                        "title": f"<b>{query_name}</b><br>Status: AMBIGUOUS (Near-Miss Typo)<br>{warning or 'Excluded from active calculation'}",
                        "metadata": {
                            "type": "Ambiguous Medication",
                            "name": query_name,
                            "confidence": conf,
                            "status": "AMBIGUOUS",
                            "warning": warning,
                        },
                    })

    # Collect all cleanly resolved ingredient names to visualize
    all_query_ingredients = set(canonical_names)
    for constituents in fdc_dict.values():
        for ing in constituents:
            all_query_ingredients.add(ing)

    query_ingredients_list = sorted(list(all_query_ingredients))

    # 2. Render Indian Fixed-Dose Combination (:BrandFDC) nodes
    for brand_name, constituents in fdc_dict.items():
        brand_node_id = f"brand_{brand_name}"
        if brand_node_id not in seen_nodes:
            seen_nodes.add(brand_node_id)
            nodes.append({
                "id": brand_node_id,
                "label": f"🏷️ {brand_name}\n[Indian FDC Brand]\nFormulary Ref",
                "group": "brand_fdc",
                "shape": "box",
                "borderWidth": 2,
                "borderWidthSelected": 3,
                "color": {
                    "background": "#7c2d12",
                    "border": "#fb923c",
                    "highlight": {"background": "#9a3412", "border": "#fed7aa"},
                },
                "font": {"color": "#ffedd5", "face": "Inter, sans-serif", "size": 12, "bold": True},
                "title": f"<b>{brand_name}</b> (Indian FDC Brand)<br>Active Constituents: {', '.join(constituents)}<br>Source: Standard Formulation Reference",
                "metadata": {
                    "type": "Fixed-Dose Combination",
                    "name": brand_name,
                    "constituents": constituents,
                },
            })

    # If no active ingredients to query from graph, return early
    if not query_ingredients_list:
        return {"nodes": nodes, "edges": edges}

    graph = get_graph()

    # 3. Canonical Drug Nodes from Graph
    drug_res = graph.query(
        """
        MATCH (d:Ingredient)
        WHERE d.name IN $drug_names
        RETURN d.name, d.rxnorm_id, d.drug_class, d.lasa_pairs, d.clinical_risk, d.source_citation, d.indication_class
        """,
        {"drug_names": query_ingredients_list},
    )
    drug_data_map = {}
    if hasattr(drug_res, "result_set") and drug_res.result_set:
        for row in drug_res.result_set:
            d_name = row[0]
            drug_data_map[d_name] = {
                "rxnorm_id": row[1] or "",
                "drug_class": row[2] or "",
                "lasa_pairs": row[3] or "",
                "clinical_risk": row[4] or "",
                "source_citation": row[5] or "",
                "indication_class": row[6] or "",
            }

    for drug_name in query_ingredients_list:
        node_id = f"drug_{drug_name}"
        if node_id in seen_nodes:
            continue
        seen_nodes.add(node_id)

        meta = drug_data_map.get(drug_name, {})
        rxnorm = meta.get("rxnorm_id", "")
        d_class = meta.get("drug_class", "Medication")
        risk_text = meta.get("clinical_risk", "")
        citation = meta.get("source_citation", "")
        lasa_val = meta.get("lasa_pairs", "")

        is_problematic = drug_name in problematic_set
        acb_score = next((r.get("score", 0) for r in acb_recs if r.get("drug") == drug_name), 0)

        if is_problematic or acb_score >= 3:
            bg_col = "#450a0a"
            border_col = "#ef4444"
            hl_col = "#f87171"
            icon = "⚠️ "
        else:
            bg_col = "#0f172a"
            border_col = "#38bdf8"
            hl_col = "#7dd3fc"
            icon = "💊 "

        nodes.append({
            "id": node_id,
            "label": f"{icon}{drug_name}\n[{d_class}]\nRxNorm: {rxnorm} • ACB: {acb_score}",
            "group": "drug",
            "shape": "box",
            "borderWidth": 2,
            "borderWidthSelected": 3,
            "color": {
                "background": bg_col,
                "border": border_col,
                "highlight": {"background": bg_col, "border": hl_col},
            },
            "font": {"color": "#ffffff", "face": "Inter, sans-serif", "size": 12, "bold": True},
            "title": f"<b>{drug_name}</b> ({d_class})<br>RxNorm: {rxnorm}<br>ACB: {acb_score}<br>Risk: {risk_text}<br>Source: {citation}",
            "metadata": {
                "type": "Drug / Ingredient",
                "name": drug_name,
                "rxnorm_id": rxnorm,
                "drug_class": d_class,
                "acb_score": acb_score,
                "clinical_risk": risk_text,
                "source_citation": citation,
                "lasa_pairs": lasa_val,
            },
        })

    # Connect BrandFDC to its constituent ingredients via [:CONTAINS_INGREDIENT]
    for brand_name, constituents in fdc_dict.items():
        brand_node_id = f"brand_{brand_name}"
        for ing in constituents:
            edge_id = f"fdc_{brand_name}_{ing}"
            if edge_id not in seen_edges and f"drug_{ing}" in seen_nodes:
                seen_edges.add(edge_id)
                edges.append({
                    "id": edge_id,
                    "from": brand_node_id,
                    "to": f"drug_{ing}",
                    "label": "CONTAINS_INGREDIENT",
                    "arrows": "to",
                    "color": {"color": "#fb923c"},
                    "dashes": [4, 4],
                    "width": 2,
                    "font": {"color": "#fed7aa", "size": 10, "strokeWidth": 3, "strokeColor": "#0f172a"},
                    "title": f"<b>{brand_name}</b> contains active ingredient <b>{ing}</b>",
                })

    # 4. CYP Enzymes & Edges (INHIBITS & SUBSTRATE_OF)
    inh_res = graph.query(
        """
        MATCH (d:Ingredient)-[r:INHIBITS]->(e:Enzyme)
        WHERE d.name IN $drug_names
        RETURN d.name, r.strength, e.name, r.source
        """,
        {"drug_names": query_ingredients_list},
    )
    if hasattr(inh_res, "result_set") and inh_res.result_set:
        for row in inh_res.result_set:
            d_name, strength, e_name, source = row[0], str(row[1] or "standard").lower(), row[2], row[3]
            enz_id = f"enzyme_{e_name}"
            if enz_id not in seen_nodes:
                seen_nodes.add(enz_id)
                nodes.append({
                    "id": enz_id,
                    "label": f"⚡ {e_name}\n[CYP450 Enzyme]",
                    "group": "enzyme",
                    "shape": "hexagon",
                    "size": 32,
                    "color": {
                        "background": "#78350f",
                        "border": "#f59e0b",
                        "highlight": {"background": "#b45309", "border": "#fbbf24"},
                    },
                    "font": {"color": "#fef3c7", "face": "Inter, sans-serif", "size": 13, "bold": True},
                    "title": f"<b>Enzyme: {e_name}</b><br>Cytochrome P450 Metabolic Pathway<br>Source: {source}",
                    "metadata": {"type": "Enzyme", "name": e_name},
                })

            edge_id = f"inh_{d_name}_{e_name}"
            if edge_id not in seen_edges and f"drug_{d_name}" in seen_nodes:
                seen_edges.add(edge_id)
                edges.append({
                    "id": edge_id,
                    "from": f"drug_{d_name}",
                    "to": enz_id,
                    "label": f"INHIBITS ({strength.upper()})",
                    "arrows": "to",
                    "color": {"color": "#ef4444", "highlight": "#f87171"},
                    "width": 3 if strength in ("strong", "moderate") else 2,
                    "font": {"color": "#fca5a5", "size": 11, "strokeWidth": 3, "strokeColor": "#0f172a"},
                    "title": f"<b>INHIBITS</b> ({strength})<br>Source: {source}",
                })

    sub_res = graph.query(
        """
        MATCH (d:Ingredient)-[r:SUBSTRATE_OF]->(e:Enzyme)
        WHERE d.name IN $drug_names
        RETURN d.name, e.name, r.source
        """,
        {"drug_names": query_ingredients_list},
    )
    if hasattr(sub_res, "result_set") and sub_res.result_set:
        for row in sub_res.result_set:
            d_name, e_name, source = row[0], row[1], row[2]
            enz_id = f"enzyme_{e_name}"
            if enz_id not in seen_nodes:
                seen_nodes.add(enz_id)
                nodes.append({
                    "id": enz_id,
                    "label": f"⚡ {e_name}\n[CYP450 Enzyme]",
                    "group": "enzyme",
                    "shape": "hexagon",
                    "size": 32,
                    "color": {
                        "background": "#78350f",
                        "border": "#f59e0b",
                        "highlight": {"background": "#b45309", "border": "#fbbf24"},
                    },
                    "font": {"color": "#fef3c7", "face": "Inter, sans-serif", "size": 13, "bold": True},
                    "title": f"<b>Enzyme: {e_name}</b><br>Cytochrome P450 Metabolic Pathway<br>Source: {source}",
                    "metadata": {"type": "Enzyme", "name": e_name},
                })

            edge_id = f"sub_{d_name}_{e_name}"
            if edge_id not in seen_edges and f"drug_{d_name}" in seen_nodes:
                seen_edges.add(edge_id)
                edges.append({
                    "id": edge_id,
                    "from": f"drug_{d_name}",
                    "to": enz_id,
                    "label": "SUBSTRATE_OF",
                    "arrows": "to",
                    "color": {"color": "#38bdf8", "highlight": "#7dd3fc"},
                    "width": 2,
                    "font": {"color": "#7dd3fc", "size": 11, "strokeWidth": 3, "strokeColor": "#0f172a"},
                    "title": f"<b>SUBSTRATE_OF</b><br>Source: {source}",
                })

    # 5. ACB Scores & HAS_ACB Edges
    acb_q_res = graph.query(
        """
        MATCH (d:Ingredient)-[r:HAS_ACB]->(a:ACBScore)
        WHERE d.name IN $drug_names
        RETURN d.name, a.score, r.source
        """,
        {"drug_names": query_ingredients_list},
    )
    if hasattr(acb_q_res, "result_set") and acb_q_res.result_set:
        for row in acb_q_res.result_set:
            d_name, score, source = row[0], int(row[1]), row[2]
            acb_node_id = f"acb_{score}"
            if acb_node_id not in seen_nodes:
                seen_nodes.add(acb_node_id)
                desc = "Severe Delirium/Fall Risk" if score >= 3 else "Moderate Burden" if score > 0 else "Negligible Burden"
                bg = "#831843" if score >= 3 else "#581c87" if score > 0 else "#1e293b"
                border = "#ec4899" if score >= 3 else "#c084fc" if score > 0 else "#64748b"
                text_col = "#fdf2f8" if score >= 3 else "#faf5ff" if score > 0 else "#cbd5e1"
                nodes.append({
                    "id": acb_node_id,
                    "label": f"🧠 ACB: {score}\n({desc})",
                    "group": "acb",
                    "shape": "circle",
                    "size": 28,
                    "color": {
                        "background": bg,
                        "border": border,
                        "highlight": {"background": border, "border": "#ffffff"},
                    },
                    "font": {"color": text_col, "face": "Inter, sans-serif", "size": 12},
                    "title": f"<b>ACB Score: {score}</b><br>{desc}<br>Source: {source}",
                    "metadata": {"type": "ACB Score", "score": score, "description": desc},
                })

            edge_id = f"has_acb_{d_name}_{score}"
            if edge_id not in seen_edges and f"drug_{d_name}" in seen_nodes:
                seen_edges.add(edge_id)
                edges.append({
                    "id": edge_id,
                    "from": f"drug_{d_name}",
                    "to": acb_node_id,
                    "label": f"HAS_ACB ({score})",
                    "arrows": "to",
                    "color": {"color": "#c084fc", "highlight": "#e9d5ff"},
                    "width": 2,
                    "font": {"color": "#e9d5ff", "size": 11, "strokeWidth": 3, "strokeColor": "#0f172a"},
                    "title": f"<b>HAS_ACB</b> ({score})<br>Source: {source}",
                })

    # 6. Pharmacist Ticket Node
    if ticket_id and str(ticket_id).startswith("TICK-"):
        ticket_node_id = f"ticket_{ticket_id}"
        if ticket_node_id not in seen_nodes:
            seen_nodes.add(ticket_node_id)
            nodes.append({
                "id": ticket_node_id,
                "label": f"📋 PHARMACIST TICKET\n{ticket_id}\n[PENDING_REVIEW]",
                "group": "ticket",
                "shape": "diamond",
                "size": 36,
                "color": {
                    "background": "#991b1b",
                    "border": "#f87171",
                    "highlight": {"background": "#dc2626", "border": "#ffffff"},
                },
                "font": {"color": "#ffffff", "face": "Inter, sans-serif", "size": 13, "bold": True},
                "title": f"<b>FalkorDB Pharmacist Ticket: {ticket_id}</b><br>State Persisted: PENDING_REVIEW",
                "metadata": {"type": "Pharmacist Ticket", "id": ticket_id, "status": "PENDING_REVIEW"},
            })

        for p_drug in problematic_set:
            edge_id = f"ticket_edge_{ticket_id}_{p_drug}"
            if edge_id not in seen_edges and f"drug_{p_drug}" in seen_nodes:
                seen_edges.add(edge_id)
                edges.append({
                    "id": edge_id,
                    "from": ticket_node_id,
                    "to": f"drug_{p_drug}",
                    "label": "FLAGGED_DRUG",
                    "arrows": "to",
                    "color": {"color": "#f87171"},
                    "dashes": True,
                    "width": 2,
                    "font": {"color": "#fca5a5", "size": 10, "strokeWidth": 3, "strokeColor": "#0f172a"},
                    "title": f"Ticket flagged {p_drug}",
                })

    # 7. Candidate Alternatives for Pharmacist Review
    for target_drug, swap_list in swaps.items():
        if isinstance(swap_list, list):
            for candidate_dict in swap_list[:2]:
                cand_name = candidate_dict.get("recommended_swap")
                new_acb = candidate_dict.get("new_acb", 0)
                if cand_name:
                    cand_node_id = f"swap_{cand_name}"
                    if cand_node_id not in seen_nodes:
                        seen_nodes.add(cand_node_id)
                        nodes.append({
                            "id": cand_node_id,
                            "label": f"🔄 Candidate Alternative\n{cand_name}\n(ACB={new_acb})",
                            "group": "candidate_alternative",
                            "shape": "box",
                            "borderWidth": 2,
                            "shapeProperties": {"borderDashes": [4, 4]},
                            "color": {
                                "background": "#064e3b",
                                "border": "#34d399",
                                "highlight": {"background": "#047857", "border": "#6ee7b7"},
                            },
                            "font": {"color": "#ecfdf5", "face": "Inter, sans-serif", "size": 11},
                            "title": f"<b>Candidate Alternative: {cand_name}</b><br>ACB Score: {new_acb}<br>No CYP conflicts. Clinician review required.",
                            "metadata": {"type": "Candidate Alternative", "name": cand_name, "acb": new_acb},
                        })
                    swap_edge_id = f"swap_{target_drug}_{cand_name}"
                    if swap_edge_id not in seen_edges and f"drug_{target_drug}" in seen_nodes:
                        seen_edges.add(swap_edge_id)
                        edges.append({
                            "id": swap_edge_id,
                            "from": f"drug_{target_drug}",
                            "to": cand_node_id,
                            "label": "CANDIDATE_ALTERNATIVE",
                            "arrows": "to",
                            "color": {"color": "#34d399"},
                            "dashes": [4, 4],
                            "width": 1,
                            "font": {"color": "#6ee7b7", "size": 10, "strokeWidth": 3, "strokeColor": "#0f172a"},
                            "title": f"Candidate alternative for {target_drug}",
                        })

    return {"nodes": nodes, "edges": edges}


@app.route("/", methods=["GET"])
def root_index():
    """Serves the clinical decision support frontend UI."""
    return send_from_directory(app.static_folder, "index.html")


@app.route("/health", methods=["GET"])
def health_check():
    """Health status and configuration verification."""
    falkordb_status = "UNKNOWN"
    try:
        graph = get_graph()
        falkordb_status = "CONNECTED" if graph is not None else "UNAVAILABLE"
    except Exception as e:
        falkordb_status = f"DISCONNECTED: {str(e)}"

    return jsonify({
        "status": "HEALTHY",
        "service": "Polypharmacy Guardian Clinical Engine",
        "queries_loaded": list(CYPHER_QUERIES.keys()),
        "falkordb_host": f"{FALKORDB_HOST}:{FALKORDB_PORT}",
        "falkordb_status": falkordb_status,
        "indexed_aliases": len(resolver.alias_names),
        "static_folder_exists": os.path.exists(app.static_folder),
    })


@app.route("/api/drugs", methods=["GET"])
def get_available_drugs():
    """Returns canonical drugs indexed in the local clinical seed."""
    drugs: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for entry in resolver.aliases.values():
        c_name = entry["canonical_name"]
        if c_name not in seen:
            seen.add(c_name)
            drugs.append({
                "name": c_name,
                "is_fdc": entry.get("is_fdc", False),
                "lasa_pairs": entry.get("lasa_pairs", []),
            })
    return jsonify({"count": len(drugs), "drugs": drugs})


@app.route("/api/graph-subgraph", methods=["GET"])
def graph_subgraph_endpoint():
    """Helper endpoint returning Vis.js nodes and edges for queried medications."""
    drugs_param = request.args.get("drugs", "")
    ticket_id = request.args.get("ticket_id")
    if not drugs_param:
        return jsonify({"nodes": [], "edges": []})

    drug_list = [d.strip() for d in re.split(r"[|,;]", drugs_param) if d.strip()]
    resolutions = resolver.resolve_regimen(drug_list)
    # ONLY evaluate cleanly resolved drugs
    canonical_names = [r.canonical_name for r in resolutions if r.canonical_name and r.status == "RESOLVED"]

    # Decompose any FDCs
    fdc_map = {}
    for r in resolutions:
        if r.status == "RESOLVED" and (r.is_fdc or r.canonical_name in INDIAN_FDC_BRANDS) and r.canonical_name:
            fdc_map[r.canonical_name] = r.constituents or INDIAN_FDC_BRANDS.get(r.canonical_name, [])

    acb_records = []
    cyp_conflicts = []
    problematic_drugs: Set[str] = set()
    try:
        if canonical_names:
            acb_records = execute_cypher("calculate_acb", {"drug_names": canonical_names})
            cyp_conflicts = execute_cypher("check_cyp_conflicts", {"drug_names": canonical_names})
            for c in cyp_conflicts:
                if str(c.get("inhibitor_strength", "")).lower() in ("strong", "moderate"):
                    if c.get("inhibitor"):
                        problematic_drugs.add(str(c["inhibitor"]))
                    if c.get("substrate"):
                        problematic_drugs.add(str(c["substrate"]))
            for a in acb_records:
                if float(a.get("score", 0)) >= 3:
                    problematic_drugs.add(str(a.get("drug")))
    except Exception:
        pass

    subgraph = build_visjs_subgraph(
        canonical_names=canonical_names,
        ticket_id=ticket_id,
        cyp_conflicts=cyp_conflicts,
        acb_records=acb_records,
        problematic_drugs=problematic_drugs,
        resolutions=resolutions,
        fdc_map=fdc_map,
    )
    return jsonify(subgraph)


@app.route("/triage", methods=["POST"])
@app.route("/api/triage", methods=["POST"])
@app.route("/api/analyze-regimen", methods=["POST"])
def triage_endpoint():
    """Clinical AI Agent Triage evaluation endpoint for drug regimens."""
    body = request.get_json(silent=True) or {}
    raw_input = (
        body.get("note")
        or body.get("clinical_note")
        or body.get("text")
        or body.get("drugs")
        or body.get("regimen")
        or body.get("medications")
    )

    if raw_input is None or (isinstance(raw_input, (list, str)) and len(raw_input) == 0):
        return jsonify({
            "verdict": "UNKNOWN_COVERAGE",
            "reasons": ["Missing or invalid 'drugs' list or clinical 'note' in request payload."],
            "resolutions": [],
            "acb_summary": {"total_score": 0, "breakdown": []},
            "cyp_conflicts": [],
            "lasa_warnings": [],
            "candidate_alternatives": {},
            "recommended_swaps": {},
            "fdc_decompositions": {},
            "ticket_id": None,
            "agent_trace": [],
            "clinical_narrative": "No medication input provided.",
            "graph_data": {"nodes": [], "edges": []},
            "cypher_trace": [],
        }), 400

    # Execute 3-step Agent loop
    triage_result = agent.execute_triage_loop(raw_input, execute_cypher)

    # Extract ingredients for Vis.js subgraph rendering (ONLY cleanly resolved)
    resolutions_meta = triage_result.get("resolutions", [])
    active_canonical = [
        r.get("canonical_name") for r in resolutions_meta
        if r.get("canonical_name") and r.get("status") == "RESOLVED"
    ]
    problematic_drugs = set(triage_result.get("candidate_alternatives", {}).keys())

    # Build Vis.js subgraph
    graph_data = build_visjs_subgraph(
        canonical_names=active_canonical,
        ticket_id=triage_result.get("ticket_id"),
        cyp_conflicts=triage_result.get("cyp_conflicts"),
        acb_records=triage_result.get("acb_summary", {}).get("breakdown"),
        problematic_drugs=problematic_drugs,
        swaps_by_drug=triage_result.get("candidate_alternatives"),
        resolutions=resolutions_meta,
        fdc_map=triage_result.get("fdc_decompositions"),
    )

    triage_result["graph_data"] = graph_data
    return jsonify(triage_result)


@app.route("/api/tickets/<ticket_id>/decision", methods=["POST"])
def update_ticket_decision_endpoint(ticket_id: str):
    """Updates the decision state of a pharmacist ticket in FalkorDB (Approve Swap / Override)."""
    body = request.get_json(silent=True) or {}
    new_status = str(body.get("status", "")).strip().upper()
    decision_note = str(body.get("note", "")).strip()

    if new_status not in ("APPROVED_SWAP", "OVERRIDDEN", "PENDING_REVIEW"):
        return jsonify({
            "error": "Invalid status. Allowed values: 'APPROVED_SWAP', 'OVERRIDDEN', 'PENDING_REVIEW'"
        }), 400

    timestamp = datetime.now(timezone.utc).isoformat()
    try:
        res = execute_cypher(
            "update_ticket_decision",
            {
                "ticket_id": ticket_id,
                "status": new_status,
                "note": decision_note or f"Pharmacist decision recorded: {new_status}",
                "timestamp": timestamp,
            }
        )
        if res:
            return jsonify({
                "ticket_id": ticket_id,
                "status": new_status,
                "note": decision_note,
                "timestamp": timestamp,
                "updated": True,
            })
        else:
            # Ticket ID not found in graph database
            return jsonify({
                "error": f"Ticket '{ticket_id}' not found in graph database.",
                "updated": False,
            }), 404
    except Exception as e:
        return jsonify({"error": f"Failed to update ticket: {str(e)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
