"""FalkorDB Graph Ingestion Script for Polypharmacy Guardian.

Loads data/drugs_seed.csv into FalkorDB running on localhost:6379.
Constructs:
- (:Ingredient) nodes with clinical properties (rxnorm_id, drug_class, indication_class, lasa_pairs, clinical_risk, source_citation)
- (:Enzyme) nodes (e.g. CYP2D6, CYP3A4, CYP2C19, CYP1A2, CYP2C9, CYP2B6)
- (:ACBScore) nodes (0, 1, 2, 3)
- (:BrandFDC) nodes (e.g. Solvin Cold, Ultracet, Ecosprin AV, Combiflam)
- Relationships:
  * (Ingredient)-[:HAS_ACB {source: ...}]->(ACBScore)
  * (Ingredient)-[:INHIBITS {strength: ..., source: ...}]->(Enzyme)
  * (Ingredient)-[:SUBSTRATE_OF {source: ...}]->(Enzyme)
  * (Ingredient)-[:INDUCES {source: ...}]->(Enzyme)
  * (BrandFDC)-[:CONTAINS_INGREDIENT {source: 'Standard Formulation Reference'}]->(Ingredient)

Guardrails:
- Reads directly from curated reference CSV data/drugs_seed.csv.
- Exact provenance attribution:
  * ACB citations strictly mapped to 'Boustani et al. 2008 ACB Scale' for positive scores.
  * Metabolic enzyme citations mapped to 'Flockhart CYP450 Table 2024' / FDA Labels.
- Uses stdlib csv reader (zero pandas dependency).
- Decoupled parameterized Cypher queries.
"""

import csv
import os
import re
from typing import Dict, List, Optional, Tuple
from falkordb import FalkorDB


CSV_PATH = os.getenv("DRUGS_CSV_PATH", "data/drugs_seed.csv")
FALKORDB_HOST = os.getenv("FALKORDB_HOST", "localhost")
FALKORDB_PORT = int(os.getenv("FALKORDB_PORT", "6379"))
FALKORDB_GRAPH = os.getenv("FALKORDB_GRAPH", "polypharmacy_guardian")

# Standard Indian Fixed-Dose Combinations (FDCs)
INDIAN_FDC_BRANDS: Dict[str, List[str]] = {
    "Solvin Cold": ["Paracetamol", "Chlorpheniramine", "Phenylephrine"],
    "Ultracet": ["Tramadol", "Paracetamol"],
    "Ecosprin AV": ["Aspirin", "Atorvastatin"],
    "Combiflam": ["Ibuprofen", "Paracetamol"],
}

# Common single-ingredient Indian brand aliases
INDIAN_BRAND_ALIASES: Dict[str, str] = {
    "Dolo 650": "Paracetamol",
    "Dolo": "Paracetamol",
    "Crocin": "Paracetamol",
    "Calpol": "Paracetamol",
}


def parse_inhibitor_token(token: str) -> Tuple[str, str]:
    """Extracts enzyme name and inhibitor strength from a token like CYP2D6_STRONG."""
    token_upper = token.upper().strip()
    if "STRONG" in token_upper:
        strength = "strong"
    elif "MODERATE" in token_upper:
        strength = "moderate"
    elif "WEAK" in token_upper:
        strength = "weak"
    else:
        strength = "standard"

    match = re.search(r"CYP[0-9A-Z]+", token_upper)
    enzyme_name = match.group(0) if match else token.split("_")[0].strip()
    return enzyme_name, strength


def parse_enzyme_name(token: str) -> str:
    """Extracts the enzyme identifier (e.g., CYP2D6 from CYP2D6_STRONG or CYP2D6)."""
    token_upper = token.upper().strip()
    match = re.search(r"CYP[0-9A-Z]+", token_upper)
    return match.group(0) if match else token_upper.split("_")[0].strip()


def ingest_seed_data(
    csv_path: str = CSV_PATH,
    host: str = FALKORDB_HOST,
    port: int = FALKORDB_PORT,
    graph_name: str = FALKORDB_GRAPH,
) -> None:
    """Reads clinical seed CSV and populates the graph database."""
    if not os.path.exists(csv_path):
        print(f"[INGESTION] CSV seed file not found: {csv_path}")
        return

    print(f"[INGESTION] Connecting to FalkorDB at {host}:{port}, graph: '{graph_name}'...")
    db = FalkorDB(host=host, port=port)
    graph = db.select_graph(graph_name)

    # Clean previous graph state to eliminate stale shifted edges
    print("[INGESTION] Clearing existing graph nodes and edges...")
    graph.query("MATCH (n) DETACH DELETE n")

    # 1. Base Enzyme Nodes
    base_enzymes = ["CYP2D6", "CYP3A4", "CYP2C19", "CYP1A2", "CYP2C9", "CYP2B6"]
    for enzyme in base_enzymes:
        graph.query("MERGE (e:Enzyme {name: $enzyme})", {"enzyme": enzyme})

    # 2. Base ACB Score Nodes
    for score_val in [0, 1, 2, 3]:
        graph.query("MERGE (a:ACBScore {score: $score})", {"score": score_val})

    inserted_count = 0

    with open(csv_path, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            drug_name = (row.get("drug_name") or row.get("canonical_name") or "").strip()
            if not drug_name:
                continue

            rxnorm_id = (row.get("rxnorm_id") or "").strip()
            drug_class = (row.get("drug_class") or "").strip()
            indication_class = (row.get("indication_class") or drug_class).strip()
            brand_names = (row.get("brand_names") or "").strip()
            lasa_pairs = (row.get("lasa_pair") or row.get("lasa_pairs") or "").strip()
            clinical_risk = (row.get("clinical_risk") or "").strip()
            source_citation = (row.get("source_citation") or "Flockhart CYP450 Table 2024").strip()

            # Merge Ingredient Node
            graph.query(
                """
                MERGE (d:Ingredient {name: $name})
                SET d.rxnorm_id = $rxnorm_id,
                    d.drug_class = $drug_class,
                    d.indication_class = $indication_class,
                    d.brand_names = $brand_names,
                    d.lasa_pairs = $lasa_pairs,
                    d.clinical_risk = $clinical_risk,
                    d.source_citation = $source_citation
                """,
                {
                    "name": drug_name,
                    "rxnorm_id": rxnorm_id,
                    "drug_class": drug_class,
                    "indication_class": indication_class,
                    "brand_names": brand_names,
                    "lasa_pairs": lasa_pairs,
                    "clinical_risk": clinical_risk,
                    "source_citation": source_citation,
                },
            )

            # Anticholinergic Cognitive Burden (ACB) Edge
            acb_raw = (row.get("acb_score") or "").strip()
            if acb_raw != "":
                try:
                    acb_score = int(float(acb_raw))
                except ValueError:
                    acb_score = 0

                # Strict provenance attribution: cite Boustani et al. 2008 only for listed drugs with positive burden
                if acb_score > 0:
                    acb_source = "Boustani et al. 2008 ACB Scale"
                else:
                    acb_source = "Unlisted in ACB Scale (Negligible Anticholinergic Burden)"

                graph.query(
                    """
                    MATCH (d:Ingredient {name: $name})
                    MERGE (a:ACBScore {score: $score})
                    MERGE (d)-[r:HAS_ACB]->(a)
                    SET r.source = $source
                    """,
                    {
                        "name": drug_name,
                        "score": acb_score,
                        "source": acb_source,
                    },
                )

            # CYP enzyme interactions
            cyp_source = "Flockhart CYP450 Table 2024"
            if "FDA" in source_citation:
                cyp_source = source_citation

            # CYP Substrates
            cyp_sub_raw = (row.get("cyp_substrate") or "").strip()
            if cyp_sub_raw:
                sub_tokens = [t.strip() for t in re.split(r"[|;,]", cyp_sub_raw) if t.strip()]
                for token in sub_tokens:
                    enzyme_name = parse_enzyme_name(token)
                    if enzyme_name:
                        graph.query("MERGE (e:Enzyme {name: $enzyme})", {"enzyme": enzyme_name})
                        graph.query(
                            """
                            MATCH (d:Ingredient {name: $name})
                            MATCH (e:Enzyme {name: $enzyme})
                            MERGE (d)-[r:SUBSTRATE_OF]->(e)
                            SET r.source = $source
                            """,
                            {
                                "name": drug_name,
                                "enzyme": enzyme_name,
                                "source": cyp_source,
                            },
                        )

            # CYP Inhibitors
            cyp_inh_raw = (row.get("cyp_inhibitor") or "").strip()
            if cyp_inh_raw:
                inh_tokens = [t.strip() for t in re.split(r"[|;,]", cyp_inh_raw) if t.strip()]
                for token in inh_tokens:
                    enzyme_name, strength = parse_inhibitor_token(token)
                    if enzyme_name:
                        graph.query("MERGE (e:Enzyme {name: $enzyme})", {"enzyme": enzyme_name})
                        graph.query(
                            """
                            MATCH (d:Ingredient {name: $name})
                            MATCH (e:Enzyme {name: $enzyme})
                            MERGE (d)-[r:INHIBITS]->(e)
                            SET r.strength = $strength,
                                r.source = $source
                            """,
                            {
                                "name": drug_name,
                                "enzyme": enzyme_name,
                                "strength": strength,
                                "source": cyp_source,
                            },
                        )

            # CYP Inducers
            cyp_ind_raw = (row.get("cyp_inducer") or "").strip()
            if cyp_ind_raw:
                ind_tokens = [t.strip() for t in re.split(r"[|;,]", cyp_ind_raw) if t.strip()]
                for token in ind_tokens:
                    enzyme_name = parse_enzyme_name(token)
                    if enzyme_name:
                        graph.query("MERGE (e:Enzyme {name: $enzyme})", {"enzyme": enzyme_name})
                        graph.query(
                            """
                            MATCH (d:Ingredient {name: $name})
                            MATCH (e:Enzyme {name: $enzyme})
                            MERGE (d)-[r:INDUCES]->(e)
                            SET r.source = $source
                            """,
                            {
                                "name": drug_name,
                                "enzyme": enzyme_name,
                                "source": cyp_source,
                            },
                        )

            inserted_count += 1

    # 3. Ingest Indian Fixed-Dose Combinations (:BrandFDC)
    print("[INGESTION] Linking Indian Fixed-Dose Combinations (:BrandFDC)...")
    for brand_name, constituents in INDIAN_FDC_BRANDS.items():
        graph.query(
            """
            MERGE (b:BrandFDC {name: $brand_name})
            SET b.country = 'India',
                b.source = 'Standard Formulation Reference'
            """,
            {"brand_name": brand_name},
        )
        for ing in constituents:
            graph.query(
                """
                MATCH (b:BrandFDC {name: $brand_name})
                MATCH (i:Ingredient {name: $ing_name})
                MERGE (b)-[r:CONTAINS_INGREDIENT]->(i)
                SET r.source = 'Standard Formulation Reference'
                """,
                {"brand_name": brand_name, "ing_name": ing},
            )

    print(f"[INGESTION] Completed ingestion of {inserted_count} drug records and {len(INDIAN_FDC_BRANDS)} FDC brands into FalkorDB.")


if __name__ == "__main__":
    ingest_seed_data()
