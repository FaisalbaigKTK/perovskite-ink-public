# Pseudocode & Workflow — Perovskite Ink Recipe Mining Pipeline

## Overview

This pipeline automatically mines perovskite solar cell ink recipes from
open-access scientific literature, extracts structured chemistry data, scores
and clusters recipes, and provides ML-powered search and recommendation.

---

## Stage 0 — Initialisation

```
CONFIGURE paths: data/, scripts/, logs/
SET search queries (perovskite + inkjet + solvents keywords)
SET API parameters: per_page=25, max_pages=20, retry=7
```

---

## Stage 1 — Literature Search (step1)

```
FOR EACH query IN QUERIES:
    page ← 1
    WHILE page ≤ MAX_PAGES:
        response ← GET OpenAlex API(query, page, filter=open_access)
        IF response is empty: BREAK
        FOR EACH paper IN response:
            EXTRACT: title, doi, year, cited_by, venue, is_oa, pdf_url
        page ← page + 1
        SLEEP(1s)   # rate limiting

DEDUPLICATE by openalex_id
SORT by cited_by DESC
SAVE → data/01_search/step1_results.csv
```

---

## Stage 2 — Candidate Ranking (step2)

```
LOAD step1_results.csv
IF ML classifier model exists:
    SCORE each paper using pre-download classifier
        features: title TF-IDF, venue, year, cited_by, is_oa
ELSE:
    SCORE by keyword match in title + citation count

RANK papers by combined score
SAVE top-N candidates → data/01_search/step2_candidates.csv
```

---

## Stage 3 — PDF Acquisition (step3a, step3b)

```
LOAD step2_candidates.csv
FOR EACH candidate:
    pdf_url ← best available OA pdf url
    IF pdf_url is null:
        TRY unpaywall / semantic scholar API

FOR EACH candidate WITH pdf_url:
    ATTEMPT download with retry + exponential backoff
    IF success: SAVE → data/02_pdfs/{doi_key}.pdf
    IF fail:    LOG → failed_downloads.txt

SAVE download manifest → data/02_pdfs/pdf_manifest.csv
```

---

## Stage 4 — Recipe Extraction (step4c, step4d, step4e)

```
LOAD pdf_manifest.csv
FOR EACH downloaded PDF:
    text ← extract_text_from_pdf(pdf)
    chunks ← split_into_sections(text, max_tokens=3000)

    FOR EACH chunk:
        prompt ← FILL(PROMPT_1_RECIPE_EXTRACTION, text=chunk)
        response ← LLM_API(prompt, model="claude-sonnet-4-20250514")
        recipes ← parse_JSON(response)
        APPEND recipes to raw_extractions

    FILTER recipes: must contain perovskite keywords
    SCORE each recipe with PROMPT_2_RECIPE_SCORING

SAVE → data/03_extracted/raw_recipes.csv
FILTER perovskite-only → data/03_extracted/perovskite_recipes.csv
```

---

## Stage 9 — Preprocessing & Normalisation (step9b–step9k)

```
LOAD perovskite_recipes.csv

# step9b: Filter real ink recipes
REMOVE rows where recipe is vague or clearly not an ink formulation
KEEP rows with at least formula OR solvent present

# step9c: Clean ratios
PARSE solvent ratio strings → numeric (e.g. "4:1" → [4.0, 1.0])
NORMALISE to decimal fractions

# step9d: Extract composition
PARSE perovskite formula into ion components:
    A-site: MA, FA, Cs, Rb
    B-site: Pb, Sn
    X-site: I, Br, Cl
COMPUTE stoichiometric fractions

# step9e: Parse formula fractions
HANDLE mixed-halide formulas (e.g. MAPbI2.55Br0.45)
EXTRACT x, y coefficients

# step9g: Extract solvent system
CANONICALISE solvent names using abbreviation map:
    "dimethylformamide" → "DMF"
    "dimethyl sulfoxide" → "DMSO"
    "gamma-butyrolactone" → "GBL"
NORMALISE to canonical form: "DMF:DMSO 4:1 v/v"

# step9h/9i: Fix ratio mapping
MATCH ratio to correct solvent pair
SELECT best ratio from multiple candidates

# step9j: Extract additives
SCAN text for additive mentions (MACl, methylamine, PEAI, PbCl2, etc.)
CLASSIFY role: crystallisation | passivation | viscosity | other

# step9k: Export tidy dataset
MERGE all columns
COMPUTE has_formula, has_solvent, has_ratio, has_molarity, has_additives flags
SAVE → data/05_final/perovskite_ink_dataset_TIDY.csv
```

---

## Stage 10–12 — ML Classifiers

```
# step10: Gold classifier (post-download)
LOAD TIDY dataset
LABEL: is_gold = 1 if recipe_score >= GOLD_THRESHOLD else 0
FEATURES: recipe_score, molarity_M, has_formula, has_solvent, has_ratio, has_additives
MODEL: LogisticRegression (balanced class weights)
EVALUATE: ROC-AUC, P@K on 25% held-out split
SAVE model + ranked probabilities

# step11: Pre-download classifier (step1 features only)
FEATURES: title TF-IDF, venue, year, cited_by, is_oa
MODEL: LogisticRegression (balanced)
EVALUATE: ROC-AUC, Average Precision, P@25/50/100

# step12: Weighted pre-download classifier
GOLD label weight = 1.0
SILVER label weight = 0.4 (recipe_score >= 6 but not gold)
BACKGROUND weight = 0.05
MODEL: LogisticRegression (sample_weight=w)
```

---

## Stage 13–15 — Chemistry Clustering

```
# step13: Cluster chemistry
FEATURES: solvent_system_final (OHE), formula_family (OHE),
          has_additive, molarity_M (scaled)
ALGORITHM: KMeans(k=4, random_state=42)
ASSIGN cluster labels to all TIDY rows
COMPUTE cluster centroid statistics

# step14: Cluster breakdown
FOR EACH cluster:
    REPORT: top solvents, top additives, formula families, mean recipe_score
IDENTIFY best-performing cluster

# step15: Cluster predictor
TRAIN gradient boosting / logistic model:
    INPUT: paper features (step1 + step2)
    TARGET: is_in_best_cluster
SAVE predictor for use in stage 2 re-ranking
```

---

## Stage 16–20 — Search, Recommendation & Exploration

```
# step16: Find similar inks
QUERY: user provides a partial recipe (formula, solvent, molarity)
COMPUTE: Jaccard similarity on solvent tokens + formula family match
RETURN: top-K most similar recipes from TIDY dataset

# step17: Recommend by constraints
FILTER TIDY by user constraints (e.g. "no Pb", "DMSO only", "PCE > 15%")
RANK by recipe_score DESC
RETURN top matches with provenance (paper doi)

# step18: Generate knowledge report
AGGREGATE statistics: solvent distribution, formula families, recipe tiers
IDENTIFY best cluster chemistry
PRODUCE natural language report → data/07_reports/knowledge_report.txt

# step19: Predict recipe score
MODEL: Ridge regression
FEATURES: solvent_system_final, formula_family, additive_tokens (TF-IDF), molarity_M
TARGET: recipe_score
EVALUATE: R2, MAE, RMSE on 25% held-out
RANK all recipes by predicted score

# step20: Design exploration engine
GENERATE candidate recipes via grid search:
    solvent_system × formula_family × molarity range × additive combos
SCORE each candidate with trained regression model
RANK by predicted recipe_score
RETURN top-M novel candidate formulations for experimental testing
```

---

## Data Flow Diagram

```
OpenAlex API
    │
    ▼
step1_results.csv  ──► step2_candidates.csv
                              │
                              ▼
                        PDF download
                              │
                              ▼
                    LLM extraction (prompts 1–5)
                              │
                              ▼
                    raw_recipes.csv
                              │
                        preprocessing
                    (step9b → step9k)
                              │
                              ▼
              perovskite_ink_dataset_TIDY.csv
                    ┌─────────┴─────────┐
                    ▼                   ▼
             ML classifiers       Chemistry clustering
             (steps 10–12)         (steps 13–15)
                    │                   │
                    └─────────┬─────────┘
                              ▼
                  Search / Recommend / Explore
                      (steps 16–20)
                              │
                              ▼
                    knowledge_report.txt
                    design_candidates.csv
```
