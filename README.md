# Perovskite Ink Recipe Mining — Public Repository

> **Methodology, prompts, preprocessing scripts, evaluation scripts,  
> pseudocode, and reproducible sample data for the perovskite ink database.**

This repository contains the **full open-source methodology** of an automated
pipeline that mines perovskite solar cell ink recipes from scientific literature
using large language models, classic NLP, and machine learning.

The production SaaS platform built on top of this pipeline is under separate
development and is not included here.

---

## 📁 Repository Structure

```
perovskite_ink_public/
│
├── scripts/                  # All pipeline scripts (Steps 1–20)
│   ├── step1_search_openalex.py         # Literature search via OpenAlex API
│   ├── step2_pick_candidates.py         # ML-powered candidate ranking
│   ├── step3a_merge_pdf_urls.py         # PDF URL consolidation
│   ├── step3b_download_pdfs_smart.py    # Robust PDF downloader
│   ├── step4c_refine_best_recipe.py     # Recipe refinement
│   ├── step4d_make_final_csv.py         # Final CSV assembly
│   ├── step4e_filter_perovskite_only.py # Perovskite filter
│   ├── step9b_filter_real_ink_recipes.py  # Recipe authenticity filter
│   ├── step9c_clean_ratios.py           # Solvent ratio normalisation
│   ├── step9d2_extract_composition_better.py  # Perovskite composition parser
│   ├── step9e_parse_formula_fractions.py  # Formula fraction parser
│   ├── step9g_extract_solvent_system.py  # Solvent system canonicalisation
│   ├── step9h_fix_solvent_ratio_mapping.py
│   ├── step9i_best_ratio_anywhere.py
│   ├── step9j_extract_additives.py      # Additive extraction
│   ├── step9k_export_tidy_dataset.py    # Final tidy dataset export
├── prompts/
│   └── llm_extraction_prompts.md        # All LLM prompts used for extraction
│
├── pseudocode/
│   └── pipeline_pseudocode.md           # Step-by-step pseudocode & data flow diagram
│
├── evaluation/
│   ├── evaluate_extraction.py           # Field-level precision/recall/F1
│   └── evaluate_classifiers.py          # ROC-AUC, P@K for ML classifiers
│
├── data_sample/
│   ├── extracted_recipes_sample.csv     # 30-row reproducible sample dataset
│   ├── step1_results_sample.csv         # 15-paper search results sample
│   └── gold_annotations.csv            # Manually annotated gold standard
│
└── README.md
```

---

## 🔬 What This Pipeline Does

1. **Search** — Queries the OpenAlex open-access literature API with
   domain-specific queries (perovskite + inkjet + solvent keywords).

2. **Rank** — Scores candidate papers using title TF-IDF + citation features
   before downloading, to prioritise the most likely recipe-containing papers.

3. **Download** — Fetches open-access PDFs with retry logic and exponential
   backoff.

4. **Extract** — Uses an LLM (Claude claude-sonnet-4-20250514) with structured prompts to
   extract ink recipes from PDF text: formula, solvent, ratio, molarity,
   additives, annealing conditions.

5. **Preprocess** — Normalises chemical names, parses formulas, canonicalises
   solvent systems, cleans ratios, classifies additives.

6. **Score & Cluster** — Assigns recipe completeness scores (0–10) and uses
   K-means to cluster chemistry space into interpretable groups.

7. **ML Models** — Trains logistic regression classifiers to rank future papers
   and a Ridge regression model to predict recipe quality.

8. **Search & Recommend** — Provides Jaccard-similarity search and
   constraint-based recommendation over the extracted dataset.

9. **Design Exploration** — Grid-searches novel formulation candidates scored
   by the trained regression model.

---

## 🚀 Quick Start

### Requirements

```bash
pip install pandas numpy scikit-learn requests pathlib
```

For LLM extraction steps, set your Anthropic API key:
```bash
export ANTHROPIC_API_KEY=your_key_here
```

### Run the search step on sample data

```bash
python scripts/step1_search_openalex.py
```

### Run evaluation on sample data

```bash
python evaluation/evaluate_extraction.py \
    --gold  data_sample/gold_annotations.csv \
    --pred  data_sample/extracted_recipes_sample.csv \
    --out   evaluation_report.txt

python evaluation/evaluate_classifiers.py \
    --tidy  data_sample/extracted_recipes_sample.csv \
    --step1 data_sample/step1_results_sample.csv
```

### Run full pipeline (requires API key + internet)

```bash
python scripts/run_all.py
```

---

## 📊 Sample Dataset

`data_sample/extracted_recipes_sample.csv` contains 30 representative records
with the following fields:

| Field | Description |
|-------|-------------|
| `pdf_file` | Source paper identifier |
| `perovskite_formula` | Chemical formula (e.g. MAPbI3, Cs0.05FA0.81MA0.14PbI2.55Br0.45) |
| `solvent_system_final` | Canonical solvent system (e.g. DMF:DMSO, GBL:DMSO) |
| `solvent_ratio_final` | Volumetric ratio (e.g. 4:1, 9:1) |
| `molarity_M` | Total precursor molarity in mol/L |
| `additives_found` | Additives or dopants used |
| `recipe_class` | minimal / partial / complete |
| `recipe_score` | Completeness score 0–10 |
| `recipe_tier` | bronze / silver / gold |
| `formula_family` | I-rich / Br-rich / Cl-containing |
| `cluster` | Chemistry cluster ID (0–3) |

---

## 🧪 Evaluation Metrics

| Stage | Metric | Target |
|-------|--------|--------|
| Extraction (field-level) | Macro F1 | ≥ 0.80 |
| Pre-download ranking | P@50 | ≥ 0.40 |
| Gold classifier | ROC-AUC | ≥ 0.85 |
| Recipe score regression | R² | ≥ 0.70 |

---

## 📖 LLM Prompts

All prompts used for extraction are documented in
`prompts/llm_extraction_prompts.md`. Five prompts cover:

- **Prompt 1**: Recipe extraction (main)
- **Prompt 2**: Recipe quality scoring
- **Prompt 3**: Solvent system normalisation
- **Prompt 4**: Additive identification and classification
- **Prompt 5**: Paper relevance pre-filter (title + abstract)

---

## 🗂️ Data Sources

Literature is retrieved from [OpenAlex](https://openalex.org/) — a free,
open, and comprehensive index of scientific literature. Only open-access papers
are downloaded. No proprietary publisher APIs are used.

---

## 📄 Citation

If you use this methodology or code in your research, please cite:

```bibtex
@misc{perovskite_ink_mining_2024,
  title   = {Automated Mining of Perovskite Ink Recipes from Scientific Literature},
  author  = {[Author names]},
  year    = {2024},
  url     = {https://github.com/[your-github]/perovskite_ink_public},
  note    = {Methodology code, prompts, and reproducible dataset}
}
```

---

## ⚖️ License

The code in this repository is released under the **MIT License**.  
The sample dataset (`data_sample/`) is released under **CC BY 4.0**.  
See `LICENSE` for details.

---

## 🔒 What Is NOT Included

The following are part of the production SaaS platform and are **not** in
this public repository:

- Full extracted dataset (thousands of recipes)
- Production API and web interface
- Proprietary post-processing and deduplication logic
- Trained model weights (regenerate from scripts)

---

## Contact

For questions about the methodology, open a GitHub Issue.
