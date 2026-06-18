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
🔬 What This Pipeline Does
Search — Queries the OpenAlex open-access literature API with domain-specific queries (perovskite + inkjet + solvent keywords).

Rank — Scores candidate papers using title TF-IDF + citation features before downloading, to prioritise the most likely recipe-containing papers.

Download — Fetches open-access PDFs with retry logic and exponential backoff.

Extract — Uses an LLM (Claude claude-sonnet-4-20250514) with structured prompts to extract ink recipes from PDF text: formula, solvent, ratio, molarity, additives, annealing conditions.

Preprocess — Normalises chemical names, parses formulas, canonicalises solvent systems, cleans ratios, classifies additives.

Score & Cluster — Assigns recipe completeness scores (0–10) and uses K-means to cluster chemistry space into interpretable groups.

ML Models — Trains logistic regression classifiers to rank future papers and a Ridge regression model to predict recipe quality.

Search & Recommend — Provides Jaccard-similarity search and constraint-based recommendation over the extracted dataset.

Design Exploration — Grid-searches novel formulation candidates scored by the trained regression model.

🚀 Quick Start
Requirements
pip install pandas numpy scikit-learn requests pathlib
For LLM extraction steps, set your Anthropic API key:

export ANTHROPIC_API_KEY=your_key_here
Run the search step on sample data
python scripts/step1_search_openalex.py
Run evaluation on sample data
python evaluation/evaluate_extraction.py \
    --gold  data_sample/gold_annotations.csv \
    --pred  data_sample/extracted_recipes_sample.csv \
    --out   evaluation_report.txt

python evaluation/evaluate_classifiers.py \
    --tidy  data_sample/extracted_recipes_sample.csv \
    --step1 data_sample/step1_results_sample.csv
Run full pipeline (requires API key + internet)
python scripts/run_all.py

