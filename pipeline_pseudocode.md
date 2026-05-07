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
