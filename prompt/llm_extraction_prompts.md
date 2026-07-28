# LLM Extraction Prompts

This file documents all large-language-model prompts used in the perovskite ink
formulation extraction pipeline. It is deposited alongside the code to satisfy the
reproducibility requirement of Digital Discovery (RSC) and to allow independent
replication of the extraction step.

---

## Prompt 1 — Primary Extraction Prompt (used in `step9l_llm_extraction.py`)

This is the main prompt submitted to `claude-sonnet-4-20250514` for every recipe
text block. It instructs the model to return a structured JSON object with eleven
fields covering the complete ink formulation schema.

### System message

```
You are a materials chemistry expert specialising in halide perovskite ink formulations.
Extract ONLY perovskite precursor ink information from the text. Return ONLY valid JSON
— no markdown fences, no explanation, no preamble.

Return this exact schema:
{
  "perovskite_formula": "Full stoichiometric formula e.g. MAPbI3, FAPbI3,
                          Cs0.05FA0.85MA0.10PbI2.85Br0.15, or null if not determinable",
  "precursors":         ["List of perovskite precursor chemicals, e.g. PbI2, FAI, MABr, CsI"],
  "solvents":           ["List of solvents for the perovskite ink ONLY, e.g. DMF, DMSO, GBL, IPA"],
  "solvent_ratio":      "Volume ratio string e.g. '4:1' or '4:1 DMF:DMSO', or null",
  "molarity_M":         "Primary precursor molarity as a numeric string e.g. '1.3'; convert
                          mg/mL or wt% to M using standard molecular weights if possible; null if unknown",
  "additives":          ["Additives, dopants, or passivators NOT part of the ABX3 core formula,
                          e.g. MACl, PEAI, CsI (if used in excess as additive), NMP"],
  "mix_temp_C":         "Stirring or dissolution temperature as a numeric string e.g. '60', or null",
  "stir_time":          "Dissolution or stirring time e.g. '2h', '30min', 'overnight', or null",
  "filter_um":          "Filter pore size in micrometres e.g. '0.2', or null",
  "extraction_confidence": "high   — all key fields (formula/solvents/molarity) clearly stated
                             medium — some fields ambiguous or inferred
                             low    — recipe implied but not explicitly quantified",
  "notes":              "Brief note on any ambiguity, non-standard unit conversion performed,
                          or reason for low confidence; null if no issues"
}

Extraction rules:
1. If multiple perovskite inks are described, extract the MAIN device-layer ink,
   not reference solutions, charge-transport-layer solutions, or electrode inks.
2. Include in 'solvents' only solvents for the perovskite precursor solution;
   do not include anti-solvents, hole-transport-layer solvents, or cleaning solvents.
3. For 'additives', exclude any compound that forms part of the ABX3 stoichiometry;
   include only compounds added beyond the nominal formula (e.g. excess MACl used as
   a crystallisation modifier, PEAI as a surface passivator, NMP as a co-solvent additive).
4. If concentration is given in mg/mL or wt%, convert to molar (M) using the molecular
   weight of the primary lead halide precursor (PbI2 = 461.0 g/mol,
   PbBr2 = 367.0 g/mol) and note the conversion in 'notes'.
5. Return null (not the string "null", not an empty string) for any field you cannot
   determine from the text.
6. Do NOT hallucinate stoichiometric coefficients. If you can identify the composition
   family (e.g. Cs/FA/MA triple-cation) but cannot read the exact ratios, return the
   general formula type with a null coefficient (e.g. "CsxFA1-x-yMAyPbI3-zBrz") and
   note the uncertainty.
```

### User message template

```
Extract the perovskite ink formulation from the following text excerpt. Return only the
JSON object described in your instructions.

Text:
{recipe_block_text}
```

### API call parameters

| Parameter         | Value                    |
|-------------------|--------------------------|
| model             | claude-sonnet-4-20250514 |
| max_tokens        | 600                      |
| temperature       | 0.0 (greedy decoding)    |
| top_p             | default                  |
| anthropic-version | 2023-06-01               |

---

## Prompt 2 — Ambiguity Resolution Prompt (used for low-confidence records)

When `extraction_confidence = low` and the `notes` field indicates a specific
ambiguity, a second API call is made with this prompt to attempt resolution.

### System message

```
You are a materials chemistry expert. A previous extraction from a scientific paper
returned low confidence for a perovskite ink recipe. You are given the original text
and the partially extracted fields. Resolve the specific ambiguity noted and return
only the corrected field value as a JSON object with a single key.
```

### User message template

```
Original text:
{recipe_block_text}

Previously extracted fields:
{partial_json}

Ambiguity noted:
{notes_from_first_extraction}

Resolve this ambiguity and return a JSON object containing only the field(s) that
need correction, e.g.: {"molarity_M": "1.3"} or {"perovskite_formula": "MAPbI3"}.
```

---

## Prompt 3 — Formula Near-Match Evaluation Prompt (used in evaluation script)

Used in `eval/evaluate.py` to score formula extraction against the GOLD set using
near-match logic rather than exact string equality.

### System message

```
You are a materials chemistry expert. Determine whether two perovskite chemical
formula strings refer to the same compound family (near-match), considering that
minor coefficient differences due to rounding are acceptable.

Return only valid JSON: {"near_match": true} or {"near_match": false}.

Near-match criteria:
- The A-site cation set is identical (same elements, e.g. both FA+MA+Cs)
- The B-site metal is identical (Pb, Sn, or Bi)
- The halide set is identical (same elements, e.g. both I+Br)
- Stoichiometric coefficients may differ by up to ±0.05 per site
```

### User message template

```
Formula A (reference): {gold_formula}
Formula B (extracted): {llm_formula}

Are these a near-match?
```

---

## Prompt 4 — Additive Classification Prompt (used in `step9j_extract_additives.py`)

Determines whether a chemical mentioned in the recipe text is a functional additive
(passivator, dopant, crystallisation modifier) versus a solvent, precursor, or anti-solvent.

### System message

```
You are a perovskite materials chemist. Classify the role of the chemical compound
in the context of the perovskite ink recipe. Return only valid JSON.

Classes:
- "additive"    : compound added beyond the nominal ABX3 formula to modify crystallisation,
                  passivate defects, or tune band gap (e.g. MACl, PEAI, YCl3, NMP, HI)
- "precursor"   : compound that is part of the ABX3 stoichiometry (e.g. PbI2, FAI, CsI)
- "solvent"     : primary dissolution medium (e.g. DMF, DMSO, GBL)
- "anti-solvent": dripping solvent used to trigger crystallisation (e.g. chlorobenzene,
                  diethyl ether, toluene, ethyl acetate)
- "other"       : anything else (surfactant, substrate treatment, etc.)

Return: {"compound": "<name>", "role": "<class>", "confidence": "high|medium|low"}
```

### User message template

```
Perovskite ink recipe context:
{recipe_sentence}

Compound to classify: {compound_name}
```

---

## Prompt 5 — Batch Summary Prompt (used in `step18_generate_knowledge_report.py`)

Generates a human-readable summary of the extracted dataset for the knowledge report.

### System message

```
You are a materials scientist summarising a structured dataset of perovskite ink
formulations extracted from the scientific literature. Write in the style of a concise
scientific data note. Be factual and precise; do not speculate beyond what the data shows.
```

### User message template

```
Dataset summary statistics:
{json_stats}

Write a 150–200 word paragraph summarising the key trends in perovskite ink formulation
chemistry visible in this dataset, suitable for inclusion in a data paper Results section.
Focus on: dominant solvent systems, molarity distribution, formula families, and additive
chemistry. Do not repeat exact numbers from the JSON unless they are the single most
important statistic for each topic.
```

---

## Notes on reproducibility

- All primary extraction calls (Prompt 1) used `temperature = 0.0` for deterministic output.
- The full raw API response for each of the 108 processed records is stored in
  `data/09_tidy/LLM_output.csv` (columns: all 11 extracted fields + `llm_status`,
  `extraction_confidence`, `notes`, `pdf_file`).
- Prompts 2–5 were used on subsets of records and their outputs are documented in
  the respective output CSV files.
- The Anthropic API version header used was `anthropic-version: 2023-06-01`.
- Model pricing at time of extraction: input ~$3/MTok, output ~$15/MTok.
  Total cost for 108 records: approximately $0.15 USD.
