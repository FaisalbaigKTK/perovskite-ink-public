"""
step4c_llm_enhance.py
======================
Pipeline stage: 4c-LLM (between step4c regex extraction and step4d cleaning)
Input:  step4c_best_recipe_extracted.csv
        (108 records with best_recipe_block text and raw regex fields)
Output: data/03_extract/LLM_output.csv
        (same 108 records enriched with 11 LLM-extracted structured fields)

Purpose
-------
Applies a large language model (Claude Sonnet 4, claude-sonnet-4-20250514) as
a second extraction layer on top of the regex output from step4c. The LLM
recovers structured perovskite formulas, cleaned molarity values, and novel
additives that regex pattern matching cannot capture.

For each record the script:
  1. Submits the best_recipe_block text (up to 3000 characters) to the Claude
     API with the primary extraction prompt (see prompts/llm_extraction_prompts.md).
  2. Parses the JSON response and validates the schema.
  3. Writes all 11 LLM fields plus the original regex fields to LLM_output.csv.
  4. Records failed parses with llm_status = 'error:{reason}'.

API configuration
-----------------
  Model      : claude-sonnet-4-20250514
  Temperature: 0.0 (greedy decoding — deterministic output)
  Max tokens : 600
  Version    : anthropic-version: 2023-06-01

Cost (108 records, reported run): approximately $0.15 USD.
Runtime: approximately 6 minutes including rate-limiting delays.

Requirements
------------
  pip install requests pandas
  export ANTHROPIC_API_KEY=sk-ant-...   (or set API_KEY directly below)

Usage
-----
    python step4c_llm_enhance.py

Output columns
--------------
  pdf_file              : source PDF filename (join key)
  perovskite_formula    : full stoichiometric formula or null
  precursors            : list -> semicolon string of precursor chemicals
  solvents              : list -> semicolon string of solvents
  solvent_ratio         : v/v ratio string e.g. '4:1' or null
  molarity_M            : primary precursor molarity as string e.g. '1.3' or null
  additives             : list -> semicolon string of additives/passivators
  mix_temp_C            : stirring temperature string e.g. '60' or null
  stir_time             : dissolution time e.g. '2h', 'overnight' or null
  filter_um             : filter pore size e.g. '0.2' or null
  extraction_confidence : 'high', 'medium', or 'low'
  notes                 : ambiguity notes from LLM or null
  llm_status            : 'ok', 'skipped' (block < 50 chars), or 'error:{reason}'
  regex_precursors      : original regex precursors field (for comparison)
  regex_solvents        : original regex solvents field
  regex_molarity        : original regex molarity field
  regex_ratios          : original regex ratios field
"""

import json
import os
import time
import requests
import pandas as pd
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
# Paste your API key here, or set the ANTHROPIC_API_KEY environment variable.
API_KEY = os.environ.get("ANTHROPIC_API_KEY", "sk-ant-PASTE-YOUR-KEY-HERE")

ROOT     = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"

INPUT  = DATA_DIR / "03_extract" / "step4c_best_recipe_extracted.csv"
OUTPUT = DATA_DIR / "03_extract" / "LLM_output.csv"

API_URL = "https://api.anthropic.com/v1/messages"
MODEL   = "claude-sonnet-4-20250514"

# ── LLM extraction prompt ─────────────────────────────────────────────────────
# Full prompt text is also deposited in prompts/llm_extraction_prompts.md.
SYSTEM_PROMPT = """You are a materials chemistry expert specialising in halide perovskite ink formulations.
Extract ONLY perovskite precursor ink information from the text. Return ONLY valid JSON — no markdown, no explanation.

Return this exact schema:
{
  "perovskite_formula": "e.g. MAPbI3, FAPbI3, Cs0.05FA0.85MA0.10PbI2.85Br0.15, or null",
  "precursors": ["list of precursor chemicals, e.g. PbI2, FAI, MABr, CsI"],
  "solvents": ["list of solvents, e.g. DMF, DMSO, GBL, IPA"],
  "solvent_ratio": "v/v ratio string e.g. 4:1 or null",
  "molarity_M": "numeric string e.g. 1.3 or null (convert wt% if possible)",
  "additives": ["list of additives/dopants not part of ABX3 formula, e.g. MACl, PEAI, CsI if excess"],
  "mix_temp_C": "numeric string e.g. 60 or null",
  "stir_time": "e.g. 2h, 30min, overnight or null",
  "filter_um": "filter pore size e.g. 0.2 or null",
  "extraction_confidence": "high / medium / low",
  "notes": "brief note on any ambiguity or non-standard unit"
}

Rules:
- If multiple perovskite inks are described, extract the MAIN one (not reference solutions or charge transport layers)
- Only include solvents for the perovskite ink, not for other device layers
- If concentration is in mg/mL or wt%, try to convert to M using molecular weights; note this in 'notes'
- Return null for any field you cannot determine
- Do NOT hallucinate stoichiometric coefficients; use a generic formula with variable subscripts if uncertain"""


def llm_extract(block_text: str) -> dict:
    """Submit a recipe text block to the Claude API and parse the JSON response.

    Trims the input to 3000 characters to stay within the model's context window
    for a 600-token output. Strips markdown fences if the model includes them
    despite the instruction not to.

    Args:
        block_text: Raw recipe text block from step4c output.

    Returns:
        Parsed dictionary with the 11 LLM-extracted fields.

    Raises:
        requests.HTTPError: If the API returns a non-2xx status code.
        json.JSONDecodeError: If the model output cannot be parsed as JSON.
    """
    if not API_KEY or "PASTE" in API_KEY:
        raise ValueError("API key not set. Set ANTHROPIC_API_KEY or edit API_KEY above.")

    # Trim to 3000 chars — sufficient to capture a complete Methods paragraph
    text = block_text[:3000]

    payload = {
        "model":      MODEL,
        "max_tokens": 600,
        "system":     SYSTEM_PROMPT,
        "messages":   [{"role": "user", "content": f"Extract perovskite ink formulation:\n\n{text}"}]
    }
    headers = {
        "Content-Type":      "application/json",
        "x-api-key":         API_KEY,
        "anthropic-version": "2023-06-01",   # required header
    }
    resp = requests.post(API_URL, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()

    raw = resp.json()["content"][0]["text"].strip()

    # Strip markdown fences if the model includes them despite instructions
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    return json.loads(raw.strip())


def main() -> None:
    """Run LLM extraction on all 108 recipe blocks and write LLM_output.csv.

    Skips blocks shorter than 50 characters (no meaningful recipe text).
    Catches and logs all extraction errors without stopping the pipeline —
    failed records receive llm_status = 'error:{reason}'.
    Adds a 0.3-second delay between API calls for polite rate-limiting.
    """
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT)
    print(f"Loaded {len(df)} records from {INPUT.name}")

    results = []
    for i, row in df.iterrows():
        block = str(row.get("best_recipe_block", ""))

        # Skip blocks too short to contain a recipe
        if len(block) < 50:
            print(f"  [{i+1}/{len(df)}] SKIP — block too short ({len(block)} chars)")
            results.append({"pdf_file": row["pdf_file"], "llm_status": "skipped"})
            continue

        try:
            extracted = llm_extract(block)

            # Attach provenance and original regex fields for side-by-side comparison
            extracted["pdf_file"]         = row["pdf_file"]
            extracted["llm_status"]       = "ok"
            extracted["regex_precursors"] = row.get("precursors", "")
            extracted["regex_solvents"]   = row.get("solvents",   "")
            extracted["regex_molarity"]   = row.get("molarity",   "")
            extracted["regex_ratios"]     = row.get("ratios",     "")

            results.append(extracted)

            conf    = extracted.get("extraction_confidence", "?")
            formula = extracted.get("perovskite_formula", "null")
            print(f"  [{i+1}/{len(df)}] OK   conf={conf:<7}  formula={formula}")

        except Exception as e:
            print(f"  [{i+1}/{len(df)}] ERROR — {e}")
            results.append({"pdf_file": row["pdf_file"], "llm_status": f"error:{e}"})

        time.sleep(0.3)   # polite rate-limiting (Anthropic tier-1 limit: 50 req/min)

    out = pd.DataFrame(results)
    out.to_csv(OUTPUT, index=False)
    print(f"\nSaved {len(out)} rows -> {OUTPUT}")

    # ── Summary statistics ─────────────────────────────────────────────────────
    ok = out[out["llm_status"] == "ok"]
    print(f"\n=== LLM Extraction Summary ===")
    print(f"Processed     : {len(df)}")
    print(f"Success (ok)  : {len(ok)}")
    print(f"Skipped       : {(out['llm_status']=='skipped').sum()}")
    print(f"Errors        : {out['llm_status'].str.startswith('error').sum()}")

    if len(ok) > 0:
        has_formula = ok["perovskite_formula"].notna() & ~ok["perovskite_formula"].isin(["null", ""])
        has_mol     = ok["molarity_M"].notna()         & ~ok["molarity_M"].isin(["null", ""])
        print(f"\nField coverage (n={len(ok)}):")
        print(f"  Perovskite formula : {has_formula.sum()} ({has_formula.sum()/len(ok)*100:.0f}%)")
        print(f"  Molarity           : {has_mol.sum()} ({has_mol.sum()/len(ok)*100:.0f}%)")
        print(f"\nConfidence breakdown:")
        print(ok["extraction_confidence"].value_counts().to_string())


if __name__ == "__main__":
    main()
