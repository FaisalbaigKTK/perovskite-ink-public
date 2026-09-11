"""
step4L_llm_enhance.py
======================
Pipeline stage: 4L  (runs after step4c_refine_best_recipe.py)

Input:  data/03_extract/step4c_best_recipe_extracted.csv
Output: data/03_extract/LLM_output.csv

Purpose
-------
Submits each recipe text block to Claude Sonnet (claude-sonnet-4-20250514)
and extracts 11 structured fields that regex alone cannot recover:
full perovskite formula, cleaned molarity, functional additives, etc.

Requirements
------------
    pip install requests pandas
    export ANTHROPIC_API_KEY=sk-ant-...   (Linux/macOS)
    set  ANTHROPIC_API_KEY=sk-ant-...     (Windows CMD)

Usage
-----
    python scripts/step4L_llm_enhance.py

If ANTHROPIC_API_KEY is not set, the script exits immediately with a clear
error message rather than sending 108 failed API requests.
"""

import json
import os
import time
import requests
import pandas as pd
from pathlib import Path

# ── Paths (relative to repo root, one level above scripts/) ──────────────────
ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

INPUT  = DATA_DIR / "03_extract" / "step4c_best_recipe_extracted.csv"
OUTPUT = DATA_DIR / "03_extract" / "LLM_output.csv"

# ── API configuration ─────────────────────────────────────────────────────────
API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
API_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-haiku-4-5-20251001"

# ── Extraction prompt (full text deposited in prompts/llm_extraction_prompts.md) ─
SYSTEM_PROMPT = """You are a materials chemistry expert specialising in halide perovskite ink formulations.
Extract ONLY perovskite precursor ink information from the text. Return ONLY valid JSON — no markdown fences, no explanation.

Return this exact schema:
{
  "perovskite_formula": "full stoichiometric formula e.g. MAPbI3, Cs0.05FA0.85MA0.10PbI2.85Br0.15, or null",
  "precursors": ["list of precursor chemicals e.g. PbI2, FAI, MABr, CsI"],
  "solvents": ["list of solvents for the perovskite ink ONLY e.g. DMF, DMSO, GBL"],
  "solvent_ratio": "v/v ratio string e.g. 4:1 or null",
  "molarity_M": "primary precursor molarity as numeric string e.g. 1.3 or null",
  "additives": ["additives not part of ABX3 formula e.g. MACl, PEAI, YCl3"],
  "mix_temp_C": "stirring temperature numeric string e.g. 60 or null",
  "stir_time": "dissolution time e.g. 2h, 30min, overnight or null",
  "filter_um": "filter pore size e.g. 0.2 or null",
  "extraction_confidence": "high / medium / low",
  "notes": "brief note on ambiguity or unit conversion or null"
}

Rules:
- Extract ONLY the main perovskite precursor ink, not charge-transport or reference solutions
- Include in 'solvents' only solvents for the perovskite ink
- Convert mg/mL or wt% to M using standard molecular weights and note this
- Return null (not the string null) for any field you cannot determine
- Do NOT hallucinate stoichiometric coefficients; use variable subscripts if uncertain"""


def llm_extract(block_text: str) -> dict:
    """Submit one recipe block to Claude API and return parsed JSON.

    Args:
        block_text: Raw recipe text block (up to 3000 chars used).

    Returns:
        Parsed dict with 11 extraction fields.

    Raises:
        requests.HTTPError: Non-2xx API response.
        json.JSONDecodeError: Model returned non-JSON.
    """
    text = block_text[:3000]   # stay within output token budget
    payload = {
        "model":       MODEL,
        "max_tokens":  600,
        "temperature": 0.0,    # greedy decoding — deterministic, reproducible
        "system":      SYSTEM_PROMPT,
        "messages":    [{"role": "user",
                         "content": f"Extract the perovskite ink formulation:\n\n{text}"}]
    }
    headers = {
        "Content-Type":      "application/json",
        "x-api-key":         API_KEY,
        "anthropic-version": "2023-06-01",
    }
    resp = requests.post(API_URL, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()

    raw = resp.json()["content"][0]["text"].strip()
    # Strip accidental markdown fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def main() -> None:
    # ── Pre-flight checks ─────────────────────────────────────────────────────
    if not API_KEY or "PASTE" in API_KEY:
        print(
            "\n[step4L] ERROR: ANTHROPIC_API_KEY is not set.\n"
            "  Set it with:  export ANTHROPIC_API_KEY=sk-ant-...\n"
            "  Or run the pipeline with:  python scripts/run_all.py --skip-llm\n"
        )
        raise SystemExit(1)

    if not INPUT.exists():
        print(f"[step4L] ERROR: Input not found: {INPUT}")
        print("  Run step4c_refine_best_recipe.py first.")
        raise SystemExit(1)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    # ── Load and process ──────────────────────────────────────────────────────
    df = pd.read_csv(INPUT)
    print(f"[step4L] Loaded {len(df)} records from {INPUT.name}")

    results = []
    n_ok = n_skip = n_err = 0

    for i, row in df.iterrows():
        block = str(row.get("best_recipe_block", ""))

        if len(block.strip()) < 50:
            print(f"  [{i+1}/{len(df)}] SKIP — block too short")
            results.append({"pdf_file": row.get("pdf_file", ""), "llm_status": "skipped"})
            n_skip += 1
            continue

        try:
            extracted = llm_extract(block)
            extracted["pdf_file"]         = row.get("pdf_file", "")
            extracted["llm_status"]       = "ok"
            # Carry regex fields for side-by-side comparison in eval/
            extracted["regex_precursors"] = row.get("precursors", "")
            extracted["regex_solvents"]   = row.get("solvents",   "")
            extracted["regex_molarity"]   = row.get("molarity",   "")
            extracted["regex_ratios"]     = row.get("ratios",     "")
            results.append(extracted)
            conf    = extracted.get("extraction_confidence", "?")
            formula = extracted.get("perovskite_formula", "null")
            print(f"  [{i+1}/{len(df)}] OK   conf={conf:<7} formula={formula}")
            n_ok += 1

        except Exception as e:
            print(f"  [{i+1}/{len(df)}] ERROR — {e}")
            results.append({"pdf_file": row.get("pdf_file", ""), "llm_status": f"error:{e}"})
            n_err += 1

        time.sleep(0.3)   # polite rate limit — stays within Anthropic tier-1 (50 req/min)

    # ── Save ──────────────────────────────────────────────────────────────────
    out = pd.DataFrame(results)
    out.to_csv(OUTPUT, index=False)

    ok  = out[out["llm_status"] == "ok"]
    print(f"\n[step4L] === Summary ===")
    print(f"  Processed : {len(df)}")
    print(f"  OK        : {n_ok}")
    print(f"  Skipped   : {n_skip}")
    print(f"  Errors    : {n_err}")

    if len(ok):
        has_f = ok["perovskite_formula"].notna() & ~ok["perovskite_formula"].isin(["null", ""])
        has_m = ok["molarity_M"].notna()          & ~ok["molarity_M"].isin(["null", ""])
        print(f"  Formula extracted : {has_f.sum()} / {len(ok)} ({has_f.sum()/len(ok)*100:.0f}%)")
        print(f"  Molarity extracted: {has_m.sum()} / {len(ok)} ({has_m.sum()/len(ok)*100:.0f}%)")
        print(f"  Confidence breakdown:")
        print(ok["extraction_confidence"].value_counts().to_string())

    print(f"\n[step4L] Saved {len(out)} rows -> {OUTPUT}")


if __name__ == "__main__":
    main()
