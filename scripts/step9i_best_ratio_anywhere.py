import pandas as pd
import re

INFILE = "final_perovskite_ink_recipes_ML_READY3.csv"
OUTFILE = "final_perovskite_ink_recipes_ML_READY4.csv"

# patterns
RE_PAIR = re.compile(r"\b(DMF|DMSO|GBL|GVL|NMP|ACN)\s*[:/]\s*(DMF|DMSO|GBL|GVL|NMP|ACN)\b", re.IGNORECASE)
RE_RATIO2 = re.compile(r"(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)")
RE_RATIO3 = re.compile(r"(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)")  # stock mixing

CTX_STRONG = ["v/v", "vol/vol", "vol/ vol", "volume ratio"]
CTX_MED = ["mixture", "mixed", "solvent", "in a", "ratio"]

def score_ratio(window: str, ratio_span_start: int) -> int:
    w = window.lower()
    s = 0
    # context bonuses if near ratio
    nearby = w[max(0, ratio_span_start-80): ratio_span_start+120]
    if any(c in nearby for c in CTX_STRONG):
        s += 10
    if any(c in nearby for c in CTX_MED):
        s += 6
    # penalize 3-part ratios (often solution mixing)
    if RE_RATIO3.search(nearby):
        s -= 12
    return s

def best_ratio_for_pair(block: str, pair: str):
    txt = block.replace("\n", " ")
    txt_l = txt.lower()
    pair_l = pair.lower()

    best = ("", -10**9)  # (ratio, score)

    # search all pair occurrences (could be multiple)
    for m in re.finditer(re.escape(pair_l), txt_l):
        center = m.start()
        start = max(0, center - 600)
        end = min(len(txt), center + 900)
        window = txt[start:end]
        window_l = window.lower()

        # find all ratios in this window
        for rm in RE_RATIO2.finditer(window):
            a, b = rm.group(1), rm.group(2)
            try:
                af = float(a); bf = float(b)
                if not (0 < af <= 20 and 0 < bf <= 20):
                    continue
            except:
                continue

            sc = score_ratio(window, rm.start())
            # prefer simple solvent ratios: common ones are 4:1, 1:4, 7:3, 2:1, 1:1
            if f"{a}:{b}" in ["4:1", "1:4", "7:3", "2:1", "1:1"]:
                sc += 2

            if sc > best[1]:
                best = (f"{a}:{b}", sc)

    return best[0]

def main():
    df = pd.read_csv(INFILE)

    systems = []
    ratios = []

    for _, row in df.iterrows():
        block = str(row.get("evidence_block", ""))
        if not block or block.lower() == "nan":
            systems.append("")
            ratios.append("")
            continue

        # pick the first explicit pair if exists
        m = RE_PAIR.search(block)
        if not m:
            systems.append("")
            ratios.append("")
            continue

        pair = f"{m.group(1).upper()}:{m.group(2).upper()}"
        systems.append(pair)

        ratio_best = best_ratio_for_pair(block, pair)
        ratios.append(ratio_best)

    df["solvent_system_best"] = systems
    df["solvent_ratio_best"] = ratios

    # Final: prefer best ratio; else keep previous final if single ratio
    def final_ratio(row):
        b = str(row.get("solvent_ratio_best","")).strip()
        if b:
            return b
        prev = str(row.get("solvent_ratio_final","")).strip()
        if ";" in prev:
            return ""
        return prev

    df["solvent_ratio_FINAL2"] = df.apply(final_ratio, axis=1)

    df.to_csv(OUTFILE, index=False)

    #print(df[["doi","solvent_system_best","solvent_ratio_best","solvent_ratio_FINAL2","solvent_ratio"]].head(20).to_string(index=False))
    def safe_cols(df, wanted):
        return [c for c in wanted if c in df.columns]
    preview_cols = safe_cols(df, [
        "doi", "pdf_file",
        "solvent_system_best", "solvent_ratio_best",
        "solvent_ratio_FINAL2", "solvent_ratio"])
    if preview_cols:
        print(df[preview_cols].head(20).to_string(index=False))
    else:
        print(df.head(20).to_string(index=False))


    print("\nSaved:", OUTFILE)

if __name__ == "__main__":
    main()
