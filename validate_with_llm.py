#!/usr/bin/env python3
"""
Validate causal discovery results using:
1) Rule-based sleep physiology checks
2) Optional OpenAI LLM validation (robust JSON output)

Setup:
  pip install python-dotenv openai>=1.0.0

.env example:
  OPENAI_API_KEY=sk-...
  OPENAI_MODEL=gpt-5.2   # or gpt-4o, etc.

Usage:
  python validate_with_llm.py --in outputs/discovery_epss.json --out outputs/validation_epss.json --use_llm 0
  python validate_with_llm.py --in outputs/discovery_epss.json --out outputs/validation_epss.json --use_llm 1
"""

from __future__ import annotations
import argparse
import json
import os
from typing import Dict, List, Any

from dotenv import load_dotenv
load_dotenv()


# -----------------------
# Domain knowledge rules
# -----------------------

PLAUSIBLE_FACTORS = {
    "AHI": "OSA severity increases sleepiness via hypoxia and arousals.",
    "ObstructiveApneaIndex": "Obstructive apneas cause hypoxia and fragmentation.",
    "HypopneaIndex": "Hypopneas contribute to desaturation/arousals.",
    "DesaturationIndex": "Oxygen desaturations impair restorative sleep.",
    "ODI": "Oxygen desaturation burden correlates with sleepiness.",
    "T90": "Sustained hypoxia (SpO2 < 90%) worsens recovery.",
    "MeanSpO2": "Lower mean oxygenation reflects hypoxic burden.",
    "MinSpO2": "Lower minimum oxygenation reflects hypoxic burden.",
    "ArousalIndex": "Frequent arousals fragment sleep and worsen recovery.",
    "AwakeningIndex": "Night awakenings reduce sleep continuity.",
    "StageShiftIndex": "Frequent stage transitions reflect unstable/fragmented sleep.",
    "WASO": "Wake after sleep onset reflects poor sleep maintenance.",
    "SleepEfficiency": "Lower efficiency reduces restorative sleep.",
    "TST": "Short total sleep time increases sleepiness.",
    "SOL": "Longer sleep onset latency suggests insomnia → worse recovery.",
    "REM_pct": "Reduced REM can affect cognitive/emotional recovery.",
    "N3_pct": "Reduced deep sleep weakens physiological restoration.",
    "N1_pct": "Higher light sleep often reflects poorer quality/fragmentation.",
    "RMSSD": "Lower vagal tone indicates poorer autonomic recovery.",
    "SDNN": "HRV reflects autonomic balance; plausible recovery marker.",
    "MeanHR": "Higher nocturnal HR can reflect sympathetic activation/stress.",
    "MeanRR": "RR relates to HR/autonomic tone; interpret with HRV.",
    "PLMI": "Periodic limb movements can fragment sleep and increase sleepiness.",
    "SWA": "Slow-wave activity reflects homeostatic recovery.",
    "bmi": "BMI increases OSA risk and cardiometabolic stress.",
    "wrksched": "Shift work disrupts circadian rhythm → sleepiness.",
    "extrahrs": "Extra work hours proxy sleep restriction/stress."
}

POTENTIAL_CONFOUNDERS = {
    "race": "Structural variable; may reflect confounding/heterogeneity, not mechanistic.",
    "gender": "Sex/gender differences may reflect confounding/heterogeneity.",
    "sleepage": "Age is structural; affects sleep architecture and baseline sleepiness."
}

LIKELY_LEAKAGE = {
    "sleepy5": "Overlaps strongly with ESS (measurement leakage).",
    "tired5": "Fatigue overlaps with ESS construct."
}


def rule_validate(feature: str, outcome: str) -> Dict[str, str]:
    f = feature.lower()

    for k, msg in LIKELY_LEAKAGE.items():
        if k in f:
            return {"label": "Likely leakage", "reason": msg}

    for k, msg in POTENTIAL_CONFOUNDERS.items():
        if k in f:
            return {"label": "Needs caution", "reason": msg}

    for k, msg in PLAUSIBLE_FACTORS.items():
        if k.lower() in f:
            return {"label": "Plausible", "reason": msg}

    return {"label": "Needs caution", "reason": "No matched rule; requires review."}


# -----------------------
# LLM validation (robust)
# -----------------------

def llm_validate_edges_json(edges: List[str], outcome: str) -> Dict[str, Dict[str, str]]:
    """
    Forces the LLM to return strict JSON in the format:
    {
      "<feature>": {"label": "...", "reason": "..."},
      ...
    }
    """
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found. Put it in .env")

    client = OpenAI(api_key=api_key)

    instruction = (
        "You are a sleep physiology + causal inference expert.\n"
        f"Outcome: {outcome} (if epslpscl5c, that's Epworth Sleepiness Scale).\n"
        "For EACH candidate factor, return a JSON object mapping the EXACT factor string to:\n"
        '  {"label": "Plausible|Needs caution|Likely leakage", "reason": "<1-2 sentences>"}\n'
        "Rules:\n"
        "- Use 'Likely leakage' if the factor measures the same construct as the outcome.\n"
        "- Use 'Needs caution' for structural covariates (race/sex/age) or likely confounding.\n"
        "- Otherwise use 'Plausible' if there is a reasonable physiological pathway.\n"
        "Return ONLY valid JSON. No markdown, no extra text."
    )

    payload = {
        "role": "user",
        "content": instruction + "\n\nCandidates:\n" + "\n".join(edges)
    }

    resp = client.chat.completions.create(
        model=model,
        messages=[payload],
        temperature=0.1
    )

    txt = resp.choices[0].message.content.strip()

    # Robust JSON parse
    try:
        obj = json.loads(txt)
        if not isinstance(obj, dict):
            raise ValueError("LLM did not return a JSON object.")
        # light schema check
        cleaned = {}
        for k, v in obj.items():
            if isinstance(v, dict) and "label" in v and "reason" in v:
                cleaned[k] = {"label": str(v["label"]), "reason": str(v["reason"])}
        return cleaned
    except Exception as e:
        raise RuntimeError(f"Failed to parse LLM JSON output. Raw output:\n{txt}\n\nParse error: {e}")


# -----------------------
# JSON input compatibility
# -----------------------

def extract_discovery_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Supports:
    - simple format: {direct, indirect, top_incoming}
    - exported format: {results: {notears: {direct, indirect}}, top_incoming}
    """
    outcome = data.get("outcome", "OUTCOME")

    if "results" in data and isinstance(data["results"], dict):
        notears = data["results"].get("notears", {})
        direct = notears.get("direct", [])
        indirect = notears.get("indirect", [])
    else:
        direct = data.get("direct", [])
        indirect = data.get("indirect", [])

    top_incoming = data.get("top_incoming", [])
    return {
        "outcome": outcome,
        "direct": direct if isinstance(direct, list) else [],
        "indirect": indirect if isinstance(indirect, list) else [],
        "top_incoming": top_incoming if isinstance(top_incoming, list) else [],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input discovery JSON")
    ap.add_argument("--out", dest="outp", required=True, help="Output validation JSON")
    ap.add_argument("--use_llm", type=int, default=0, help="0=rule-only, 1=OpenAI validation")
    ap.add_argument("--max_indirect", type=int, default=25, help="How many indirect candidates to include")
    args = ap.parse_args()

    with open(args.inp, "r") as f:
        raw = json.load(f)

    fields = extract_discovery_fields(raw)
    outcome = fields["outcome"]
    direct = fields["direct"]
    indirect = fields["indirect"]
    top_incoming = fields["top_incoming"]

    # Build candidate list: direct + top_incoming + first N indirect
    candidates: List[str] = []
    candidates.extend(direct)
    for item in top_incoming:
        if isinstance(item, list) and len(item) >= 1:
            candidates.append(item[0])
    candidates.extend(indirect[: max(0, args.max_indirect)])

    # dedupe preserve order
    seen = set()
    candidates = [c for c in candidates if not (c in seen or seen.add(c))]

    rule_results = {c: rule_validate(c, outcome) for c in candidates}

    report: Dict[str, Any] = {
        "outcome": outcome,
        "n_candidates": len(candidates),
        "candidates": candidates,
        "rule_validation": rule_results,
        "llm_validation": None,
        "meta": {
            "openai_model": os.getenv("OPENAI_MODEL", "gpt-4o-mini") if args.use_llm == 1 else None
        }
    }

    if args.use_llm == 1:
        llm_results = llm_validate_edges_json(candidates, outcome)
        report["llm_validation"] = llm_results

    with open(args.outp, "w") as f:
        json.dump(report, f, indent=2)

    print(f"✅ Validation report saved to: {args.outp}")
    print(f"Candidates validated: {len(candidates)}")
    if args.use_llm == 1:
        print(f"LLM model used: {report['meta']['openai_model']}")


if __name__ == "__main__":
    main()
