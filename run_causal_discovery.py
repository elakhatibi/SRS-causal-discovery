#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import networkx as nx

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from notears_linear import notears_linear


# ==========================
# Dataset-specific schemas
# ==========================

MESA_PSG_PHYSIO_COLS = [
    "TST_min", "SleepEfficiency_pct", "SOL_min", "WASO_min",
    "StageShiftIndex", "AwakeningIndex",
    "N1_pct", "N2_pct", "N3_pct", "REM_pct",
    "HypopneaIndex", "ObstructiveApneaIndex", "AHI",
    "DesaturationIndex", "ODI",
    "MeanSpO2", "MinSpO2", "T90_pct",
    "ArousalIndex", "PLMI",
    "SWA_NREM", "Sigma_NREM", "SWA_Early", "SWA_Late", "SWA_Decline_Ratio",
    "MeanHR_bpm", "MeanRR_ms", "SDNN_ms", "RMSSD_ms", "pNN50_pct",
]
MESA_CONFOUNDERS = [
    "sleepage5c", "race1c", "gender1", "bmi5c", "smkstat5",
    "wrksched5", "extrahrs5", "types5", "examnumber",
]
MESA_ID_COLS = ["subject_id"]

MROS_PSG_PHYSIO_COLS = [
    "tib_min", "tst_min", "tst_hours", "sleep_efficiency",
    "pct_n1", "pct_n2", "pct_n3", "pct_rem", "pct_wake",
    "sleep_latency_min", "rem_latency_min", "waso_min",
    "stage_transition_count",
    "ahi_total", "hypopnea_index", "arousal_index", "plm_index",
    "odi3", "odi4", "num_apnea_hypopnea_events", "num_arousals", "num_plm", "num_desats",
    "baseline_hr", "hrv_rmssd_ms", "hrv_sdnn_ms",
    "spo2_mean", "spo2_min", "spo2_t90_frac",
    "powers_c3_delta", "powers_c3_theta", "powers_c3_alpha", "powers_c3_sigma", "powers_c3_beta",
    "powers_c4_delta", "powers_c4_theta", "powers_c4_alpha", "powers_c4_sigma", "powers_c4_beta",
]
MROS_CONFOUNDERS = [
    "visit", "gender",
    "girace", "gierace", "gieduc", "gisoc",
    "tursmoke", "tusmkcgn", "tu12drin", "tudramt",
    "gimstat",
]
MROS_ID_COLS = ["subject_id"]


def get_schema(dataset: str):
    if dataset == "mesa":
        return MESA_PSG_PHYSIO_COLS, MESA_CONFOUNDERS, MESA_ID_COLS
    if dataset == "mros":
        return MROS_PSG_PHYSIO_COLS, MROS_CONFOUNDERS, MROS_ID_COLS
    raise ValueError("dataset must be 'mesa' or 'mros'")


# ==========================
# Data selection / cleaning
# ==========================

def select_frame(df: pd.DataFrame, outcome: str, dataset: str) -> pd.DataFrame:
    df = df.copy()
    if outcome not in df.columns:
        raise ValueError(f"Outcome '{outcome}' not found in dataset columns.")
    df["OUTCOME"] = df[outcome]

    psg_cols, conf_cols, id_cols = get_schema(dataset)

    for c in id_cols:
        if c in df.columns:
            df = df.drop(columns=[c])

    df = df.loc[~df["OUTCOME"].isna()].reset_index(drop=True)

    keep = {"OUTCOME"}
    keep.update([c for c in psg_cols if c in df.columns])
    keep.update([c for c in conf_cols if c in df.columns])

    return df[[c for c in df.columns if c in keep]].copy()


def drop_constant_and_duplicate_columns(df: pd.DataFrame, min_unique: int = 2) -> pd.DataFrame:
    df = df.copy()
    protected = {"OUTCOME"}

    to_drop = []
    for c in df.columns:
        if c in protected:
            continue
        if df[c].nunique(dropna=True) < min_unique:
            to_drop.append(c)
    if to_drop:
        df = df.drop(columns=to_drop)

    dup_mask = df.T.duplicated()
    if "OUTCOME" in df.columns:
        dup_mask.iloc[df.columns.get_loc("OUTCOME")] = False
    return df.loc[:, ~dup_mask]


# ==========================
# Preprocessing
# ==========================

def build_preprocessor(df: pd.DataFrame, categorical: List[str]) -> ColumnTransformer:
    categorical = [c for c in categorical if c in df.columns and c != "OUTCOME"]
    numeric = [c for c in df.columns if c not in categorical]

    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first")),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric),
            ("cat", cat_pipe, categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def fit_transform(df: pd.DataFrame, categorical: List[str]) -> Tuple[np.ndarray, List[str]]:
    pre = build_preprocessor(df, categorical)
    X = pre.fit_transform(df)
    names = list(pre.get_feature_names_out())
    return np.asarray(X, dtype=float), names


def find_outcome_feature_name(names: List[str]) -> str:
    if "OUTCOME" in names:
        return "OUTCOME"
    c = [n for n in names if n.endswith("__OUTCOME")]
    if c:
        return c[0]
    c = [n for n in names if "OUTCOME" in n]
    if c:
        return c[0]
    raise RuntimeError("OUTCOME not found after preprocessing.")


# ==========================
# NOTEARS + utilities
# ==========================

def run_notears(X: np.ndarray, lambda1: float) -> np.ndarray:
    return notears_linear(X, lambda1=lambda1, w_threshold=0.0)


def threshold_W_to_digraph(W: np.ndarray, names: List[str], thr: float) -> nx.DiGraph:
    G = nx.DiGraph()
    for n in names:
        G.add_node(n)
    for i, src in enumerate(names):
        for j, dst in enumerate(names):
            if i == j:
                continue
            w = W[i, j]
            if abs(w) >= thr:
                G.add_edge(src, dst, weight=float(w))
    return G


def parents_of(G: nx.DiGraph, node: str) -> List[str]:
    return sorted(list(G.predecessors(node))) if node in G else []


def ancestors_of(G: nx.DiGraph, node: str) -> List[str]:
    return sorted(list(nx.ancestors(G, node))) if node in G else []


def summarize_causes(G: nx.DiGraph, outcome_node: str) -> Tuple[List[str], List[str]]:
    direct = parents_of(G, outcome_node)
    indirect = sorted(list(set(ancestors_of(G, outcome_node)) - set(direct) - {outcome_node}))
    return direct, indirect


def topk_incoming(W: np.ndarray, names: List[str], target: str, k: int = 20):
    if target not in names:
        return []
    j = names.index(target)
    arr = []
    for i, src in enumerate(names):
        if i == j:
            continue
        w = float(W[i, j])
        if np.isnan(w) or np.isinf(w):
            continue
        arr.append((src, w))
    arr.sort(key=lambda t: abs(t[1]), reverse=True)
    return arr[:k]


def export_results_json(path: str, payload: Dict[str, Any]):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n✅ Exported discovery JSON to: {path}")


# ==========================
# Main
# ==========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--dataset", default="mesa", choices=["mesa", "mros"])
    ap.add_argument("--outcome", required=True)
    ap.add_argument("--lambda1", type=float, default=0.02)
    ap.add_argument("--thr", type=float, default=0.01)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--max_rows", type=int, default=0)
    ap.add_argument("--export_json", default="")
    args = ap.parse_args()

    df_raw = pd.read_csv(args.csv)

    df = select_frame(df_raw, outcome=args.outcome, dataset=args.dataset)
    df = drop_constant_and_duplicate_columns(df)

    print(f"[INFO] Dataset={args.dataset} | Outcome={args.outcome} | Rows={len(df)} | Cols_kept={len(df.columns)}")

    if args.max_rows and args.max_rows > 0:
        df = df.iloc[:args.max_rows].copy()

    # Auto-detect categorical columns: any non-numeric dtype except OUTCOME
    categorical = [
        c for c in df.columns
        if c != "OUTCOME" and not pd.api.types.is_numeric_dtype(df[c])
    ]

    # Force include known categoricals if present
    if args.dataset == "mesa":
        for c in ["race1c", "gender1", "smkstat5", "wrksched5", "types5", "examnumber"]:
            if c in df.columns and c not in categorical:
                categorical.append(c)
    else:
        for c in ["gender", "girace", "gierace", "gieduc", "gisoc", "visit", "tursmoke", "tusmkcgn", "tu12drin"]:
            if c in df.columns and c not in categorical:
                categorical.append(c)

    if categorical:
        print("[INFO] Categorical columns detected:", categorical[:25], ("..." if len(categorical) > 25 else ""))

    X, names = fit_transform(df, categorical=categorical)
    outcome_node = find_outcome_feature_name(names)

    W = run_notears(X, lambda1=args.lambda1)
    incoming = topk_incoming(W, names, outcome_node, k=args.topk)
    G_thr = threshold_W_to_digraph(W, names, thr=args.thr)
    direct, indirect = summarize_causes(G_thr, outcome_node)

    print("\n" + "=" * 90)
    print(f"NOTEARS dataset={args.dataset} outcome={args.outcome} (outcome_node={outcome_node})")
    print(f"Top-{args.topk} incoming edges into OUTCOME (src -> OUTCOME, weight):")
    for src, w in incoming:
        print(f"  {src} -> {outcome_node}   w={w:.4f}")
    print(f"\nThresholded graph (thr={args.thr}) parents of OUTCOME:\n{direct}")
    print(f"\nThresholded graph (thr={args.thr}) indirect ancestors of OUTCOME:\n{indirect}")

    if args.export_json:
        payload = {
            "dataset": args.dataset,
            "outcome": args.outcome,
            "outcome_node": outcome_node,
            "direct": direct,
            "indirect": indirect,
            "top_incoming": [[a, float(b)] for a, b in incoming],
            "extras": {
                "args": vars(args),
                "cols_kept": list(df.columns),
                "categorical_detected": categorical,
            }
        }
        export_results_json(args.export_json, payload)


if __name__ == "__main__":
    main()
