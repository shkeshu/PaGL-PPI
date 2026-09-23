from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


# =============================================================================
# Project paths
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

BASELINE_SCRIPT = ROOT / "Scripts" / "04_train_strong_symmetric_ablation.py"
SEGMENT_SCRIPT = ROOT / "Scripts" / "06_train_segment_cross_ablation.py"

BASELINE_CFG = ROOT / "Configs" / "strong_baseline.json"
SEGMENT_CFG = ROOT / "Configs" / "segment_cross_ablation.json"

NEW_OUT_ROOT = ROOT / "Results" / "final_seed_confirmation"
SUMMARY_OUT_ROOT = ROOT / "Results" / "fixed_fusion_5seed"

# 已有 3 个 seed + 新增 2 个 seed
OLD_SEEDS = [42, 123, 456]
NEW_SEEDS = [789, 1234]
ALL_SEEDS = OLD_SEEDS + NEW_SEEDS

# 已锁定，不再搜索
FUSION_ALPHA_B2 = 0.70
FUSION_ALPHA_B0 = 0.30


# =============================================================================
# Basic utilities
# =============================================================================

def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def run_command(cmd):
    print()
    print("=" * 120)
    print("RUN:")
    print(" ".join(f'"{x}"' if " " in str(x) else str(x) for x in cmd))
    print("=" * 120)
    subprocess.run(cmd, check=True)


def safe_logit(p, eps=1e-6):
    p = np.clip(
        np.asarray(p, dtype=float),
        eps,
        1.0 - eps,
    )
    return np.log(p / (1.0 - p))


def sigmoid(x):
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-x))


# =============================================================================
# Metrics
# =============================================================================

def calc_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    specificity = tn / max(tn + fp, 1)

    return {
        "threshold": float(threshold),
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "specificity": float(specificity),
        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "mcc": float(
            matthews_corrcoef(
                y_true,
                y_pred,
            )
        ),
        "auroc": float(
            roc_auc_score(
                y_true,
                y_prob,
            )
        ),
        "auprc": float(
            average_precision_score(
                y_true,
                y_prob,
            )
        ),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


# =============================================================================
# Prediction paths
# =============================================================================

def old_seed_prediction_paths(seed: int):
    """
    Existing results from previous experiments.

    seed=42:
      Results/strong_baseline/...
      Results/segment_cross_ablation/...

    seed=123/456:
      Results/multiseed_confirmation/seed_x/...
    """
    if seed == 42:
        b0 = (
            ROOT
            / "Results"
            / "strong_baseline"
            / "sum_diff_product"
            / "val_predictions.csv"
        )

        b2 = (
            ROOT
            / "Results"
            / "segment_cross_ablation"
            / "B2_segment_cross"
            / "val_predictions.csv"
        )

    else:
        b0 = (
            ROOT
            / "Results"
            / "multiseed_confirmation"
            / f"seed_{seed}"
            / "baseline"
            / "sum_diff_product"
            / "val_predictions.csv"
        )

        b2 = (
            ROOT
            / "Results"
            / "multiseed_confirmation"
            / f"seed_{seed}"
            / "segment"
            / "B2_segment_cross"
            / "val_predictions.csv"
        )

    return b0, b2


def new_seed_prediction_paths(seed: int):
    seed_dir = NEW_OUT_ROOT / f"seed_{seed}"

    b0 = (
        seed_dir
        / "baseline"
        / "sum_diff_product"
        / "val_predictions.csv"
    )

    b2 = (
        seed_dir
        / "segment"
        / "B2_segment_cross"
        / "val_predictions.csv"
    )

    return b0, b2


def prediction_paths(seed: int):
    if seed in OLD_SEEDS:
        return old_seed_prediction_paths(seed)

    return new_seed_prediction_paths(seed)


# =============================================================================
# Load predictions
# =============================================================================

def load_predictions(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Prediction file not found:\n{path}"
        )

    df = pd.read_csv(path)

    required = {
        "label",
        "probability",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{path}: missing columns {sorted(missing)}"
        )

    y = df["label"].to_numpy(dtype=int)
    p = df["probability"].to_numpy(dtype=float)

    if not np.isfinite(p).all():
        raise ValueError(
            f"{path}: probability contains NaN/Inf"
        )

    return y, p


# =============================================================================
# Run new seeds
# =============================================================================

def run_new_seed(seed: int, force=False):
    print()
    print("#" * 120)
    print(f"NEW SEED CONFIRMATION: {seed}")
    print("#" * 120)

    seed_dir = NEW_OUT_ROOT / f"seed_{seed}"
    cfg_dir = seed_dir / "configs"

    baseline_out = seed_dir / "baseline"
    segment_out = seed_dir / "segment"

    # -------------------------------------------------------------------------
    # B0 strong baseline
    # -------------------------------------------------------------------------
    baseline_cfg = read_json(BASELINE_CFG)

    baseline_cfg["seed"] = seed
    baseline_cfg["pair_modes"] = [
        "sum_diff_product"
    ]
    baseline_cfg["output_dir"] = str(
        baseline_out
    )

    baseline_cfg_path = (
        cfg_dir
        / "strong_baseline.json"
    )

    write_json(
        baseline_cfg_path,
        baseline_cfg,
    )

    baseline_result = (
        baseline_out
        / "sum_diff_product"
        / "best_val_metrics.json"
    )

    if force or not baseline_result.exists():
        run_command(
            [
                sys.executable,
                str(BASELINE_SCRIPT),
                "--config",
                str(baseline_cfg_path),
                "--only",
                "sum_diff_product",
            ]
        )
    else:
        print(
            f"[SKIP] B0 seed={seed} exists:\n"
            f"  {baseline_result}"
        )

    # -------------------------------------------------------------------------
    # B2 only
    # -------------------------------------------------------------------------
    segment_cfg = read_json(
        SEGMENT_CFG
    )

    segment_cfg["seed"] = seed
    segment_cfg["modes"] = [
        "B2_segment_cross"
    ]
    segment_cfg["output_dir"] = str(
        segment_out
    )

    segment_cfg_path = (
        cfg_dir
        / "segment_cross_ablation.json"
    )

    write_json(
        segment_cfg_path,
        segment_cfg,
    )

    segment_result = (
        segment_out
        / "B2_segment_cross"
        / "best_val_metrics.json"
    )

    if force or not segment_result.exists():
        run_command(
            [
                sys.executable,
                str(SEGMENT_SCRIPT),
                "--config",
                str(segment_cfg_path),
                "--only",
                "B2_segment_cross",
            ]
        )
    else:
        print(
            f"[SKIP] B2 seed={seed} exists:\n"
            f"  {segment_result}"
        )


# =============================================================================
# Fixed fusion
# =============================================================================

def fixed_logit_fusion(
    p_b0,
    p_b2,
):
    """
    Locked fusion:

        z_final =
            0.70 * z_B2
          + 0.30 * z_B0

        p_final = sigmoid(z_final)

    No alpha search is performed here.
    """
    z_b0 = safe_logit(p_b0)
    z_b2 = safe_logit(p_b2)

    z_final = (
        FUSION_ALPHA_B2 * z_b2
        + FUSION_ALPHA_B0 * z_b0
    )

    return sigmoid(z_final)


# =============================================================================
# Collect all five seeds
# =============================================================================

def collect_all_five_seeds():
    rows = []
    prediction_rows = []

    for seed in ALL_SEEDS:
        b0_path, b2_path = prediction_paths(
            seed
        )

        y0, p0 = load_predictions(
            b0_path
        )
        y2, p2 = load_predictions(
            b2_path
        )

        if len(y0) != len(y2):
            raise RuntimeError(
                f"seed={seed}: "
                "B0 and B2 prediction counts differ"
            )

        if not np.array_equal(y0, y2):
            raise RuntimeError(
                f"seed={seed}: "
                "B0/B2 validation labels or row order differ"
            )

        pf = fixed_logit_fusion(
            p_b0=p0,
            p_b2=p2,
        )

        m0 = calc_metrics(
            y0,
            p0,
            threshold=0.5,
        )

        m2 = calc_metrics(
            y0,
            p2,
            threshold=0.5,
        )

        mf = calc_metrics(
            y0,
            pf,
            threshold=0.5,
        )

        print()
        print("-" * 120)
        print(f"seed={seed}")
        print(
            f"  B0     AUPRC={m0['auprc']:.6f} "
            f"AUROC={m0['auroc']:.6f}"
        )
        print(
            f"  B2     AUPRC={m2['auprc']:.6f} "
            f"AUROC={m2['auroc']:.6f}"
        )
        print(
            f"  Fusion AUPRC={mf['auprc']:.6f} "
            f"AUROC={mf['auroc']:.6f}"
        )
        print(
            f"  Δ Fusion-B2 = "
            f"{mf['auprc'] - m2['auprc']:+.6f}"
        )
        print(
            f"  Δ Fusion-B0 = "
            f"{mf['auprc'] - m0['auprc']:+.6f}"
        )

        for model_name, m in [
            ("B0_strong", m0),
            ("B2_cross", m2),
            ("FixedFusion_0.70B2_0.30B0", mf),
        ]:
            row = {
                "seed": seed,
                "model": model_name,
                **m,
            }
            rows.append(row)

        prediction_rows.append(
            pd.DataFrame(
                {
                    "seed": seed,
                    "row_index": np.arange(
                        len(y0)
                    ),
                    "label": y0,
                    "prob_b0": p0,
                    "prob_b2": p2,
                    "prob_fixed_fusion": pf,
                }
            )
        )

    raw = pd.DataFrame(rows)
    predictions = pd.concat(
        prediction_rows,
        ignore_index=True,
    )

    return raw, predictions


# =============================================================================
# Summary
# =============================================================================

def summarize_models(
    raw: pd.DataFrame,
):
    metric_cols = [
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "mcc",
        "auroc",
        "auprc",
    ]

    rows = []

    for model, g in raw.groupby(
        "model"
    ):
        row = {
            "model": model,
            "n_seeds": int(len(g)),
        }

        for metric in metric_cols:
            row[f"{metric}_mean"] = float(
                g[metric].mean()
            )
            row[f"{metric}_std"] = float(
                g[metric].std(ddof=1)
            )
            row[f"{metric}_min"] = float(
                g[metric].min()
            )
            row[f"{metric}_max"] = float(
                g[metric].max()
            )

        rows.append(row)

    summary = pd.DataFrame(rows)

    return summary.sort_values(
        "auprc_mean",
        ascending=False,
    )


def paired_deltas(
    raw: pd.DataFrame,
):
    pivot = raw.pivot(
        index="seed",
        columns="model",
        values="auprc",
    )

    fusion = pivot[
        "FixedFusion_0.70B2_0.30B0"
    ]
    b2 = pivot[
        "B2_cross"
    ]
    b0 = pivot[
        "B0_strong"
    ]

    comparisons = {
        "Fusion - B2": fusion - b2,
        "Fusion - B0": fusion - b0,
        "B2 - B0": b2 - b0,
    }

    rows = []

    for name, delta in comparisons.items():
        rows.append(
            {
                "comparison": name,
                "n_seeds": int(
                    len(delta)
                ),
                "delta_auprc_mean": float(
                    delta.mean()
                ),
                "delta_auprc_std": float(
                    delta.std(ddof=1)
                ),
                "delta_min": float(
                    delta.min()
                ),
                "delta_max": float(
                    delta.max()
                ),
                "positive_seed_count": int(
                    (delta > 0).sum()
                ),
                "all_seeds_positive": bool(
                    (delta > 0).all()
                ),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--collect-only",
        action="store_true",
        help=(
            "Do not train seed 789/1234. "
            "Only collect existing prediction files."
        ),
    )

    ap.add_argument(
        "--force",
        action="store_true",
        help=(
            "Re-run seed 789/1234 even if result files already exist."
        ),
    )

    args = ap.parse_args()

    print("=" * 120)
    print("Bernett fixed-fusion 5-seed confirmation")
    print("=" * 120)
    print("Existing seeds :", OLD_SEEDS)
    print("New seeds      :", NEW_SEEDS)
    print("Fusion         : logit fusion")
    print(
        f"Locked weights : "
        f"B2={FUSION_ALPHA_B2:.2f}, "
        f"B0={FUSION_ALPHA_B0:.2f}"
    )
    print()
    print(
        "IMPORTANT: Fusion weight is FROZEN. "
        "No alpha search will be performed."
    )
    print(
        "IMPORTANT: Bernett Test is NOT accessed."
    )
    print("=" * 120)

    # -------------------------------------------------------------------------
    # Train only new seeds
    # -------------------------------------------------------------------------
    if not args.collect_only:
        for seed in NEW_SEEDS:
            run_new_seed(
                seed,
                force=args.force,
            )

    # -------------------------------------------------------------------------
    # Collect all five seeds
    # -------------------------------------------------------------------------
    raw, predictions = (
        collect_all_five_seeds()
    )

    summary = summarize_models(
        raw
    )

    deltas = paired_deltas(
        raw
    )

    SUMMARY_OUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_path = (
        SUMMARY_OUT_ROOT
        / "five_seed_raw_metrics.csv"
    )

    summary_path = (
        SUMMARY_OUT_ROOT
        / "five_seed_model_summary.csv"
    )

    delta_path = (
        SUMMARY_OUT_ROOT
        / "five_seed_auprc_deltas.csv"
    )

    pred_path = (
        SUMMARY_OUT_ROOT
        / "five_seed_validation_predictions.csv"
    )

    raw.to_csv(
        raw_path,
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    deltas.to_csv(
        delta_path,
        index=False,
        encoding="utf-8-sig",
    )

    predictions.to_csv(
        pred_path,
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------------------
    # Display
    # -------------------------------------------------------------------------
    print()
    print("=" * 120)
    print("FIVE-SEED MODEL SUMMARY")
    print("=" * 120)

    display_cols = [
        "model",
        "n_seeds",
        "auprc_mean",
        "auprc_std",
        "auroc_mean",
        "auroc_std",
        "f1_mean",
        "f1_std",
        "mcc_mean",
        "mcc_std",
    ]

    print(
        summary[
            display_cols
        ].to_string(
            index=False
        )
    )

    print()
    print("=" * 120)
    print("PAIRED AUPRC DELTAS")
    print("=" * 120)

    print(
        deltas.to_string(
            index=False
        )
    )

    # -------------------------------------------------------------------------
    # Final diagnostic
    # -------------------------------------------------------------------------
    pivot = raw.pivot(
        index="seed",
        columns="model",
        values="auprc",
    )

    fusion_gain_b2 = (
        pivot["FixedFusion_0.70B2_0.30B0"]
        - pivot["B2_cross"]
    )

    print()
    print("=" * 120)
    print("FINAL VALIDATION DECISION")
    print("=" * 120)

    print(
        f"Fusion > B2 in "
        f"{int((fusion_gain_b2 > 0).sum())}/5 seeds"
    )

    print(
        f"Mean ΔAUPRC Fusion-B2 = "
        f"{fusion_gain_b2.mean():+.6f}"
    )

    print(
        f"Minimum ΔAUPRC Fusion-B2 = "
        f"{fusion_gain_b2.min():+.6f}"
    )

    if (fusion_gain_b2 > 0).all():
        print()
        print(
            "DECISION: PASS — fixed late fusion improves B2 "
            "on all five validation seeds."
        )
        print(
            "Model/fusion design can now be frozen before "
            "the final Bernett Test stage."
        )
    elif (fusion_gain_b2 > 0).sum() >= 4:
        print()
        print(
            "DECISION: PARTIAL PASS — fusion improves B2 in "
            "at least 4/5 seeds, but inspect the failed seed "
            "before freezing the final model."
        )
    else:
        print()
        print(
            "DECISION: FAIL — fixed fusion is not sufficiently stable."
        )
        print(
            "Use B2 alone as the final candidate instead."
        )

    print()
    print("Saved:")
    print(" ", raw_path)
    print(" ", summary_path)
    print(" ", delta_path)
    print(" ", pred_path)

    print()
    print(
        "Bernett Test remains untouched."
    )


if __name__ == "__main__":
    main()
