from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

SEEDS = [42, 123, 456]


def get_paths(seed: int):
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


def safe_logit(p, eps=1e-6):
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def load_predictions(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    required = {"label", "probability"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{path}: missing columns {sorted(missing)}"
        )

    y = df["label"].to_numpy(dtype=int)
    p = df["probability"].to_numpy(dtype=float)

    if not np.isfinite(p).all():
        raise ValueError(f"{path}: probability contains NaN/Inf")

    return y, p


def score(y, p):
    return {
        "auprc": float(average_precision_score(y, p)),
        "auroc": float(roc_auc_score(y, p)),
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--alpha-step",
        type=float,
        default=0.05,
        help=(
            "alpha is the B2 weight. "
            "Fusion = alpha*B2 + (1-alpha)*B0."
        ),
    )

    args = ap.parse_args()

    if not (0 < args.alpha_step <= 0.5):
        raise ValueError("--alpha-step must be in (0, 0.5]")

    alphas = np.arange(
        0.0,
        1.0 + 1e-12,
        args.alpha_step,
    )

    seed_data = {}

    print("=" * 110)
    print("B0 + B2 validation late-fusion diagnostic")
    print("=" * 110)
    print("IMPORTANT: Bernett Test is NOT accessed.")
    print("alpha = B2 weight; (1-alpha) = B0 weight")
    print()

    for seed in SEEDS:
        b0_path, b2_path = get_paths(seed)

        y0, p0 = load_predictions(b0_path)
        y2, p2 = load_predictions(b2_path)

        if len(y0) != len(y2):
            raise RuntimeError(
                f"seed={seed}: prediction row counts differ"
            )

        if not np.array_equal(y0, y2):
            raise RuntimeError(
                f"seed={seed}: validation labels/order differ"
            )

        seed_data[seed] = {
            "y": y0,
            "b0": p0,
            "b2": p2,
        }

        s0 = score(y0, p0)
        s2 = score(y0, p2)

        print(
            f"seed={seed}: "
            f"B0 AUPRC={s0['auprc']:.6f}, "
            f"B2 AUPRC={s2['auprc']:.6f}"
        )

    print()

    rows = []

    for fusion_type in ["probability", "logit"]:
        for alpha in alphas:
            auprcs = []
            aurocs = []
            gains_vs_b0 = []
            gains_vs_b2 = []

            for seed, d in seed_data.items():
                y = d["y"]
                p0 = d["b0"]
                p2 = d["b2"]

                if fusion_type == "probability":
                    pf = (
                        alpha * p2
                        + (1.0 - alpha) * p0
                    )
                else:
                    z0 = safe_logit(p0)
                    z2 = safe_logit(p2)

                    pf = sigmoid(
                        alpha * z2
                        + (1.0 - alpha) * z0
                    )

                sf = score(y, pf)
                s0 = score(y, p0)
                s2 = score(y, p2)

                auprcs.append(sf["auprc"])
                aurocs.append(sf["auroc"])
                gains_vs_b0.append(
                    sf["auprc"] - s0["auprc"]
                )
                gains_vs_b2.append(
                    sf["auprc"] - s2["auprc"]
                )

            rows.append({
                "fusion_type": fusion_type,
                "alpha_b2": float(alpha),
                "alpha_b0": float(1.0 - alpha),
                "auprc_mean": float(np.mean(auprcs)),
                "auprc_std": float(np.std(auprcs, ddof=1)),
                "auroc_mean": float(np.mean(aurocs)),
                "auroc_std": float(np.std(aurocs, ddof=1)),
                "gain_vs_b0_mean": float(np.mean(gains_vs_b0)),
                "gain_vs_b0_min": float(np.min(gains_vs_b0)),
                "gain_vs_b2_mean": float(np.mean(gains_vs_b2)),
                "gain_vs_b2_min": float(np.min(gains_vs_b2)),
                "all_seeds_better_than_b0": bool(
                    np.all(np.asarray(gains_vs_b0) > 0)
                ),
                "all_seeds_better_than_b2": bool(
                    np.all(np.asarray(gains_vs_b2) > 0)
                ),
            })

    results = pd.DataFrame(rows)

    # Best according to mean AUPRC.
    results = results.sort_values(
        ["auprc_mean", "auprc_std"],
        ascending=[False, True],
    ).reset_index(drop=True)

    out_dir = (
        ROOT
        / "Results"
        / "late_fusion"
    )
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_path = (
        out_dir
        / "b0_b2_late_fusion_grid.csv"
    )

    results.to_csv(
        result_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("=" * 110)
    print("TOP 10 FUSION SETTINGS")
    print("=" * 110)

    display_cols = [
        "fusion_type",
        "alpha_b2",
        "alpha_b0",
        "auprc_mean",
        "auprc_std",
        "gain_vs_b0_mean",
        "gain_vs_b0_min",
        "gain_vs_b2_mean",
        "gain_vs_b2_min",
        "all_seeds_better_than_b0",
        "all_seeds_better_than_b2",
    ]

    print(
        results[display_cols]
        .head(10)
        .to_string(index=False)
    )

    best = results.iloc[0]

    print()
    print("=" * 110)
    print("BEST SETTING")
    print("=" * 110)

    for c in display_cols:
        print(f"{c:28s}: {best[c]}")

    print()
    print("Saved:")
    print(" ", result_path)

    print()
    print("=" * 110)
    print("DECISION RULE")
    print("=" * 110)
    print(
        "Late fusion is worth keeping only if it improves B2 on all three seeds "
        "or shows a clearly stable mean gain without sacrificing a seed."
    )
    print(
        "If no stable improvement exists, keep B2 alone and do not add another fusion module."
    )


if __name__ == "__main__":
    main()
