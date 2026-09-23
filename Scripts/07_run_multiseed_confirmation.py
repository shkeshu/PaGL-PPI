from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

BASELINE_SCRIPT = ROOT / "Scripts" / "04_train_strong_symmetric_ablation.py"
SEGMENT_SCRIPT = ROOT / "Scripts" / "06_train_segment_cross_ablation.py"

BASELINE_CFG = ROOT / "Configs" / "strong_baseline.json"
SEGMENT_CFG = ROOT / "Configs" / "segment_cross_ablation.json"

OUT_ROOT = ROOT / "Results" / "multiseed_confirmation"

SEEDS = [42, 123, 456]

SEGMENT_MODES = [
    "B1_segment_only",
    "B2_segment_cross",
    "B3_cross_global_concat",
]


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


def parse_baseline_result(path: Path, seed: int):
    obj = read_json(path)

    m = obj["validation_at_threshold_0.5"]

    return {
        "seed": seed,
        "model": "B0_strong_sum_diff_product",
        "best_epoch": int(obj["best_epoch"]),
        "parameters": int(obj["trainable_parameters"]),
        "accuracy": float(m["accuracy"]),
        "precision": float(m["precision"]),
        "recall": float(m["recall"]),
        "specificity": float(m["specificity"]),
        "f1": float(m["f1"]),
        "mcc": float(m["mcc"]),
        "auroc": float(m["auroc"]),
        "auprc": float(m["auprc"]),
    }


def parse_segment_result(path: Path, seed: int):
    obj = read_json(path)
    m = obj["metrics"]

    row = {
        "seed": seed,
        "model": obj["mode"],
        "best_epoch": int(obj["best_epoch"]),
        "parameters": int(obj["parameters"]),
        "accuracy": float(m["accuracy"]),
        "precision": float(m["precision"]),
        "recall": float(m["recall"]),
        "specificity": float(m["specificity"]),
        "f1": float(m["f1"]),
        "mcc": float(m["mcc"]),
        "auroc": float(m["auroc"]),
        "auprc": float(m["auprc"]),
    }

    if obj.get("gate_stats"):
        row["gate_mean"] = float(obj["gate_stats"]["mean"])
        row["gate_std"] = float(obj["gate_stats"]["std"])

    return row


def existing_seed42_results():
    """
    Reuse already completed seed=42 experiments when available.
    """
    paths = {
        "B0": (
            ROOT
            / "Results"
            / "strong_baseline"
            / "sum_diff_product"
            / "best_val_metrics.json"
        ),
        "B1": (
            ROOT
            / "Results"
            / "segment_cross_ablation"
            / "B1_segment_only"
            / "best_val_metrics.json"
        ),
        "B2": (
            ROOT
            / "Results"
            / "segment_cross_ablation"
            / "B2_segment_cross"
            / "best_val_metrics.json"
        ),
        "B3": (
            ROOT
            / "Results"
            / "segment_cross_ablation"
            / "B3_cross_global_concat"
            / "best_val_metrics.json"
        ),
    }

    return paths


def prepare_and_run_seed(seed: int):
    seed_dir = OUT_ROOT / f"seed_{seed}"
    cfg_dir = seed_dir / "configs"

    baseline_out = seed_dir / "baseline"
    segment_out = seed_dir / "segment"

    baseline_cfg = read_json(BASELINE_CFG)
    segment_cfg = read_json(SEGMENT_CFG)

    # ----------------------------------------------------------------------
    # Strong global baseline: only sum_diff_product
    # ----------------------------------------------------------------------
    baseline_cfg["seed"] = seed
    baseline_cfg["pair_modes"] = ["sum_diff_product"]
    baseline_cfg["output_dir"] = str(baseline_out)

    baseline_cfg_path = cfg_dir / "strong_baseline.json"
    write_json(baseline_cfg_path, baseline_cfg)

    baseline_result = (
        baseline_out
        / "sum_diff_product"
        / "best_val_metrics.json"
    )

    if not baseline_result.exists():
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
        print(f"[SKIP] Existing: {baseline_result}")

    # ----------------------------------------------------------------------
    # B1/B2/B3 segment models
    # ----------------------------------------------------------------------
    segment_cfg["seed"] = seed
    segment_cfg["modes"] = SEGMENT_MODES
    segment_cfg["output_dir"] = str(segment_out)

    segment_cfg_path = cfg_dir / "segment_cross_ablation.json"
    write_json(segment_cfg_path, segment_cfg)

    for mode in SEGMENT_MODES:
        result_path = segment_out / mode / "best_val_metrics.json"

        if result_path.exists():
            print(f"[SKIP] Existing: {result_path}")
            continue

        run_command(
            [
                sys.executable,
                str(SEGMENT_SCRIPT),
                "--config",
                str(segment_cfg_path),
                "--only",
                mode,
            ]
        )


def collect_all_results(seeds, reuse_seed42: bool):
    rows = []

    existing42 = existing_seed42_results()

    for seed in seeds:
        if seed == 42 and reuse_seed42:
            required = list(existing42.values())

            if all(p.exists() for p in required):
                print("[REUSE] Using existing seed=42 results.")

                rows.append(
                    parse_baseline_result(
                        existing42["B0"],
                        seed,
                    )
                )
                rows.append(
                    parse_segment_result(
                        existing42["B1"],
                        seed,
                    )
                )
                rows.append(
                    parse_segment_result(
                        existing42["B2"],
                        seed,
                    )
                )
                rows.append(
                    parse_segment_result(
                        existing42["B3"],
                        seed,
                    )
                )
                continue

            print(
                "[WARN] Seed=42 reuse requested, but one or more files "
                "are missing. It will be rerun."
            )

        seed_dir = OUT_ROOT / f"seed_{seed}"

        rows.append(
            parse_baseline_result(
                seed_dir
                / "baseline"
                / "sum_diff_product"
                / "best_val_metrics.json",
                seed,
            )
        )

        for mode in SEGMENT_MODES:
            rows.append(
                parse_segment_result(
                    seed_dir
                    / "segment"
                    / mode
                    / "best_val_metrics.json",
                    seed,
                )
            )

    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame):
    metrics = [
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "mcc",
        "auroc",
        "auprc",
    ]

    summary_rows = []

    for model, g in df.groupby("model"):
        row = {
            "model": model,
            "n_seeds": len(g),
        }

        for metric in metrics:
            row[f"{metric}_mean"] = g[metric].mean()
            row[f"{metric}_std"] = g[metric].std(ddof=1)

        row["best_epoch_mean"] = g["best_epoch"].mean()
        row["best_epoch_min"] = g["best_epoch"].min()
        row["best_epoch_max"] = g["best_epoch"].max()

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary = summary.sort_values(
        "auprc_mean",
        ascending=False,
    )

    return summary


def add_paired_deltas(df: pd.DataFrame):
    pivot = df.pivot(
        index="seed",
        columns="model",
        values="auprc",
    )

    baseline = "B0_strong_sum_diff_product"

    delta_rows = []

    for model in [
        "B1_segment_only",
        "B2_segment_cross",
        "B3_cross_global_concat",
    ]:
        values = pivot[model] - pivot[baseline]

        delta_rows.append(
            {
                "comparison": f"{model} - {baseline}",
                "delta_auprc_mean": values.mean(),
                "delta_auprc_std": values.std(ddof=1),
                "min_delta": values.min(),
                "max_delta": values.max(),
                "all_seeds_positive": bool((values > 0).all()),
                "all_seeds_ge_0.01": bool((values >= 0.01).all()),
            }
        )

    # Incremental module deltas.
    for model_a, model_b in [
        ("B2_segment_cross", "B1_segment_only"),
        ("B3_cross_global_concat", "B2_segment_cross"),
    ]:
        values = pivot[model_a] - pivot[model_b]

        delta_rows.append(
            {
                "comparison": f"{model_a} - {model_b}",
                "delta_auprc_mean": values.mean(),
                "delta_auprc_std": values.std(ddof=1),
                "min_delta": values.min(),
                "max_delta": values.max(),
                "all_seeds_positive": bool((values > 0).all()),
                "all_seeds_ge_0.01": bool((values >= 0.01).all()),
            }
        )

    return pd.DataFrame(delta_rows)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=SEEDS,
    )
    ap.add_argument(
        "--rerun-seed42",
        action="store_true",
        help="Do not reuse existing seed=42 results.",
    )
    ap.add_argument(
        "--collect-only",
        action="store_true",
        help="Only aggregate existing results; run no training.",
    )

    args = ap.parse_args()

    seeds = args.seeds

    print("=" * 120)
    print("Bernett multi-seed confirmation")
    print("=" * 120)
    print("Seeds:", seeds)
    print("Models:")
    print("  B0 strong global baseline")
    print("  B1 segment only")
    print("  B2 segment + cross-attention")
    print("  B3 cross-attention + global concat")
    print()
    print("Bernett Test is NOT accessed.")
    print("=" * 120)

    reuse_seed42 = not args.rerun_seed42

    if not args.collect_only:
        for seed in seeds:
            if seed == 42 and reuse_seed42:
                existing42 = existing_seed42_results()

                if all(p.exists() for p in existing42.values()):
                    print(
                        "\n[REUSE] Seed=42 already completed. "
                        "No retraining required."
                    )
                    continue

            prepare_and_run_seed(seed)

    df = collect_all_results(
        seeds=seeds,
        reuse_seed42=reuse_seed42
    )

    OUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_path = OUT_ROOT / "multiseed_raw_results.csv"
    df.to_csv(
        raw_path,
        index=False,
        encoding="utf-8-sig",
    )

    summary = summarize(df)
    summary_path = OUT_ROOT / "multiseed_model_summary.csv"
    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    deltas = add_paired_deltas(df)
    delta_path = OUT_ROOT / "multiseed_auprc_deltas.csv"
    deltas.to_csv(
        delta_path,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 120)
    print("MULTI-SEED MODEL SUMMARY")
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
        summary[display_cols].to_string(
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

    print()
    print("Saved:")
    print(" ", raw_path)
    print(" ", summary_path)
    print(" ", delta_path)

    print()
    print("=" * 120)
    print("DECISION RULE")
    print("=" * 120)
    print(
        "Keep B3 as the main model only if it remains consistently "
        "better than B0 across seeds, preferably mean ΔAUPRC >= 0.010."
    )
    print(
        "Treat B2->B3 as a small global-residual/concat contribution "
        "unless its incremental gain is stable across seeds."
    )
    print(
        "Do not restore B4 gate unless a redesigned gate avoids collapse."
    )


if __name__ == "__main__":
    main()
