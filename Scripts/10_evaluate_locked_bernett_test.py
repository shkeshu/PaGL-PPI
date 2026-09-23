from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
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
from torch.utils.data import DataLoader


# =============================================================================
# Locked protocol
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

TEST_CSV = ROOT / "Data" / "processed" / "canonical" / "test_pairs.csv"
GLOBAL_EMB_DIR = ROOT / "Data" / "embeddings" / "ESM2_t33"
SEGMENT_EMB_DIR = ROOT / "Data" / "embeddings" / "ESM2_t33_segments"

SCRIPT_B0 = ROOT / "Scripts" / "04_train_strong_symmetric_ablation.py"
SCRIPT_B2 = ROOT / "Scripts" / "06_train_segment_cross_ablation.py"

OUT_DIR = ROOT / "Results" / "final_locked_test"

SEEDS = [42, 123, 456, 789, 1234]

# Frozen before seeing Test.
ALPHA_B2 = 0.70
ALPHA_B0 = 0.30

# Frozen classification threshold.
THRESHOLD = 0.50


# =============================================================================
# Utilities
# =============================================================================

def load_module(name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def safe_logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def sigmoid(x):
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-x))


def fixed_logit_fusion(p_b0, p_b2):
    """
    Locked on Validation before Test:

        z_final = 0.70*z_B2 + 0.30*z_B0
        p_final = sigmoid(z_final)
    """
    z0 = safe_logit(p_b0)
    z2 = safe_logit(p_b2)
    return sigmoid(ALPHA_B2 * z2 + ALPHA_B0 * z0)


def calc_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    specificity = tn / max(tn + fp, 1)

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def checkpoint_paths(seed: int):
    if seed == 42:
        b0 = (
            ROOT
            / "Results"
            / "strong_baseline"
            / "sum_diff_product"
            / "best.pt"
        )
        b2 = (
            ROOT
            / "Results"
            / "segment_cross_ablation"
            / "B2_segment_cross"
            / "best.pt"
        )

    elif seed in (123, 456):
        base = (
            ROOT
            / "Results"
            / "multiseed_confirmation"
            / f"seed_{seed}"
        )

        b0 = (
            base
            / "baseline"
            / "sum_diff_product"
            / "best.pt"
        )
        b2 = (
            base
            / "segment"
            / "B2_segment_cross"
            / "best.pt"
        )

    elif seed in (789, 1234):
        base = (
            ROOT
            / "Results"
            / "final_seed_confirmation"
            / f"seed_{seed}"
        )

        b0 = (
            base
            / "baseline"
            / "sum_diff_product"
            / "best.pt"
        )
        b2 = (
            base
            / "segment"
            / "B2_segment_cross"
            / "best.pt"
        )

    else:
        raise ValueError(f"Unsupported seed: {seed}")

    return b0, b2


# =============================================================================
# Model loading
# =============================================================================

def build_b0_model(mod_b0, ckpt_path: Path, device):
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    ckpt = torch.load(
        ckpt_path,
        map_location=device,
        weights_only=False,
    )

    cfg = ckpt["config"]
    mc = cfg["model"]

    model = mod_b0.StrongSymmetricPPI(
        input_dim=mc["input_dim"],
        projection_dim=mc["projection_dim"],
        hidden_dim=mc["hidden_dim"],
        dropout=mc["dropout"],
        pair_mode=ckpt.get("pair_mode", "sum_diff_product"),
    ).to(device)

    model.load_state_dict(
        ckpt["model_state_dict"]
    )
    model.eval()

    return model, ckpt


def build_b2_model(mod_b2, ckpt_path: Path, device):
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    ckpt = torch.load(
        ckpt_path,
        map_location=device,
        weights_only=False,
    )

    cfg = ckpt["config"]
    mc = cfg["model"]

    mode = ckpt.get(
        "mode",
        "B2_segment_cross",
    )

    if mode != "B2_segment_cross":
        raise RuntimeError(
            f"Expected B2_segment_cross checkpoint, got {mode}"
        )

    model = mod_b2.SegmentPPIModel(
        mode=mode,
        input_dim=1280,
        segment_dim=mc["segment_dim"],
        heads=mc["heads"],
        classifier_hidden=mc["classifier_hidden"],
        dropout=mc["dropout"],
    ).to(device)

    model.load_state_dict(
        ckpt["state_dict"]
    )
    model.eval()

    return model, ckpt


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
    )

    ap.add_argument(
        "--batch-size-b0",
        type=int,
        default=512,
    )

    ap.add_argument(
        "--batch-size-b2",
        type=int,
        default=64,
    )

    args = ap.parse_args()

    device = torch.device(
        "cuda"
        if args.device == "cuda" and torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 120)
    print("LOCKED BER.NETT TEST EVALUATION")
    print("=" * 120)
    print("Test split        : Intra2")
    print("Seeds             :", SEEDS)
    print("Final fusion      : fixed logit fusion")
    print(f"Fusion weights    : B2={ALPHA_B2:.2f}, B0={ALPHA_B0:.2f}")
    print(f"Threshold         : {THRESHOLD:.2f}")
    print("Architecture      : FROZEN")
    print("Hyperparameters   : FROZEN")
    print("Fusion alpha      : FROZEN")
    print("Threshold         : FROZEN")
    print()
    print("WARNING:")
    print("  After this script is run, do NOT change the model based on Test results.")
    print("  Any further architecture tuning would invalidate this Test as a final holdout.")
    print("=" * 120)

    # -------------------------------------------------------------------------
    # Preconditions
    # -------------------------------------------------------------------------
    for p in [
        TEST_CSV,
        GLOBAL_EMB_DIR,
        SEGMENT_EMB_DIR,
        SCRIPT_B0,
        SCRIPT_B2,
    ]:
        if not p.exists():
            raise FileNotFoundError(p)

    for seed in SEEDS:
        b0_ckpt, b2_ckpt = checkpoint_paths(seed)

        if not b0_ckpt.exists():
            raise FileNotFoundError(
                f"Missing B0 checkpoint for seed {seed}:\n{b0_ckpt}"
            )

        if not b2_ckpt.exists():
            raise FileNotFoundError(
                f"Missing B2 checkpoint for seed {seed}:\n{b2_ckpt}"
            )

    # -------------------------------------------------------------------------
    # Dynamically load model definitions
    # -------------------------------------------------------------------------
    mod_b0 = load_module(
        "strong_baseline_module",
        SCRIPT_B0,
    )

    mod_b2 = load_module(
        "segment_cross_module",
        SCRIPT_B2,
    )

    # -------------------------------------------------------------------------
    # Test pair inventory
    # -------------------------------------------------------------------------
    test_df = pd.read_csv(
        TEST_CSV,
        usecols=[
            "protein_a",
            "protein_b",
            "label",
        ],
    )

    expected_rows = 52_048

    if len(test_df) != expected_rows:
        raise RuntimeError(
            f"Test rows={len(test_df):,}; expected {expected_rows:,}"
        )

    pos = int(
        (test_df["label"] == 1).sum()
    )

    neg = int(
        (test_df["label"] == 0).sum()
    )

    if pos != 26_024 or neg != 26_024:
        raise RuntimeError(
            f"Unexpected Test class counts: "
            f"pos={pos:,}, neg={neg:,}"
        )

    test_pids = sorted(
        set(
            test_df["protein_a"].astype(str)
        )
        | set(
            test_df["protein_b"].astype(str)
        )
    )

    if len(test_pids) != 3_022:
        raise RuntimeError(
            f"Test proteins={len(test_pids):,}; expected 3,022"
        )

    print()
    print("Test audit:")
    print(f"  pairs     : {len(test_df):,}")
    print(f"  positive  : {pos:,}")
    print(f"  negative  : {neg:,}")
    print(f"  proteins  : {len(test_pids):,}")
    print()

    # -------------------------------------------------------------------------
    # Preload B0 test embeddings once
    # -------------------------------------------------------------------------
    print("[1/4] Loading B0 global embeddings...")
    global_matrix, pid_to_idx_b0 = mod_b0.preload_embeddings(
        protein_ids=test_pids,
        emb_dir=GLOBAL_EMB_DIR,
        expected_dim=1280,
    )

    b0_test_ds = mod_b0.IndexedPairDataset(
        TEST_CSV,
        pid_to_idx_b0,
    )

    b0_test_loader = DataLoader(
        b0_test_ds,
        batch_size=args.batch_size_b0,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # -------------------------------------------------------------------------
    # Preload B2 segment embeddings once
    # -------------------------------------------------------------------------
    print()
    print("[2/4] Loading B2 segment embeddings...")
    segment_cache, global_segment_matrix, pid_to_idx_b2 = mod_b2.preload_features(
        test_pids,
        SEGMENT_EMB_DIR,
    )

    b2_test_ds = mod_b2.PairIndexDataset(
        TEST_CSV,
        pid_to_idx_b2,
    )

    collator = mod_b2.SegmentCollator(
        segment_cache,
        global_segment_matrix,
    )

    b2_test_loader = DataLoader(
        b2_test_ds,
        batch_size=args.batch_size_b2,
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
        pin_memory=(device.type == "cuda"),
    )

    # -------------------------------------------------------------------------
    # Per-seed locked inference
    # -------------------------------------------------------------------------
    print()
    print("[3/4] Running locked Test inference...")

    rows = []
    prediction_frames = []
    checkpoint_manifest = []

    reference_y = None

    for seed in SEEDS:
        print()
        print("-" * 120)
        print(f"SEED {seed}")
        print("-" * 120)

        b0_ckpt_path, b2_ckpt_path = checkpoint_paths(seed)

        # B0
        b0_model, b0_ckpt = build_b0_model(
            mod_b0,
            b0_ckpt_path,
            device,
        )

        y0, p0 = mod_b0.predict(
            b0_model,
            b0_test_loader,
            global_matrix,
            device,
        )

        del b0_model

        if device.type == "cuda":
            torch.cuda.empty_cache()

        # B2
        b2_model, b2_ckpt = build_b2_model(
            mod_b2,
            b2_ckpt_path,
            device,
        )

        y2, p2, _ = mod_b2.evaluate(
            b2_model,
            b2_test_loader,
            device,
            collect_gate=False,
        )

        del b2_model

        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Integrity
        if not np.array_equal(y0, y2):
            raise RuntimeError(
                f"seed={seed}: B0/B2 Test label order differs"
            )

        if reference_y is None:
            reference_y = y0.copy()
        elif not np.array_equal(reference_y, y0):
            raise RuntimeError(
                f"seed={seed}: Test order differs across seeds"
            )

        # Fixed fusion — NO SEARCH.
        pf = fixed_logit_fusion(
            p_b0=p0,
            p_b2=p2,
        )

        m0 = calc_metrics(
            y0,
            p0,
            threshold=THRESHOLD,
        )
        m2 = calc_metrics(
            y0,
            p2,
            threshold=THRESHOLD,
        )
        mf = calc_metrics(
            y0,
            pf,
            threshold=THRESHOLD,
        )

        print(
            f"B0     AUPRC={m0['auprc']:.6f} "
            f"AUROC={m0['auroc']:.6f} "
            f"F1={m0['f1']:.6f} "
            f"MCC={m0['mcc']:.6f}"
        )

        print(
            f"B2     AUPRC={m2['auprc']:.6f} "
            f"AUROC={m2['auroc']:.6f} "
            f"F1={m2['f1']:.6f} "
            f"MCC={m2['mcc']:.6f}"
        )

        print(
            f"Fusion AUPRC={mf['auprc']:.6f} "
            f"AUROC={mf['auroc']:.6f} "
            f"F1={mf['f1']:.6f} "
            f"MCC={mf['mcc']:.6f}"
        )

        print(
            f"ΔAUPRC Fusion-B2={mf['auprc'] - m2['auprc']:+.6f}"
        )

        print(
            f"ΔAUPRC Fusion-B0={mf['auprc'] - m0['auprc']:+.6f}"
        )

        for model_name, m in [
            ("B0_strong", m0),
            ("B2_cross", m2),
            ("FixedFusion_0.70B2_0.30B0", mf),
        ]:
            rows.append({
                "seed": seed,
                "model": model_name,
                **m,
            })

        prediction_frames.append(
            pd.DataFrame({
                "seed": seed,
                "row_index": np.arange(len(y0)),
                "label": y0.astype(int),
                "prob_b0": p0,
                "prob_b2": p2,
                "prob_fixed_fusion": pf,
            })
        )

        checkpoint_manifest.append({
            "seed": seed,
            "b0_checkpoint": str(b0_ckpt_path),
            "b2_checkpoint": str(b2_ckpt_path),
            "b0_best_epoch": int(b0_ckpt["best_epoch"]),
            "b2_best_epoch": int(b2_ckpt["best_epoch"]),
            "b0_validation_auprc": float(b0_ckpt["best_val_auprc"]),
            "b2_validation_auprc": float(b2_ckpt["best_val_auprc"]),
        })

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print()
    print("[4/4] Writing final locked Test report...")

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw = pd.DataFrame(rows)

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

    summary_rows = []

    for model_name, g in raw.groupby("model"):
        row = {
            "model": model_name,
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

        summary_rows.append(row)

    summary = pd.DataFrame(
        summary_rows
    ).sort_values(
        "auprc_mean",
        ascending=False,
    )

    pivot = raw.pivot(
        index="seed",
        columns="model",
        values="auprc",
    )

    delta_rows = []

    comparisons = {
        "Fusion - B2":
            pivot["FixedFusion_0.70B2_0.30B0"]
            - pivot["B2_cross"],

        "Fusion - B0":
            pivot["FixedFusion_0.70B2_0.30B0"]
            - pivot["B0_strong"],

        "B2 - B0":
            pivot["B2_cross"]
            - pivot["B0_strong"],
    }

    for name, d in comparisons.items():
        delta_rows.append({
            "comparison": name,
            "n_seeds": int(len(d)),
            "delta_auprc_mean": float(d.mean()),
            "delta_auprc_std": float(d.std(ddof=1)),
            "delta_min": float(d.min()),
            "delta_max": float(d.max()),
            "positive_seed_count": int((d > 0).sum()),
            "all_seeds_positive": bool((d > 0).all()),
        })

    deltas = pd.DataFrame(
        delta_rows
    )

    predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    raw_path = (
        OUT_DIR
        / "test_five_seed_raw_metrics.csv"
    )

    summary_path = (
        OUT_DIR
        / "test_five_seed_model_summary.csv"
    )

    delta_path = (
        OUT_DIR
        / "test_five_seed_auprc_deltas.csv"
    )

    pred_path = (
        OUT_DIR
        / "test_five_seed_predictions.csv"
    )

    manifest_path = (
        OUT_DIR
        / "locked_test_protocol.json"
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

    protocol = {
        "status": "LOCKED_FINAL_TEST",
        "dataset": "Bernett Gold Standard",
        "test_split": "Intra2",
        "test_pairs": 52048,
        "test_positive": 26024,
        "test_negative": 26024,
        "test_proteins": 3022,
        "seeds": SEEDS,
        "final_model": "FixedFusion_0.70B2_0.30B0",
        "fusion": {
            "space": "logit",
            "alpha_b2": ALPHA_B2,
            "alpha_b0": ALPHA_B0,
        },
        "classification_threshold": THRESHOLD,
        "selection_source": "Validation Intra0 only",
        "test_used_for_model_selection": False,
        "checkpoints": checkpoint_manifest,
        "warning": (
            "Do not tune model architecture, fusion weight, threshold, "
            "or select seeds after observing these Test results."
        ),
    }

    manifest_path.write_text(
        json.dumps(
            protocol,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 120)
    print("FINAL LOCKED TEST SUMMARY")
    print("=" * 120)

    display_cols = [
        "model",
        "n_seeds",
        "auprc_mean",
        "auprc_std",
        "auroc_mean",
        "auroc_std",
        "accuracy_mean",
        "accuracy_std",
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
    print("TEST AUPRC DELTAS — DESCRIPTIVE ONLY")
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
    print(" ", pred_path)
    print(" ", manifest_path)

    print()
    print("=" * 120)
    print("IMPORTANT")
    print("=" * 120)
    print(
        "The Bernett Test holdout has now been consumed."
    )
    print(
        "Do not modify the model, fusion alpha, threshold, "
        "or choose a preferred seed based on these Test results."
    )
    print(
        "Report the pre-registered five-seed mean ± std."
    )


if __name__ == "__main__":
    main()
