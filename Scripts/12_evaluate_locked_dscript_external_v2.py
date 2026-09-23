from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Dict, List, Tuple

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

from torch.utils.data import (
    Dataset,
    DataLoader,
)


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

SCRIPT_B0 = (
    ROOT
    / "Scripts"
    / "04_train_strong_symmetric_ablation.py"
)

SCRIPT_B2 = (
    ROOT
    / "Scripts"
    / "06_train_segment_cross_ablation.py"
)

CANONICAL_ROOT = (
    ROOT
    / "Data"
    / "external_canonical"
)

EMB_ROOT = (
    ROOT
    / "Data"
    / "embeddings_external"
)

OUT_ROOT = (
    ROOT
    / "Results"
    / "external_generalization"
)

SEEDS = [
    42,
    123,
    456,
    789,
    1234,
]

ALPHA_B2 = 0.70
ALPHA_B0 = 0.30
THRESHOLD = 0.50

MODEL_ORDER = [
    "B0_strong",
    "B2_cross",
    "FixedFusion_0.70B2_0.30B0",
]


def load_module(
    name: str,
    path: Path,
):
    spec = importlib.util.spec_from_file_location(
        name,
        str(path),
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise ImportError(
            path
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


def safe_logit(
    p,
    eps=1e-6,
):
    p = np.clip(
        np.asarray(
            p,
            dtype=float,
        ),
        eps,
        1.0 - eps,
    )

    return np.log(
        p
        / (
            1.0 - p
        )
    )


def sigmoid(x):
    return 1.0 / (
        1.0
        + np.exp(
            -np.asarray(
                x,
                dtype=float,
            )
        )
    )


def fixed_fusion(
    p0,
    p2,
):
    return sigmoid(
        ALPHA_B2
        * safe_logit(p2)
        + ALPHA_B0
        * safe_logit(p0)
    )


def calc_metrics(
    y_true,
    y_prob,
    threshold=THRESHOLD,
):
    y_true = np.asarray(
        y_true
    ).astype(int)

    y_prob = np.asarray(
        y_prob
    ).astype(float)

    pred = (
        y_prob
        >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        pred,
        labels=[
            0,
            1,
        ],
    ).ravel()

    specificity = (
        tn
        / max(
            tn + fp,
            1,
        )
    )

    return {
        "threshold": float(
            threshold
        ),
        "prevalence": float(
            y_true.mean()
        ),
        "random_auprc_baseline": float(
            y_true.mean()
        ),
        "accuracy": float(
            accuracy_score(
                y_true,
                pred,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                pred,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                pred,
                zero_division=0,
            )
        ),
        "specificity": float(
            specificity
        ),
        "f1": float(
            f1_score(
                y_true,
                pred,
                zero_division=0,
            )
        ),
        "mcc": float(
            matthews_corrcoef(
                y_true,
                pred,
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
        "auprc_minus_random": float(
            average_precision_score(
                y_true,
                y_prob,
            )
            - y_true.mean()
        ),
        "auprc_over_random": float(
            average_precision_score(
                y_true,
                y_prob,
            )
            / max(
                float(y_true.mean()),
                1e-12,
            )
        ),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def checkpoint_paths(
    seed: int,
):
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

    elif seed in (
        123,
        456,
    ):
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

    elif seed in (
        789,
        1234,
    ):
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
        raise ValueError(
            seed
        )

    return b0, b2


def build_b0(
    mod,
    path,
    device,
):
    ckpt = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    mc = ckpt[
        "config"
    ][
        "model"
    ]

    model = mod.StrongSymmetricPPI(
        input_dim=mc["input_dim"],
        projection_dim=mc["projection_dim"],
        hidden_dim=mc["hidden_dim"],
        dropout=mc["dropout"],
        pair_mode=ckpt.get(
            "pair_mode",
            "sum_diff_product",
        ),
    ).to(
        device
    )

    model.load_state_dict(
        ckpt[
            "model_state_dict"
        ]
    )

    model.eval()

    return model


def build_b2(
    mod,
    path,
    device,
):
    ckpt = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    mc = ckpt[
        "config"
    ][
        "model"
    ]

    mode = ckpt.get(
        "mode",
        "B2_segment_cross",
    )

    if mode != "B2_segment_cross":
        raise RuntimeError(
            f"Unexpected B2 mode: {mode}"
        )

    model = mod.SegmentPPIModel(
        mode=mode,
        input_dim=1280,
        segment_dim=mc["segment_dim"],
        heads=mc["heads"],
        classifier_hidden=mc["classifier_hidden"],
        dropout=mc["dropout"],
    ).to(
        device
    )

    model.load_state_dict(
        ckpt[
            "state_dict"
        ]
    )

    model.eval()

    return model


class ExternalPairDataset(Dataset):
    def __init__(
        self,
        pair_df: pd.DataFrame,
        pid_to_idx: Dict[str, int],
    ):
        self.a = torch.tensor(
            [
                pid_to_idx[
                    str(x)
                ]
                for x in pair_df[
                    "protein_a"
                ]
            ],
            dtype=torch.long,
        )

        self.b = torch.tensor(
            [
                pid_to_idx[
                    str(x)
                ]
                for x in pair_df[
                    "protein_b"
                ]
            ],
            dtype=torch.long,
        )

        self.y = torch.tensor(
            pair_df[
                "label"
            ].astype(
                float
            ).values,
            dtype=torch.float32,
        )

        self.row_index = torch.tensor(
            pair_df[
                "row_index"
            ].astype(
                int
            ).values,
            dtype=torch.long,
        )

    def __len__(self):
        return len(
            self.y
        )

    def __getitem__(
        self,
        idx,
    ):
        return (
            self.a[idx],
            self.b[idx],
            self.y[idx],
            self.row_index[idx],
        )


class ExternalCollator:
    def __init__(
        self,
        segments,
        global_matrix,
    ):
        self.segments = segments
        self.global_matrix = global_matrix

    def __call__(
        self,
        batch,
    ):
        a_idx = [
            int(
                x[0]
            )
            for x in batch
        ]

        b_idx = [
            int(
                x[1]
            )
            for x in batch
        ]

        y = torch.stack(
            [
                x[2]
                for x in batch
            ]
        )

        rows = torch.stack(
            [
                x[3]
                for x in batch
            ]
        )

        a_seg = [
            self.segments[
                i
            ]
            for i in a_idx
        ]

        b_seg = [
            self.segments[
                i
            ]
            for i in b_idx
        ]

        max_len = max(
            max(
                x.shape[0]
                for x in a_seg
            ),
            max(
                x.shape[0]
                for x in b_seg
            ),
        )

        B = len(
            batch
        )

        A = torch.zeros(
            (
                B,
                max_len,
                1280,
            ),
            dtype=torch.float16,
        )

        Bs = torch.zeros(
            (
                B,
                max_len,
                1280,
            ),
            dtype=torch.float16,
        )

        Am = torch.zeros(
            (
                B,
                max_len,
            ),
            dtype=torch.bool,
        )

        Bm = torch.zeros(
            (
                B,
                max_len,
            ),
            dtype=torch.bool,
        )

        for i, x in enumerate(
            a_seg
        ):
            n = x.shape[0]
            A[
                i,
                :n,
            ] = x
            Am[
                i,
                :n,
            ] = True

        for i, x in enumerate(
            b_seg
        ):
            n = x.shape[0]
            Bs[
                i,
                :n,
            ] = x
            Bm[
                i,
                :n,
            ] = True

        idx_a = torch.tensor(
            a_idx,
            dtype=torch.long,
        )

        idx_b = torch.tensor(
            b_idx,
            dtype=torch.long,
        )

        Ag = self.global_matrix[
            idx_a
        ]

        Bg = self.global_matrix[
            idx_b
        ]

        return (
            A,
            Am,
            Bs,
            Bm,
            Ag,
            Bg,
            y,
            rows,
        )


def load_external_features(
    species: str,
):
    emb_dir = (
        EMB_ROOT
        / f"D-SCRIPT_{species}"
    )

    manifest_path = (
        emb_dir
        / "embedding_manifest.csv"
    )

    if not manifest_path.exists():
        raise FileNotFoundError(
            manifest_path
        )

    manifest = pd.read_csv(
        manifest_path
    )

    pid_to_file = dict(
        zip(
            manifest[
                "protein_id"
            ].astype(str),
            manifest[
                "file_key"
            ].astype(str),
        )
    )

    pids = sorted(
        pid_to_file
    )

    pid_to_idx = {
        pid: i
        for i, pid in enumerate(
            pids
        )
    }

    global_matrix = torch.empty(
        (
            len(pids),
            1280,
        ),
        dtype=torch.float32,
    )

    segment_cache = []

    print(
        f"Loading {species} external features "
        f"for {len(pids):,} proteins..."
    )

    for i, pid in enumerate(
        pids
    ):
        key = pid_to_file[
            pid
        ]

        gp = (
            emb_dir
            / "global"
            / f"{key}.pt"
        )

        sp = (
            emb_dir
            / "segments"
            / f"{key}.pt"
        )

        gx = torch.load(
            gp,
            map_location="cpu",
            weights_only=True,
        )

        sx = torch.load(
            sp,
            map_location="cpu",
            weights_only=True,
        )

        global_matrix[
            i
        ] = torch.as_tensor(
            gx
        ).float()

        segment_cache.append(
            sx[
                "segments"
            ].half().contiguous()
        )

    return (
        global_matrix,
        segment_cache,
        pid_to_idx,
    )


@torch.inference_mode()
def predict_b0(
    model,
    loader,
    global_matrix,
    device,
):
    ys = []
    ps = []
    rows = []

    for (
        a_idx,
        b_idx,
        y,
        row_index,
    ) in loader:
        a = global_matrix[
            a_idx
        ].to(
            device,
            non_blocking=True,
        )

        b = global_matrix[
            b_idx
        ].to(
            device,
            non_blocking=True,
        )

        logits = model(
            a,
            b,
        )

        p = torch.sigmoid(
            logits
        )

        ys.append(
            y.numpy()
        )

        ps.append(
            p.cpu().numpy()
        )

        rows.append(
            row_index.numpy()
        )

    return (
        np.concatenate(
            ys
        ),
        np.concatenate(
            ps
        ),
        np.concatenate(
            rows
        ),
    )


@torch.inference_mode()
def predict_b2(
    model,
    loader,
    device,
):
    ys = []
    ps = []
    rows = []

    for batch in loader:
        (
            A,
            Am,
            B,
            Bm,
            Ag,
            Bg,
            y,
            row_index,
        ) = batch

        A = A.to(
            device,
            non_blocking=True,
        )
        Am = Am.to(
            device,
            non_blocking=True,
        )
        B = B.to(
            device,
            non_blocking=True,
        )
        Bm = Bm.to(
            device,
            non_blocking=True,
        )
        Ag = Ag.to(
            device,
            non_blocking=True,
        )
        Bg = Bg.to(
            device,
            non_blocking=True,
        )

        logits = model(
            A,
            Am,
            B,
            Bm,
            Ag,
            Bg,
        )

        p = torch.sigmoid(
            logits
        )

        ys.append(
            y.numpy()
        )
        ps.append(
            p.cpu().numpy()
        )
        rows.append(
            row_index.numpy()
        )

    return (
        np.concatenate(
            ys
        ),
        np.concatenate(
            ps
        ),
        np.concatenate(
            rows
        ),
    )


def dataset_specs():
    """
    Four formal external evaluation sets.

    Human Full:
        Cross-dataset evaluation; not strict cold-start relative to Bernett.

    Human NovelVsBernettTrain:
        Removes every pair containing a protein whose exact sequence appears
        in Bernett Train. Primary within-species strict external cold-start.

    Human NovelVsBernettAll:
        Removes every pair containing a protein whose exact sequence appears
        anywhere in Bernett Train/Val/Test. Ultra-strict subset.

    Yeast Full:
        Exact-sequence overlap with Bernett Train/Val/Test is zero, so the
        full cleaned Yeast test is already the strict cross-species zero-shot set.
    """
    human = (
        CANONICAL_ROOT
        / "D-SCRIPT_Human"
    )

    yeast = (
        CANONICAL_ROOT
        / "D-SCRIPT_Yeast"
    )

    return [
        (
            "D-SCRIPT_Human_Full",
            "Human",
            human / "test_pairs.csv",
            "external_full",
        ),
        (
            "D-SCRIPT_Human_NovelVsBernettTrain",
            "Human",
            human / "test_pairs_novel_vs_bernett_train.csv",
            "strict_no_exact_sequence_overlap_with_bernett_train",
        ),
        (
            "D-SCRIPT_Human_NovelVsBernettAll",
            "Human",
            human / "test_pairs_novel_vs_bernett_all.csv",
            "strict_no_exact_sequence_overlap_with_any_bernett_split",
        ),
        (
            "D-SCRIPT_Yeast_Full",
            "Yeast",
            yeast / "test_pairs.csv",
            "cross_species_zero_shot_no_exact_sequence_overlap",
        ),
    ]


def evaluate_dataset(
    name,
    species,
    pair_path,
    subset_type,
    mod_b0,
    mod_b2,
    device,
    args,
):
    if not pair_path.exists():
        raise FileNotFoundError(
            pair_path
        )

    df = pd.read_csv(
        pair_path
    )

    if len(
        df
    ) == 0:
        print(
            f"[SKIP] {name}: empty subset"
        )
        return [], []

    if df[
        "label"
    ].nunique() < 2:
        print(
            f"[SKIP] {name}: only one class"
        )
        return [], []

    prevalence = float(
        df["label"].mean()
    )

    print()
    print("=" * 120)
    print(f"External dataset: {name}")
    print(f"Pairs      : {len(df):,}")
    print(f"Positive   : {(df['label'] == 1).sum():,}")
    print(f"Negative   : {(df['label'] == 0).sum():,}")
    print(f"Prevalence : {prevalence:.6f}")
    print(f"Random AUPRC baseline ≈ {prevalence:.6f}")
    print("=" * 120)

    (
        global_matrix,
        segments,
        pid_to_idx,
    ) = load_external_features(
        species
    )

    missing = (
        set(
            df[
                "protein_a"
            ].astype(str)
        )
        | set(
            df[
                "protein_b"
            ].astype(str)
        )
    ) - set(
        pid_to_idx
    )

    if missing:
        raise RuntimeError(
            f"{name}: {len(missing)} proteins "
            f"lack external embeddings"
        )

    ds = ExternalPairDataset(
        df,
        pid_to_idx,
    )

    b0_loader = DataLoader(
        ds,
        batch_size=args.batch_size_b0,
        shuffle=False,
        num_workers=0,
        pin_memory=(
            device.type
            == "cuda"
        ),
    )

    collator = ExternalCollator(
        segments,
        global_matrix,
    )

    b2_loader = DataLoader(
        ds,
        batch_size=args.batch_size_b2,
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
        pin_memory=(
            device.type
            == "cuda"
        ),
    )

    metric_rows = []
    prediction_rows = []

    reference_y = None
    reference_rows = None

    for seed in SEEDS:
        b0_path, b2_path = checkpoint_paths(
            seed
        )

        print(
            f"{name} | seed={seed}"
        )

        b0 = build_b0(
            mod_b0,
            b0_path,
            device,
        )

        y0, p0, r0 = predict_b0(
            b0,
            b0_loader,
            global_matrix,
            device,
        )

        del b0

        if device.type == "cuda":
            torch.cuda.empty_cache()

        b2 = build_b2(
            mod_b2,
            b2_path,
            device,
        )

        y2, p2, r2 = predict_b2(
            b2,
            b2_loader,
            device,
        )

        del b2

        if device.type == "cuda":
            torch.cuda.empty_cache()

        if not np.array_equal(
            y0,
            y2,
        ):
            raise RuntimeError(
                f"{name}: B0/B2 label mismatch"
            )

        if not np.array_equal(
            r0,
            r2,
        ):
            raise RuntimeError(
                f"{name}: B0/B2 row order mismatch"
            )

        if reference_y is None:
            reference_y = y0.copy()
            reference_rows = r0.copy()
        else:
            if not np.array_equal(
                reference_y,
                y0,
            ):
                raise RuntimeError(
                    f"{name}: label order differs across seeds"
                )

        pf = fixed_fusion(
            p0,
            p2,
        )

        for model_name, p in [
            (
                "B0_strong",
                p0,
            ),
            (
                "B2_cross",
                p2,
            ),
            (
                "FixedFusion_0.70B2_0.30B0",
                pf,
            ),
        ]:
            m = calc_metrics(
                y0,
                p,
            )

            metric_rows.append({
                "dataset": name,
                "species": species,
                "subset_type": subset_type,
                "seed": seed,
                "model": model_name,
                "pairs": len(df),
                "positive": int(
                    (df["label"] == 1).sum()
                ),
                "negative": int(
                    (df["label"] == 0).sum()
                ),
                **m,
            })

        prediction_rows.append(
            pd.DataFrame({
                "dataset": name,
                "species": species,
                "subset_type": subset_type,
                "seed": seed,
                "row_index": r0.astype(int),
                "label": y0.astype(int),
                "prob_b0": p0,
                "prob_b2": p2,
                "prob_fixed_fusion": pf,
            })
        )

    return (
        metric_rows,
        prediction_rows,
    )


def summarize(
    raw: pd.DataFrame,
):
    metrics = [
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "mcc",
        "auroc",
        "auprc",
        "auprc_minus_random",
        "auprc_over_random",
    ]

    rows = []

    for (
        dataset,
        species,
        subset_type,
        model,
    ), g in raw.groupby(
        [
            "dataset",
            "species",
            "subset_type",
            "model",
        ]
    ):
        row = {
            "dataset": dataset,
            "species": species,
            "subset_type": subset_type,
            "model": model,
            "n_seeds": len(g),
            "pairs": int(
                g[
                    "pairs"
                ].iloc[0]
            ),
            "positive": int(
                g[
                    "positive"
                ].iloc[0]
            ),
            "negative": int(
                g[
                    "negative"
                ].iloc[0]
            ),
            "prevalence": float(
                g[
                    "prevalence"
                ].iloc[0]
            ),
            "random_auprc_baseline": float(
                g[
                    "random_auprc_baseline"
                ].iloc[0]
            ),
        }

        for metric in metrics:
            row[
                f"{metric}_mean"
            ] = float(
                g[
                    metric
                ].mean()
            )

            row[
                f"{metric}_std"
            ] = float(
                g[
                    metric
                ].std(
                    ddof=1
                )
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    ).sort_values(
        [
            "dataset",
            "auprc_mean",
        ],
        ascending=[
            True,
            False,
        ],
    )


def paired_deltas(
    raw: pd.DataFrame,
):
    rows = []

    for dataset, g in raw.groupby(
        "dataset"
    ):
        pivot = g.pivot(
            index="seed",
            columns="model",
            values="auprc",
        )

        comparisons = {
            "Fusion - B2": (
                pivot[
                    "FixedFusion_0.70B2_0.30B0"
                ]
                - pivot[
                    "B2_cross"
                ]
            ),
            "Fusion - B0": (
                pivot[
                    "FixedFusion_0.70B2_0.30B0"
                ]
                - pivot[
                    "B0_strong"
                ]
            ),
            "B2 - B0": (
                pivot[
                    "B2_cross"
                ]
                - pivot[
                    "B0_strong"
                ]
            ),
        }

        for name, d in comparisons.items():
            rows.append({
                "dataset": dataset,
                "comparison": name,
                "delta_auprc_mean": float(
                    d.mean()
                ),
                "delta_auprc_std": float(
                    d.std(
                        ddof=1
                    )
                ),
                "delta_min": float(
                    d.min()
                ),
                "delta_max": float(
                    d.max()
                ),
                "positive_seed_count": int(
                    (
                        d > 0
                    ).sum()
                ),
                "all_seeds_positive": bool(
                    (
                        d > 0
                    ).all()
                ),
            })

    return pd.DataFrame(
        rows
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--device",
        choices=[
            "cuda",
            "cpu",
        ],
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
        if args.device == "cuda"
        and torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 120)
    print(
        "STEP 16 v2 — LOCKED EXTERNAL GENERALIZATION"
    )
    print("=" * 120)
    print(
        "Training/fine-tuning: NONE"
    )
    print(
        f"Frozen fusion: "
        f"{ALPHA_B2:.2f} B2 + "
        f"{ALPHA_B0:.2f} B0 in logit space"
    )
    print(
        f"Frozen threshold: {THRESHOLD:.2f}"
    )
    print(
        f"Seeds: {SEEDS}"
    )
    print("=" * 120)

    mod_b0 = load_module(
        "locked_b0_module",
        SCRIPT_B0,
    )

    mod_b2 = load_module(
        "locked_b2_module",
        SCRIPT_B2,
    )

    for seed in SEEDS:
        p0, p2 = checkpoint_paths(
            seed
        )

        if not p0.exists():
            raise FileNotFoundError(
                p0
            )

        if not p2.exists():
            raise FileNotFoundError(
                p2
            )

    all_metrics = []
    all_predictions = []

    for (
        name,
        species,
        pair_path,
        subset_type,
    ) in dataset_specs():
        metrics_rows, pred_rows = evaluate_dataset(
            name,
            species,
            pair_path,
            subset_type,
            mod_b0,
            mod_b2,
            device,
            args,
        )

        all_metrics.extend(
            metrics_rows
        )

        all_predictions.extend(
            pred_rows
        )

    if not all_metrics:
        raise RuntimeError(
            "No evaluable external dataset/subset found"
        )

    raw = pd.DataFrame(
        all_metrics
    )

    predictions = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    summary = summarize(
        raw
    )

    deltas = paired_deltas(
        raw
    )

    OUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw.to_csv(
        OUT_ROOT
        / "external_raw_metrics_v2.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        OUT_ROOT
        / "external_model_summary_v2.csv",
        index=False,
        encoding="utf-8-sig",
    )

    deltas.to_csv(
        OUT_ROOT
        / "external_auprc_deltas_v2.csv",
        index=False,
        encoding="utf-8-sig",
    )

    predictions.to_csv(
        OUT_ROOT
        / "external_predictions_v2.csv",
        index=False,
        encoding="utf-8-sig",
    )

    protocol = {
        "status": "LOCKED_EXTERNAL_GENERALIZATION_V2",
        "source_training_dataset": "Bernett Gold Standard Intra1",
        "model_selection_dataset": "Bernett Gold Standard Intra0",
        "bernett_test_already_consumed": True,
        "external_datasets": [
            "D-SCRIPT Human Full",
            "D-SCRIPT Human NovelVsBernettTrain",
            "D-SCRIPT Human NovelVsBernettAll",
            "D-SCRIPT Yeast Full",
        ],
        "training_or_finetuning_on_external_data": False,
        "seeds": SEEDS,
        "fusion": {
            "space": "logit",
            "alpha_b2": ALPHA_B2,
            "alpha_b0": ALPHA_B0,
        },
        "classification_threshold": THRESHOLD,
        "note": (
            "Human Full is a cross-dataset evaluation but is not strict cold-start because "
            "exact-sequence overlap with Bernett exists. Human NovelVsBernettTrain is the "
            "primary within-species strict external cold-start subset. Human NovelVsBernettAll "
            "is the ultra-strict subset excluding exact-sequence overlap with every Bernett split. "
            "Yeast Full has zero exact-sequence overlap with Bernett Train/Val/Test and is the "
            "primary cross-species zero-shot evaluation."
        ),
    }

    (
        OUT_ROOT
        / "external_evaluation_protocol_v2.json"
    ).write_text(
        json.dumps(
            protocol,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 120)
    print(
        "EXTERNAL GENERALIZATION SUMMARY"
    )
    print("=" * 120)

    display_cols = [
        "dataset",
        "model",
        "pairs",
        "prevalence",
        "random_auprc_baseline",
        "auprc_mean",
        "auprc_std",
        "auprc_over_random_mean",
        "auroc_mean",
        "auroc_std",
        "f1_mean",
        "mcc_mean",
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
    print(
        "PAIRED AUPRC DELTAS"
    )
    print("=" * 120)

    print(
        deltas.to_string(
            index=False
        )
    )

    print()
    print(
        f"Saved to: {OUT_ROOT}"
    )

    print()
    print(
        "IMPORTANT: No external labels were used for training, "
        "threshold calibration, architecture selection, or alpha tuning."
    )


if __name__ == "__main__":
    main()
