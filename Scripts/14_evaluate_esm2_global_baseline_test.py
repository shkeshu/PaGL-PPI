# -*- coding: utf-8 -*-

"""
14_evaluate_esm2_global_baseline_test.py

================================================================================
ESM2-GP BASELINE — BERNNETT LOCKED TEST EVALUATION
================================================================================

Purpose
-------
对 Step 17 训练完成的 ESM2-GP baseline 进行 Bernett Intra2 Test 推理。

重要原则
--------
1. 本脚本只做 inference / evaluation，不训练模型。
2. 模型结构、Dataset、embedding读取方式均直接复用：
       13_train_esm2_global_baseline.py
3. 每个 seed 加载对应的 best validation checkpoint。
4. classification threshold 固定为 0.5。
5. 不允许使用 Test 结果重新选择模型或调整超参数。
6. 输出五随机种子的 mean ± sample std (ddof=1)。

Input
-----
Data/processed/canonical/test_pairs.csv

Results/esm2_global_baseline/
    best_seed_42.pt
    best_seed_123.pt
    best_seed_456.pt
    best_seed_789.pt
    best_seed_1234.pt

Output
------
Results/esm2_global_baseline/

    test_seed_metrics.csv
    test_metrics_summary.csv
    test_metrics.json
    test_predictions.csv

================================================================================
"""


from pathlib import Path
import json
import importlib.util

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

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
# 1. 项目路径
# =============================================================================

ROOT = Path(
    str(Path(__file__).resolve().parents[1])
)

SCRIPT19_PATH = (
    ROOT
    /
    "Scripts"
    /
    "13_train_esm2_global_baseline.py"
)

TEST_FILE = (
    ROOT
    /
    "Data"
    /
    "processed"
    /
    "canonical"
    /
    "test_pairs.csv"
)


# =============================================================================
# 2. 动态加载 Step 17
# =============================================================================
#
# 注意：
# 文件名以数字16开头，不能直接:
#
#     import 16_train_esm2_global_baseline
#
# 因此使用 importlib 加载。
#
# 这样 Step 18 不再重复写 Model/Dataset，
# 保证和 Step 17 使用完全相同的代码。
# =============================================================================

if not SCRIPT19_PATH.exists():

    raise FileNotFoundError(
        f"\nCannot find Step 17 script:\n{SCRIPT19_PATH}\n"
    )


spec = importlib.util.spec_from_file_location(
    "esm2_global_baseline_step19",
    SCRIPT19_PATH
)

step19 = importlib.util.module_from_spec(spec)

spec.loader.exec_module(step19)


# =============================================================================
# 3. 从 Step 17 直接复用
# =============================================================================

BernettDataset = step19.BernettDataset

ESM2GlobalBaseline = step19.ESM2GlobalBaseline

SAVE_DIR = step19.SAVE_DIR

BATCH_SIZE = step19.BATCH_SIZE


# =============================================================================
# 4. Test evaluation configuration
# =============================================================================

SEEDS = [
    42,
    123,
    456,
    789,
    1234,
]


DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


THRESHOLD = 0.50


# =============================================================================
# 5. 安全加载 checkpoint
# =============================================================================

def load_checkpoint(path, device):

    """
    优先使用 weights_only=True。

    如果当前 PyTorch 版本不支持，
    自动回退到普通 torch.load。
    """

    try:

        return torch.load(
            path,
            map_location=device,
            weights_only=True
        )

    except TypeError:

        return torch.load(
            path,
            map_location=device
        )

    except Exception as e:

        print(
            f"[Warning] weights_only=True failed:\n"
            f"{e}\n"
            f"Fallback to standard torch.load..."
        )

        return torch.load(
            path,
            map_location=device
        )


# =============================================================================
# 6. 单个 seed Test 推理
# =============================================================================

def evaluate_one_seed(
    seed,
    dataset,
    loader
):

    print("\n" + "=" * 100)

    print(
        f"Evaluating ESM2-GP | Seed = {seed}"
    )

    print("=" * 100)


    # -------------------------------------------------------------------------
    # checkpoint
    # -------------------------------------------------------------------------

    ckpt_path = (
        SAVE_DIR
        /
        f"best_seed_{seed}.pt"
    )


    if not ckpt_path.exists():

        raise FileNotFoundError(
            f"\nMissing checkpoint:\n{ckpt_path}\n"
        )


    checkpoint = load_checkpoint(
        ckpt_path,
        DEVICE
    )


    # -------------------------------------------------------------------------
    # 模型
    # -------------------------------------------------------------------------

    model = ESM2GlobalBaseline()

    model.load_state_dict(
        checkpoint["model"],
        strict=True
    )

    model.to(DEVICE)

    model.eval()


    best_val_auprc = checkpoint.get(
        "best_val_auprc",
        np.nan
    )


    print(
        f"Checkpoint       : {ckpt_path.name}"
    )

    print(
        f"Best Val AUPRC  : {best_val_auprc:.6f}"
    )

    print(
        f"Device           : {DEVICE}"
    )


    # -------------------------------------------------------------------------
    # Test inference
    # -------------------------------------------------------------------------

    all_labels = []

    all_probs = []


    with torch.no_grad():

        for batch_idx, (a, b, y) in enumerate(loader):

            a = a.to(
                DEVICE,
                non_blocking=True
            )

            b = b.to(
                DEVICE,
                non_blocking=True
            )


            logits = model(
                a,
                b
            )


            probs = torch.sigmoid(
                logits
            )


            all_probs.append(
                probs.detach().cpu().numpy()
            )

            all_labels.append(
                y.detach().cpu().numpy()
            )


            if (
                batch_idx == 0
                or
                (batch_idx + 1) % 200 == 0
            ):

                print(
                    f"  batch "
                    f"{batch_idx + 1:4d}"
                    f"/"
                    f"{len(loader):4d}"
                )


    labels = np.concatenate(
        all_labels
    ).astype(np.int64)


    probs = np.concatenate(
        all_probs
    ).astype(np.float64)


    preds = (
        probs >= THRESHOLD
    ).astype(np.int64)


    # -------------------------------------------------------------------------
    # 数据完整性检查
    # -------------------------------------------------------------------------

    if len(labels) != len(dataset):

        raise RuntimeError(
            f"Prediction count mismatch: "
            f"{len(labels)} vs {len(dataset)}"
        )


    if len(probs) != len(dataset):

        raise RuntimeError(
            f"Probability count mismatch: "
            f"{len(probs)} vs {len(dataset)}"
        )


    if not np.all(
        np.isfinite(probs)
    ):

        raise RuntimeError(
            "Non-finite prediction probability detected."
        )


    # -------------------------------------------------------------------------
    # Confusion matrix
    # -------------------------------------------------------------------------

    tn, fp, fn, tp = confusion_matrix(
        labels,
        preds,
        labels=[0, 1]
    ).ravel()


    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )


    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------

    metrics = {

        "seed":
            int(seed),

        "best_val_auprc":
            float(best_val_auprc),

        "threshold":
            float(THRESHOLD),

        "n_test":
            int(len(labels)),

        "n_positive":
            int(np.sum(labels == 1)),

        "n_negative":
            int(np.sum(labels == 0)),

        "accuracy":
            float(
                accuracy_score(
                    labels,
                    preds
                )
            ),

        "precision":
            float(
                precision_score(
                    labels,
                    preds,
                    zero_division=0
                )
            ),

        "recall":
            float(
                recall_score(
                    labels,
                    preds,
                    zero_division=0
                )
            ),

        "specificity":
            float(
                specificity
            ),

        "f1":
            float(
                f1_score(
                    labels,
                    preds,
                    zero_division=0
                )
            ),

        "mcc":
            float(
                matthews_corrcoef(
                    labels,
                    preds
                )
            ),

        "auroc":
            float(
                roc_auc_score(
                    labels,
                    probs
                )
            ),

        "auprc":
            float(
                average_precision_score(
                    labels,
                    probs
                )
            ),

        "tp":
            int(tp),

        "tn":
            int(tn),

        "fp":
            int(fp),

        "fn":
            int(fn),
    }


    print("\nTest metrics")

    print("-" * 100)

    print(
        f"Accuracy     : "
        f"{metrics['accuracy']:.6f}"
    )

    print(
        f"Precision    : "
        f"{metrics['precision']:.6f}"
    )

    print(
        f"Recall       : "
        f"{metrics['recall']:.6f}"
    )

    print(
        f"Specificity  : "
        f"{metrics['specificity']:.6f}"
    )

    print(
        f"F1           : "
        f"{metrics['f1']:.6f}"
    )

    print(
        f"MCC          : "
        f"{metrics['mcc']:.6f}"
    )

    print(
        f"AUROC        : "
        f"{metrics['auroc']:.6f}"
    )

    print(
        f"AUPRC        : "
        f"{metrics['auprc']:.6f}"
    )

    print(
        f"TP/TN/FP/FN  : "
        f"{tp}/{tn}/{fp}/{fn}"
    )


    return (
        metrics,
        labels,
        probs,
        preds
    )


# =============================================================================
# 7. 汇总 mean ± std
# =============================================================================

def build_summary(
    metrics_df
):

    metric_names = [

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


    for metric in metric_names:

        values = (
            metrics_df[metric]
            .astype(float)
            .to_numpy()
        )


        mean_value = float(
            np.mean(values)
        )


        # -------------------------------------------------------------
        # 与已有 five-seed tables 保持一致：
        # sample standard deviation, ddof=1
        # -------------------------------------------------------------

        std_value = float(
            np.std(
                values,
                ddof=1
            )
        )


        rows.append({

            "metric":
                metric,

            "n_seeds":
                len(values),

            "mean":
                mean_value,

            "std":
                std_value,

            "min":
                float(
                    np.min(values)
                ),

            "max":
                float(
                    np.max(values)
                ),

            "paper_format":
                f"{mean_value:.4f} ± {std_value:.4f}",

        })


    return pd.DataFrame(
        rows
    )


# =============================================================================
# 8. 主程序
# =============================================================================

def main():

    print("=" * 110)

    print(
        "ESM2-GP BASELINE — BERNNETT PROTEIN-DISJOINT TEST"
    )

    print("=" * 110)

    print(
        f"Project       : {ROOT}"
    )

    print(
        f"Step 17       : {SCRIPT19_PATH}"
    )

    print(
        f"Test file     : {TEST_FILE}"
    )

    print(
        f"Checkpoint dir: {SAVE_DIR}"
    )

    print(
        f"Device        : {DEVICE}"
    )

    print(
        f"Threshold     : {THRESHOLD}"
    )

    print(
        f"Seeds         : {SEEDS}"
    )

    print("=" * 110)


    # -------------------------------------------------------------------------
    # 输入检查
    # -------------------------------------------------------------------------

    if not TEST_FILE.exists():

        raise FileNotFoundError(
            f"\nTest file not found:\n{TEST_FILE}\n"
        )


    missing_ckpts = [

        seed

        for seed in SEEDS

        if not (
            SAVE_DIR
            /
            f"best_seed_{seed}.pt"
        ).exists()

    ]


    if missing_ckpts:

        raise FileNotFoundError(

            "Missing checkpoints for seeds: "
            +
            str(missing_ckpts)

        )


    # -------------------------------------------------------------------------
    # Test dataset
    # -------------------------------------------------------------------------

    test_dataset = BernettDataset(
        TEST_FILE
    )


    test_loader = DataLoader(

        test_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=0,

        pin_memory=(
            DEVICE == "cuda"
        )

    )


    print(
        f"\nTest samples   : {len(test_dataset)}"
    )


    # -------------------------------------------------------------------------
    # 读取原始 test CSV
    # -------------------------------------------------------------------------

    test_df = pd.read_csv(
        TEST_FILE
    ).reset_index(
        drop=True
    )


    if len(test_df) != len(test_dataset):

        raise RuntimeError(
            "Test dataframe and dataset length mismatch."
        )


    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------

    all_seed_metrics = []

    prediction_df = (
        test_df.copy()
    )


    reference_labels = None


    for seed in SEEDS:

        (
            metrics,
            labels,
            probs,
            preds

        ) = evaluate_one_seed(

            seed,

            test_dataset,

            test_loader

        )


        all_seed_metrics.append(
            metrics
        )


        # -------------------------------------------------------------
        # 确保每个 seed 使用完全相同的 labels
        # -------------------------------------------------------------

        if reference_labels is None:

            reference_labels = labels.copy()

        else:

            if not np.array_equal(
                reference_labels,
                labels
            ):

                raise RuntimeError(
                    f"Label mismatch detected at seed {seed}"
                )


        # -------------------------------------------------------------
        # 保存每个 seed 的 probability / prediction
        # -------------------------------------------------------------

        prediction_df[
            f"prob_seed_{seed}"
        ] = probs


        prediction_df[
            f"pred_seed_{seed}"
        ] = preds


    # -------------------------------------------------------------------------
    # Per-seed metrics
    # -------------------------------------------------------------------------

    seed_metrics_df = pd.DataFrame(
        all_seed_metrics
    )


    seed_metrics_path = (
        SAVE_DIR
        /
        "test_seed_metrics.csv"
    )


    seed_metrics_df.to_csv(

        seed_metrics_path,

        index=False,

        encoding="utf-8-sig"

    )


    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    summary_df = build_summary(
        seed_metrics_df
    )


    summary_path = (
        SAVE_DIR
        /
        "test_metrics_summary.csv"
    )


    summary_df.to_csv(

        summary_path,

        index=False,

        encoding="utf-8-sig"

    )


    # -------------------------------------------------------------------------
    # Prediction mean / std
    # -------------------------------------------------------------------------

    prob_columns = [

        f"prob_seed_{seed}"

        for seed in SEEDS

    ]


    prediction_df[
        "prob_mean"
    ] = prediction_df[
        prob_columns
    ].mean(
        axis=1
    )


    # 与多seed统计保持一致：
    # sample std
    prediction_df[
        "prob_std"
    ] = prediction_df[
        prob_columns
    ].std(
        axis=1,
        ddof=1
    )


    prediction_df[
        "pred_from_mean_prob"
    ] = (

        prediction_df[
            "prob_mean"
        ]

        >= THRESHOLD

    ).astype(int)


    predictions_path = (
        SAVE_DIR
        /
        "test_predictions.csv"
    )


    prediction_df.to_csv(

        predictions_path,

        index=False,

        encoding="utf-8-sig"

    )


    # -------------------------------------------------------------------------
    # JSON summary
    # -------------------------------------------------------------------------

    summary_json = {}


    for _, row in summary_df.iterrows():

        summary_json[
            row["metric"]
        ] = {

            "mean":
                float(row["mean"]),

            "std":
                float(row["std"]),

            "min":
                float(row["min"]),

            "max":
                float(row["max"]),

            "paper_format":
                row["paper_format"],

        }


    output_json = {

        "experiment":
            "ESM2-GP baseline Bernett protein-disjoint Test",

        "evaluation_type":
            "inference-only",

        "source_training_script":
            str(SCRIPT19_PATH),

        "test_file":
            str(TEST_FILE),

        "threshold":
            THRESHOLD,

        "seeds":
            SEEDS,

        "std_definition":
            "sample standard deviation, ddof=1",

        "seed_results":
            all_seed_metrics,

        "summary":
            summary_json,

    }


    json_path = (
        SAVE_DIR
        /
        "test_metrics.json"
    )


    with open(

        json_path,

        "w",

        encoding="utf-8"

    ) as f:

        json.dump(

            output_json,

            f,

            indent=2,

            ensure_ascii=False

        )


    # -------------------------------------------------------------------------
    # Console final summary
    # -------------------------------------------------------------------------

    print("\n")

    print("=" * 110)

    print(
        "FINAL 5-SEED TEST SUMMARY"
    )

    print("=" * 110)


    for _, row in summary_df.iterrows():

        print(

            f"{row['metric']:<12s}: "
            f"{row['mean']:.4f} "
            f"± "
            f"{row['std']:.4f}"

        )


    print("=" * 110)


    print("\nOutputs:")

    print(
        f"  {seed_metrics_path}"
    )

    print(
        f"  {summary_path}"
    )

    print(
        f"  {json_path}"
    )

    print(
        f"  {predictions_path}"
    )


    print(
        "\nEvaluation completed successfully."
    )


# =============================================================================
# 9. Entry
# =============================================================================

if __name__ == "__main__":

    main()