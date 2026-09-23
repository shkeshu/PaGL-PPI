# -*- coding: utf-8 -*-

"""Build manuscript Tables 3–7 from generated metric summaries.

Run after the Bernett, ESM2-GP, and external evaluation steps. The compact
entry point requires only five-seed metric summaries and does not require the
original full-package diagnostic archive or its 113-check audit.

Output: Results/paper_results_final_v2/tables/*.csv and a Markdown summary.
Literature values in Table 3 are reference comparisons, not a controlled
head-to-head re-evaluation.
"""


from __future__ import annotations

import json
import math
import platform
import sys

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# 1. Version
# =============================================================================

SCRIPT_VERSION = "3.0"


# =============================================================================
# 2. Project
# =============================================================================

SCRIPT_DIR = Path(
    __file__
).resolve().parent

ROOT = SCRIPT_DIR.parent

RESULTS = (
    ROOT
    /
    "Results"
)


# =============================================================================
# 3. Input paths
# =============================================================================

# -------------------------------------------------------------------------
# Bernett original models
# -------------------------------------------------------------------------

VAL_SUMMARY = (
    RESULTS
    /
    "fixed_fusion_5seed"
    /
    "five_seed_model_summary.csv"
)


TEST_SUMMARY = (
    RESULTS
    /
    "final_locked_test"
    /
    "test_five_seed_model_summary.csv"
)


# -------------------------------------------------------------------------
# Controlled ablation
# -------------------------------------------------------------------------

ABLATION_SUMMARY = (
    RESULTS
    /
    "multiseed_confirmation"
    /
    "multiseed_model_summary.csv"
)


# -------------------------------------------------------------------------
# ESM2-GP
# -------------------------------------------------------------------------

ESM2GP_VAL = (
    RESULTS
    /
    "esm2_global_baseline"
    /
    "summary.json"
)


ESM2GP_TEST = (
    RESULTS
    /
    "esm2_global_baseline"
    /
    "test_metrics_summary.csv"
)


# -------------------------------------------------------------------------
# Original external B0 / B2 / Fusion
# -------------------------------------------------------------------------

EXTERNAL_ORIGINAL = (
    RESULTS
    /
    "external_generalization"
    /
    "external_model_summary_v2.csv"
)


EXTERNAL_ORIGINAL_PROTOCOL = (
    RESULTS
    /
    "external_generalization"
    /
    "external_evaluation_protocol_v2.json"
)


# -------------------------------------------------------------------------
# ESM2-GP external
# -------------------------------------------------------------------------

EXTERNAL_ESM2GP = (
    RESULTS
    /
    "external_generalization"
    /
    "esm2_gp_external_summary.csv"
)


EXTERNAL_ESM2GP_PROTOCOL = (
    RESULTS
    /
    "external_generalization"
    /
    "esm2_gp_external_protocol.json"
)


# -------------------------------------------------------------------------
# Step 15
# -------------------------------------------------------------------------

AUDIT_DIR = (
    RESULTS
    /
    "second_round"
    /
    "dataset_audit"
)


AUDIT_SUMMARY = (
    AUDIT_DIR
    /
    "dataset_audit_summary.json"
)


AUDIT_TABLE = (
    AUDIT_DIR
    /
    "table01_dataset_statistics.csv"
)


AUDIT_SEQUENCE_OVERLAP = (
    AUDIT_DIR
    /
    "table04_external_sequence_overlap_audit.csv"
)


# -------------------------------------------------------------------------
# Step 19
# -------------------------------------------------------------------------

SIM_DIR = (
    RESULTS
    /
    "second_round"
    /
    "val_test_similarity_v2"
)


SIM_SUMMARY = (
    SIM_DIR
    /
    "val_test_similarity_v2_summary.json"
)


SIM_MODEL_SHIFT = (
    SIM_DIR
    /
    "table01_model_auprc_shift.csv"
)


# =============================================================================
# 4. Output
# =============================================================================

OUT_ROOT = (
    RESULTS
    /
    "paper_results_final_v2"
)


TABLE_DIR = (
    OUT_ROOT
    /
    "tables"
)


SUPPORT_DIR = (
    OUT_ROOT
    /
    "supporting"
)


TEXT_DIR = (
    OUT_ROOT
    /
    "text"
)


for d in [
    OUT_ROOT,
    TABLE_DIR,
    SUPPORT_DIR,
    TEXT_DIR,
]:

    d.mkdir(
        parents=True,
        exist_ok=True
    )


# =============================================================================
# 5. Model names
# =============================================================================

MODEL_NAME_MAP = {

    "B0_strong":
        "Global branch",

    "B2_cross":
        "Partner-aware Local",

    "FixedFusion_0.70B2_0.30B0":
        "PaGL-PPI",

}


ABLATION_NAME_MAP = {

    "B0_strong_sum_diff_product":
        (
            "B0",
            "Global branch"
        ),

    "B1_segment_only":
        (
            "B1",
            "Segment-only"
        ),

    "B2_segment_cross":
        (
            "B2",
            "Partner-aware Local"
        ),

    "B3_cross_global_concat":
        (
            "B3",
            "Cross-attention + global feature concatenation"
        ),

}


# =============================================================================
# 6. Literature registry for Table 3
#
# 注意：
# 这部分属于当前论文采用的 literature-reported values。
# 脚本不会声称这些结果与本文协议完全一致。
#
# 投稿前仍应逐项与原论文最终核对。
# =============================================================================

LITERATURE_ROWS = [

    {
        "Method":
            "Topsy-Turvy",

        "Representation":
            "Sequence / network representation",

        "Evaluation":
            "Reported Bernett result",

        "AUPRC":
            "0.590",

        "Comparison_scope":
            "Literature-reported; protocol may differ",
    },

    {
        "Method":
            "TUnA",

        "Representation":
            "Protein language model",

        "Evaluation":
            "Reported Bernett result",

        "AUPRC":
            "≈0.690",

        "Comparison_scope":
            "Literature-reported; protocol may differ",
    },

    {
        "Method":
            "Pool PaRTI",

        "Representation":
            "Protein language model",

        "Evaluation":
            "Reported Bernett result",

        "AUPRC":
            "≈0.701",

        "Comparison_scope":
            "Literature-reported; protocol may differ",
    },

    {
        "Method":
            "C3PI",

        "Representation":
            "Protein language model",

        "Evaluation":
            "Reported Bernett result",

        "AUPRC":
            "≈0.695",

        "Comparison_scope":
            "Literature-reported; protocol may differ",
    },

    {
        "Method":
            "ESM2 Mean Pooling",

        "Representation":
            "ESM2-650M",

        "Evaluation":
            "Reported Bernett result",

        "AUPRC":
            "≈0.717",

        "Comparison_scope":
            "Literature-reported; protocol may differ",
    },

]


# =============================================================================
# 7. Utilities
# =============================================================================


def require(
    path: Path
):

    if not path.exists():

        raise FileNotFoundError(
            f"\nRequired file not found:\n{path}\n"
        )



def load_json(
    path: Path
):

    require(path)

    with path.open(
        "r",
        encoding="utf-8-sig"
    ) as f:

        return json.load(f)



def load_csv(
    path: Path
):

    require(path)

    return pd.read_csv(path)



def fmt(
    mean,
    std,
    digits=4
):

    return (
        f"{float(mean):.{digits}f} "
        f"± "
        f"{float(std):.{digits}f}"
    )



def safe_float(
    x
):

    x = float(x)

    if not np.isfinite(x):

        raise RuntimeError(
            f"Non-finite value: {x}"
        )

    return x



def save_csv(
    df,
    path
):

    df.to_csv(
        path,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"[Saved] {path}"
    )



def markdown_table(
    df: pd.DataFrame
):

    """
    不依赖 tabulate。
    """

    cols = list(
        df.columns
    )


    lines = []


    lines.append(
        "| "
        +
        " | ".join(cols)
        +
        " |"
    )


    lines.append(
        "| "
        +
        " | ".join(
            ["---"] * len(cols)
        )
        +
        " |"
    )


    for _, row in df.iterrows():

        vals = []

        for c in cols:

            v = row[c]

            if pd.isna(v):

                v = ""

            vals.append(
                str(v)
            )

        lines.append(
            "| "
            +
            " | ".join(vals)
            +
            " |"
        )


    return "\n".join(
        lines
    )


# =============================================================================
# 8. Protocol gates
# =============================================================================


def validate_protocols():

    print("=" * 110)

    print(
        "STEP 23 — PROTOCOL VALIDATION"
    )

    print("=" * 110)


    # -------------------------------------------------------------------------
    # Dataset Audit
    # -------------------------------------------------------------------------

    audit = load_json(
        AUDIT_SUMMARY
    )


    if str(
        audit.get(
            "status",
            ""
        )
    ).upper() != "PASS":

        raise RuntimeError(
            "Step 15 Dataset Audit is not PASS."
        )


    if int(
        audit.get(
            "checks",
            {}
        ).get(
            "failed",
            -1
        )
    ) != 0:

        raise RuntimeError(
            "Step 15 contains failed checks."
        )


    if not bool(
        audit.get(
            "bernett_protein_disjoint",
            False
        )
    ):

        raise RuntimeError(
            "Bernett is not protein-disjoint."
        )


    print(
        "[PASS] Step 15 Dataset Audit"
    )


    # -------------------------------------------------------------------------
    # Validation/Test similarity
    # -------------------------------------------------------------------------

    sim = load_json(
        SIM_SUMMARY
    )


    if str(
        sim.get(
            "dataset_audit_gate",
            ""
        )
    ).upper() != "PASS":

        raise RuntimeError(
            "Step 19 dataset gate is not PASS."
        )


    causal_claim = (
        sim
        .get(
            "interpretation",
            {}
        )
        .get(
            "causal_claim",
            True
        )
    )


    if causal_claim is not False:

        raise RuntimeError(
            "Step 19 must remain descriptive, not causal."
        )


    print(
        "[PASS] Step 19 Validation/Test analysis"
    )


    # -------------------------------------------------------------------------
    # Original external protocol
    # -------------------------------------------------------------------------

    p1 = load_json(
        EXTERNAL_ORIGINAL_PROTOCOL
    )


    if p1.get(
        "training_or_finetuning_on_external_data",
        False
    ):

        raise RuntimeError(
            "Original external protocol contains external training."
        )


    status = str(
        p1.get(
            "status",
            ""
        )
    )


    if "LOCKED_EXTERNAL" not in status:

        raise RuntimeError(
            "Original external protocol is not locked."
        )


    print(
        "[PASS] Original B0/B2/Fusion external protocol"
    )


    # -------------------------------------------------------------------------
    # ESM2-GP external protocol
    # -------------------------------------------------------------------------

    p2 = load_json(
        EXTERNAL_ESM2GP_PROTOCOL
    )


    if p2.get(
        "external_tuning",
        True
    ):

        raise RuntimeError(
            "ESM2-GP external tuning detected."
        )


    if p2.get(
        "external_finetuning",
        True
    ):

        raise RuntimeError(
            "ESM2-GP external fine-tuning detected."
        )


    if str(
        p2.get(
            "evaluation_type",
            ""
        )
    ).lower() != "inference-only":

        raise RuntimeError(
            "ESM2-GP external evaluation is not inference-only."
        )


    print(
        "[PASS] ESM2-GP external protocol"
    )


    print("=" * 110)


    return (
        audit,
        sim,
        p1,
        p2,
    )


# =============================================================================
# 9. Load ESM2-GP Test metrics
# =============================================================================


def load_esm2gp_test():

    df = load_csv(
        ESM2GP_TEST
    )


    required = {
        "metric",
        "mean",
        "std",
    }


    missing = (
        required
        -
        set(
            df.columns
        )
    )


    if missing:

        raise RuntimeError(
            f"ESM2-GP Test missing columns: {missing}"
        )


    result = {}


    for _, row in df.iterrows():

        metric = str(
            row["metric"]
        ).lower()


        result[metric] = {

            "mean":
                safe_float(
                    row["mean"]
                ),

            "std":
                safe_float(
                    row["std"]
                ),

        }


    return result


# =============================================================================
# 10. Table 3
# =============================================================================


def build_table03(
    test_df,
    esm2gp
):

    fusion = test_df.loc[

        test_df[
            "model"
        ]
        ==
        "FixedFusion_0.70B2_0.30B0"

    ]


    if len(
        fusion
    ) != 1:

        raise RuntimeError(
            "Cannot uniquely locate PaGL-PPI Test result."
        )


    fusion = fusion.iloc[0]


    rows = list(
        LITERATURE_ROWS
    )


    rows.append({

        "Method":
            "ESM2-GP (ours)",

        "Representation":
            "Frozen ESM2-t33-650M protein-level representation",

        "Evaluation":
            "Bernett Intra2 protein-disjoint",

        "AUPRC":
            fmt(
                esm2gp[
                    "auprc"
                ][
                    "mean"
                ],
                esm2gp[
                    "auprc"
                ][
                    "std"
                ]
            ),

        "Comparison_scope":
            "Controlled result in this work",
    })


    rows.append({

        "Method":
            "PaGL-PPI (ours)",

        "Representation":
            "ESM2 global + partner-aware local interaction",

        "Evaluation":
            "Bernett Intra2 protein-disjoint",

        "AUPRC":
            fmt(
                fusion[
                    "auprc_mean"
                ],
                fusion[
                    "auprc_std"
                ]
            ),

        "Comparison_scope":
            "Controlled result in this work",
    })


    return pd.DataFrame(
        rows
    )


# =============================================================================
# 11. Table 4 — controlled Bernett Test
# =============================================================================


def build_table04(
    test_df,
    esm2gp
):

    rows = []


    order = [

        (
            "B0_strong",
            "Global branch"
        ),

        (
            None,
            "ESM2-GP"
        ),

        (
            "B2_cross",
            "Partner-aware Local"
        ),

        (
            "FixedFusion_0.70B2_0.30B0",
            "PaGL-PPI"
        ),

    ]


    metrics = [

        "accuracy",

        "f1",

        "mcc",

        "auroc",

        "auprc",

    ]


    for raw_name, paper_name in order:


        row = {
            "Model":
                paper_name
        }


        if raw_name is None:

            for metric in metrics:

                row[
                    metric.upper()
                ] = fmt(

                    esm2gp[
                        metric
                    ][
                        "mean"
                    ],

                    esm2gp[
                        metric
                    ][
                        "std"
                    ]

                )


                row[
                    f"_{metric}_mean"
                ] = esm2gp[
                    metric
                ][
                    "mean"
                ]


        else:

            source = test_df.loc[
                test_df[
                    "model"
                ]
                ==
                raw_name
            ]


            if len(
                source
            ) != 1:

                raise RuntimeError(
                    f"Cannot uniquely locate {raw_name}"
                )


            source = source.iloc[0]


            for metric in metrics:

                row[
                    metric.upper()
                ] = fmt(

                    source[
                        f"{metric}_mean"
                    ],

                    source[
                        f"{metric}_std"
                    ]

                )


                row[
                    f"_{metric}_mean"
                ] = float(
                    source[
                        f"{metric}_mean"
                    ]
                )


        rows.append(
            row
        )


    out = pd.DataFrame(
        rows
    )


    return out


# =============================================================================
# 12. Table 5 — controlled ablation
# =============================================================================


def build_table05(
    ablation_df
):

    rows = []


    b0_auprc = float(

        ablation_df.loc[

            ablation_df[
                "model"
            ]
            ==
            "B0_strong_sum_diff_product",

            "auprc_mean"

        ].iloc[0]

    )


    for raw_model, (
        ablation_id,
        description
    ) in ABLATION_NAME_MAP.items():


        source = ablation_df.loc[

            ablation_df[
                "model"
            ]
            ==
            raw_model

        ]


        if len(
            source
        ) != 1:

            raise RuntimeError(
                f"Missing controlled ablation: {raw_model}"
            )


        source = source.iloc[0]


        auprc = float(
            source[
                "auprc_mean"
            ]
        )


        rows.append({

            "Ablation":
                ablation_id,

            "Configuration":
                description,

            "Seeds":
                int(
                    source[
                        "n_seeds"
                    ]
                ),

            "AUPRC":
                fmt(
                    source[
                        "auprc_mean"
                    ],
                    source[
                        "auprc_std"
                    ]
                ),

            "AUROC":
                fmt(
                    source[
                        "auroc_mean"
                    ],
                    source[
                        "auroc_std"
                    ]
                ),

            "MCC":
                fmt(
                    source[
                        "mcc_mean"
                    ],
                    source[
                        "mcc_std"
                    ]
                ),

            "ΔAUPRC_vs_B0":
                f"{auprc - b0_auprc:+.4f}",

            "_auprc_mean":
                auprc,

        })


    return pd.DataFrame(
        rows
    )


# =============================================================================
# 13. External normalization
# =============================================================================


ORIGINAL_EXTERNAL_DATASET_MAP = {

    "D-SCRIPT_Human_Full":
        "Human Full",

    "D-SCRIPT_Human_NovelVsBernettTrain":
        "Human NovelVsTrain",

    "D-SCRIPT_Human_NovelVsBernettAll":
        "Human NovelVsAll",

    "D-SCRIPT_Yeast_Full":
        "Yeast Full",

}


NEW_EXTERNAL_DATASET_MAP = {

    "Human_Full":
        "Human Full",

    "Human_NovelVsTrain":
        "Human NovelVsTrain",

    "Human_NovelVsAll":
        "Human NovelVsAll",

    "Yeast_Full":
        "Yeast Full",

}


EXTERNAL_MODEL_MAP = {

    "B0_strong":
        "Global branch",

    "B2_cross":
        "Partner-aware Local",

    "FixedFusion_0.70B2_0.30B0":
        "PaGL-PPI",

}


# =============================================================================
# 14. Merge external results
# =============================================================================


def merge_external(
    original_df,
    esm2_df
):

    rows = []


    # -------------------------------------------------------------------------
    # Original models
    # -------------------------------------------------------------------------

    for _, r in original_df.iterrows():


        dataset = ORIGINAL_EXTERNAL_DATASET_MAP.get(
            str(
                r[
                    "dataset"
                ]
            )
        )


        model = EXTERNAL_MODEL_MAP.get(
            str(
                r[
                    "model"
                ]
            )
        )


        if dataset is None or model is None:

            continue


        rows.append({

            "Dataset":
                dataset,

            "Model":
                model,

            "Pairs":
                int(
                    r[
                        "pairs"
                    ]
                ),

            "Positive":
                int(
                    r[
                        "positive"
                    ]
                ),

            "Negative":
                int(
                    r[
                        "negative"
                    ]
                ),

            "Random_AUPRC":
                float(
                    r[
                        "random_auprc_baseline"
                    ]
                ),

            "Accuracy_mean":
                float(
                    r[
                        "accuracy_mean"
                    ]
                ),

            "Accuracy_std":
                float(
                    r[
                        "accuracy_std"
                    ]
                ),

            "F1_mean":
                float(
                    r[
                        "f1_mean"
                    ]
                ),

            "F1_std":
                float(
                    r[
                        "f1_std"
                    ]
                ),

            "MCC_mean":
                float(
                    r[
                        "mcc_mean"
                    ]
                ),

            "MCC_std":
                float(
                    r[
                        "mcc_std"
                    ]
                ),

            "AUROC_mean":
                float(
                    r[
                        "auroc_mean"
                    ]
                ),

            "AUROC_std":
                float(
                    r[
                        "auroc_std"
                    ]
                ),

            "AUPRC_mean":
                float(
                    r[
                        "auprc_mean"
                    ]
                ),

            "AUPRC_std":
                float(
                    r[
                        "auprc_std"
                    ]
                ),

            "AUPRC_over_random":
                float(
                    r[
                        "auprc_over_random_mean"
                    ]
                ),

        })


    # -------------------------------------------------------------------------
    # New ESM2-GP
    # -------------------------------------------------------------------------

    for _, r in esm2_df.iterrows():


        dataset = NEW_EXTERNAL_DATASET_MAP.get(
            str(
                r[
                    "dataset"
                ]
            )
        )


        if dataset is None:

            continue


        rows.append({

            "Dataset":
                dataset,

            "Model":
                "ESM2-GP",

            "Pairs":
                int(
                    r[
                        "n_pairs"
                    ]
                ),

            "Positive":
                int(
                    r[
                        "positive"
                    ]
                ),

            "Negative":
                int(
                    r[
                        "negative"
                    ]
                ),

            "Random_AUPRC":
                float(
                    r[
                        "random_auprc"
                    ]
                ),

            "Accuracy_mean":
                float(
                    r[
                        "accuracy_mean"
                    ]
                ),

            "Accuracy_std":
                float(
                    r[
                        "accuracy_std"
                    ]
                ),

            "F1_mean":
                float(
                    r[
                        "f1_mean"
                    ]
                ),

            "F1_std":
                float(
                    r[
                        "f1_std"
                    ]
                ),

            "MCC_mean":
                float(
                    r[
                        "mcc_mean"
                    ]
                ),

            "MCC_std":
                float(
                    r[
                        "mcc_std"
                    ]
                ),

            "AUROC_mean":
                float(
                    r[
                        "auroc_mean"
                    ]
                ),

            "AUROC_std":
                float(
                    r[
                        "auroc_std"
                    ]
                ),

            "AUPRC_mean":
                float(
                    r[
                        "auprc_mean"
                    ]
                ),

            "AUPRC_std":
                float(
                    r[
                        "auprc_std"
                    ]
                ),

            "AUPRC_over_random":
                float(
                    r[
                        "auprc_over_random_mean"
                    ]
                ),

        })


    out = pd.DataFrame(
        rows
    )


    expected_models = {

        "Global branch",

        "ESM2-GP",

        "Partner-aware Local",

        "PaGL-PPI",

    }


    expected_datasets = {

        "Human Full",

        "Human NovelVsTrain",

        "Human NovelVsAll",

        "Yeast Full",

    }


    for dataset in expected_datasets:

        sub = out.loc[
            out[
                "Dataset"
            ]
            ==
            dataset
        ]


        found = set(
            sub[
                "Model"
            ]
        )


        if found != expected_models:

            raise RuntimeError(

                f"{dataset} model set mismatch.\n"
                f"Found={found}\n"
                f"Expected={expected_models}"

            )


    return out


# =============================================================================
# 15. Format external table
# =============================================================================


def format_external_table(
    merged,
    datasets
):

    sub = merged.loc[

        merged[
            "Dataset"
        ].isin(
            datasets
        )

    ].copy()


    model_order = {

        "Global branch":
            0,

        "ESM2-GP":
            1,

        "Partner-aware Local":
            2,

        "PaGL-PPI":
            3,

    }


    dataset_order = {

        name:
            i

        for i, name in enumerate(
            datasets
        )

    }


    sub[
        "_dataset_order"
    ] = sub[
        "Dataset"
    ].map(
        dataset_order
    )


    sub[
        "_model_order"
    ] = sub[
        "Model"
    ].map(
        model_order
    )


    sub = sub.sort_values(

        [
            "_dataset_order",
            "_model_order",
        ]

    )


    # -------------------------------------------------------------------------
    # Determine best AUPRC per dataset
    # -------------------------------------------------------------------------

    best_map = (

        sub

        .groupby(
            "Dataset"
        )[
            "AUPRC_mean"
        ]

        .max()

        .to_dict()

    )


    stable_map = (

        sub

        .groupby(
            "Dataset"
        )[
            "AUPRC_std"
        ]

        .min()

        .to_dict()

    )


    rows = []


    for _, r in sub.iterrows():


        rows.append({

            "Setting":
                r[
                    "Dataset"
                ],

            "Model":
                r[
                    "Model"
                ],

            "Random AUPRC":
                f"{r['Random_AUPRC']:.4f}",

            "F1":
                fmt(
                    r[
                        "F1_mean"
                    ],
                    r[
                        "F1_std"
                    ]
                ),

            "MCC":
                fmt(
                    r[
                        "MCC_mean"
                    ],
                    r[
                        "MCC_std"
                    ]
                ),

            "AUROC":
                fmt(
                    r[
                        "AUROC_mean"
                    ],
                    r[
                        "AUROC_std"
                    ]
                ),

            "AUPRC":
                fmt(
                    r[
                        "AUPRC_mean"
                    ],
                    r[
                        "AUPRC_std"
                    ]
                ),

            "AUPRC / Random":
                f"{r['AUPRC_over_random']:.2f}×",

            "Best mean AUPRC":
                bool(
                    np.isclose(
                        r[
                            "AUPRC_mean"
                        ],
                        best_map[
                            r[
                                "Dataset"
                            ]
                        ]
                    )
                ),

            "Lowest AUPRC std":
                bool(
                    np.isclose(
                        r[
                            "AUPRC_std"
                        ],
                        stable_map[
                            r[
                                "Dataset"
                            ]
                        ]
                    )
                ),

            "_AUPRC_mean":
                r[
                    "AUPRC_mean"
                ],

            "_AUPRC_std":
                r[
                    "AUPRC_std"
                ],

        })


    return pd.DataFrame(
        rows
    )


# =============================================================================
# 16. Dataset audit supporting table
# =============================================================================


def build_audit_support(
    audit
):

    rows = []


    for split in [

        "train",
        "val",
        "test",

    ]:


        s = audit[
            "bernett"
        ][
            split
        ]


        rows.append({

            "Evidence":
                f"Bernett {split}",

            "Pairs":
                s[
                    "pairs"
                ],

            "Positive":
                s[
                    "positive"
                ],

            "Negative":
                s[
                    "negative"
                ],

            "Proteins":
                s[
                    "proteins"
                ],

            "Result":
                (
                    "Protein-disjoint"
                    if audit[
                        "bernett_protein_disjoint"
                    ]
                    else
                    "FAIL"
                ),

        })


    h = audit[
        "human"
    ]


    rows.append({

        "Evidence":
            "Human NovelVsTrain",

        "Pairs":
            h[
                "novel_vs_train_pairs"
            ],

        "Positive":
            "",

        "Negative":
            "",

        "Proteins":
            "",

        "Result":
            (
                "Exact subset membership PASS"
                if h[
                    "novel_vs_train_membership_exact"
                ]
                else
                "FAIL"
            ),

    })


    rows.append({

        "Evidence":
            "Human NovelVsAll",

        "Pairs":
            h[
                "novel_vs_all_pairs"
            ],

        "Positive":
            "",

        "Negative":
            "",

        "Proteins":
            "",

        "Result":
            (
                "Exact subset membership PASS"
                if h[
                    "novel_vs_all_membership_exact"
                ]
                else
                "FAIL"
            ),

    })


    rows.append({

        "Evidence":
            "Human exact-sequence overlap",

        "Pairs":
            "",

        "Positive":
            "",

        "Negative":
            "",

        "Proteins":
            "",

        "Result":
            (
                f"Train={h['exact_sequence_overlap']['train']}; "
                f"Val={h['exact_sequence_overlap']['val']}; "
                f"Test={h['exact_sequence_overlap']['test']}"
            ),

    })


    y = audit[
        "yeast"
    ]


    rows.append({

        "Evidence":
            "Yeast cross-species zero-shot",

        "Pairs":
            y[
                "full_pairs"
            ],

        "Positive":
            "",

        "Negative":
            "",

        "Proteins":
            "",

        "Result":
            (
                "Zero exact-sequence overlap with Train/Val/Test"
                if y[
                    "cross_species_zero_shot_eligible"
                ]
                else
                "FAIL"
            ),

    })


    return pd.DataFrame(
        rows
    )


# =============================================================================
# 17. Validation/Test evidence supporting table
# =============================================================================


def build_similarity_support(
    sim
):

    shifts = sim[
        "model_auprc_shift"
    ]


    rows = []


    for model, v in shifts.items():


        rows.append({

            "Evidence":
                f"{model} AUPRC",

            "Validation":
                f"{v['validation']:.6f}",

            "Test":
                f"{v['test']:.6f}",

            "Test − Validation":
                f"{v['test_minus_validation']:+.6f}",

            "Interpretation":
                "Consistent positive split shift",

        })


    p = sim[
        "pair_cosine"
    ]


    rows.append({

        "Evidence":
            "Positive-negative pair cosine gap",

        "Validation":
            f"{p['validation_gap']:.6f}",

        "Test":
            f"{p['test_gap']:.6f}",

        "Test − Validation":
            f"{p['test_gap'] - p['validation_gap']:+.6f}",

        "Interpretation":
            (
                f"Test gap is "
                f"{p['test_to_validation_gap_ratio']:.2f}× Validation"
            ),

    })


    n = sim[
        "nearest_train_cosine"
    ]


    rows.append({

        "Evidence":
            "Nearest-Train protein cosine",

        "Validation":
            f"{n['validation']:.6f}",

        "Test":
            f"{n['test']:.6f}",

        "Test − Validation":
            f"{n['test_minus_validation']:+.6f}",

        "Interpretation":
            "Test is not closer to Training",

    })


    c = sim[
        "cosine_only_classifier"
    ]


    rows.append({

        "Evidence":
            "Cosine-only AUPRC",

        "Validation":
            f"{c['validation_auprc']:.6f}",

        "Test":
            f"{c['test_auprc']:.6f}",

        "Test − Validation":
            (
                f"{c['test_auprc'] - c['validation_auprc']:+.6f}"
            ),

        "Interpretation":
            "Model-independent Test separability is higher",

    })


    ps = sim[
        "pagl_probability_separation"
    ]


    rows.append({

        "Evidence":
            "PaGL-PPI probability gap",

        "Validation":
            f"{ps['validation_gap']:.6f}",

        "Test":
            f"{ps['test_gap']:.6f}",

        "Test − Validation":
            f"{ps['test_minus_validation']:+.6f}",

        "Interpretation":
            "Positive/negative output separation increases",

    })


    return pd.DataFrame(
        rows
    )


# =============================================================================
# 18. Build manuscript markdown
# =============================================================================


def clean_hidden_columns(
    df
):

    return df[
        [
            c
            for c in df.columns
            if not str(c).startswith(
                "_"
            )
        ]
    ].copy()



def bold_best(
    df,
    metric_display,
    hidden_mean
):

    out = df.copy()


    best = float(
        out[
            hidden_mean
        ].max()
    )


    for idx, r in out.iterrows():

        if np.isclose(
            float(
                r[
                    hidden_mean
                ]
            ),
            best
        ):

            out.loc[
                idx,
                metric_display
            ] = (
                "**"
                +
                str(
                    r[
                        metric_display
                    ]
                )
                +
                "**"
            )


    return out



def table04_markdown(
    table04
):

    t = table04.copy()


    mapping = {

        "ACCURACY":
            "_accuracy_mean",

        "F1":
            "_f1_mean",

        "MCC":
            "_mcc_mean",

        "AUROC":
            "_auroc_mean",

        "AUPRC":
            "_auprc_mean",

    }


    for display, hidden in mapping.items():

        t = bold_best(
            t,
            display,
            hidden
        )


    return markdown_table(
        clean_hidden_columns(
            t
        )
    )



def external_markdown(
    table
):

    t = table.copy()


    for idx, row in t.iterrows():

        if bool(
            row[
                "Best mean AUPRC"
            ]
        ):

            t.loc[
                idx,
                "AUPRC"
            ] = (
                "**"
                +
                str(
                    row[
                        "AUPRC"
                    ]
                )
                +
                "**"
            )


    return markdown_table(

        clean_hidden_columns(
            t
        )

    )


# =============================================================================
# 19. Final numeric summary
# =============================================================================


def build_numeric_summary(
    audit,
    sim,
    table04,
    merged_external,
):

    def get_test(
        model
    ):

        row = table04.loc[
            table04[
                "Model"
            ]
            ==
            model
        ].iloc[0]


        return {

            "accuracy":
                float(
                    row[
                        "_accuracy_mean"
                    ]
                ),

            "f1":
                float(
                    row[
                        "_f1_mean"
                    ]
                ),

            "mcc":
                float(
                    row[
                        "_mcc_mean"
                    ]
                ),

            "auroc":
                float(
                    row[
                        "_auroc_mean"
                    ]
                ),

            "auprc":
                float(
                    row[
                        "_auprc_mean"
                    ]
                ),

        }


    pagl = get_test(
        "PaGL-PPI"
    )


    esm = get_test(
        "ESM2-GP"
    )


    b0 = get_test(
        "Global branch"
    )


    b2 = get_test(
        "Partner-aware Local"
    )


    external_delta = {}


    for dataset in [

        "Human Full",

        "Human NovelVsTrain",

        "Human NovelVsAll",

        "Yeast Full",

    ]:


        sub = merged_external.loc[
            merged_external[
                "Dataset"
            ]
            ==
            dataset
        ]


        p = float(

            sub.loc[
                sub[
                    "Model"
                ]
                ==
                "PaGL-PPI",

                "AUPRC_mean"

            ].iloc[0]

        )


        e = float(

            sub.loc[
                sub[
                    "Model"
                ]
                ==
                "ESM2-GP",

                "AUPRC_mean"

            ].iloc[0]

        )


        external_delta[
            dataset
        ] = {

            "pagl_auprc":
                p,

            "esm2gp_auprc":
                e,

            "pagl_minus_esm2gp":
                p - e,

        }


    return {

        "generated_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "script_version":
            SCRIPT_VERSION,

        "bernett_test": {

            "Global_branch":
                b0,

            "ESM2_GP":
                esm,

            "Partner_aware_Local":
                b2,

            "PaGL_PPI":
                pagl,

            "PaGL_minus_ESM2GP": {

                "AUPRC":
                    pagl[
                        "auprc"
                    ]
                    -
                    esm[
                        "auprc"
                    ],

                "AUROC":
                    pagl[
                        "auroc"
                    ]
                    -
                    esm[
                        "auroc"
                    ],

                "MCC":
                    pagl[
                        "mcc"
                    ]
                    -
                    esm[
                        "mcc"
                    ],

            },

            "PaGL_minus_B2_AUPRC":
                pagl[
                    "auprc"
                ]
                -
                b2[
                    "auprc"
                ],

            "PaGL_minus_B0_AUPRC":
                pagl[
                    "auprc"
                ]
                -
                b0[
                    "auprc"
                ],

        },

        "dataset_audit": {

            "status":
                audit[
                    "status"
                ],

            "checks_total":
                audit[
                    "checks"
                ][
                    "total"
                ],

            "checks_passed":
                audit[
                    "checks"
                ][
                    "passed"
                ],

            "checks_failed":
                audit[
                    "checks"
                ][
                    "failed"
                ],

            "bernett_protein_disjoint":
                audit[
                    "bernett_protein_disjoint"
                ],

            "yeast_cross_species_zero_shot":
                audit[
                    "yeast"
                ][
                    "cross_species_zero_shot_eligible"
                ],

        },

        "val_test_similarity":
            sim,

        "external_paglp_vs_esm2gp":
            external_delta,

    }


# =============================================================================
# 20. Manuscript result paragraphs
# =============================================================================


def write_result_text(
    numeric,
    merged_external,
):

    b = numeric[
        "bernett_test"
    ]


    sim = numeric[
        "val_test_similarity"
    ]


    audit = numeric[
        "dataset_audit"
    ]


    ext = numeric[
        "external_paglp_vs_esm2gp"
    ]


    text = f"""
【Bernett 主结果】

在 Bernett Intra2 protein-disjoint Test 上，PaGL-PPI 获得
AUPRC={b['PaGL_PPI']['auprc']:.4f}、AUROC={b['PaGL_PPI']['auroc']:.4f}
和 MCC={b['PaGL_PPI']['mcc']:.4f}。相较新增的 ESM2-GP reference
baseline，PaGL-PPI 的 AUPRC 提高
{b['PaGL_minus_ESM2GP']['AUPRC']:+.4f}，AUROC 提高
{b['PaGL_minus_ESM2GP']['AUROC']:+.4f}，MCC 提高
{b['PaGL_minus_ESM2GP']['MCC']:+.4f}。同时，PaGL-PPI 相较
Partner-aware Local 和 Global branch 的 AUPRC 分别提高
{b['PaGL_minus_B2_AUPRC']:+.4f} 和
{b['PaGL_minus_B0_AUPRC']:+.4f}。

【Dataset Audit】

第二轮独立数据审计共执行 {audit['checks_total']} 项检查，
其中 {audit['checks_passed']} 项通过、{audit['checks_failed']} 项失败。
Bernett Train、Validation 和 Test 保持严格 protein-disjoint。
D-SCRIPT Yeast 与 Bernett Train/Validation/Test 的 exact-sequence
overlap 均为 0，因此可作为 cross-species zero-shot benchmark。

【Validation/Test Difference】

Global branch、ESM2-GP、Partner-aware Local 和 PaGL-PPI 均表现出一致的
Validation-to-Test AUPRC 正向迁移。四种模型的平均增幅为
{sim['shift_consistency']['mean_shift']:.4f}，范围为
{sim['shift_consistency']['minimum_shift']:.4f}–
{sim['shift_consistency']['maximum_shift']:.4f}。

冻结 ESM2 表示空间中，positive-negative pair cosine gap 从 Validation 的
{sim['pair_cosine']['validation_gap']:.4f} 增至 Test 的
{sim['pair_cosine']['test_gap']:.4f}，约为
{sim['pair_cosine']['test_to_validation_gap_ratio']:.2f} 倍。
与此同时，nearest-Training protein cosine 从
{sim['nearest_train_cosine']['validation']:.4f} 略降至
{sim['nearest_train_cosine']['test']:.4f}。这些结果支持
split-level pairwise separability difference 的解释，而不能作为严格因果证明。

【External Generalization】

Human Full 上，ESM2-GP 的平均 AUPRC 为
{ext['Human Full']['esm2gp_auprc']:.4f}，PaGL-PPI 为
{ext['Human Full']['pagl_auprc']:.4f}，两者平均性能接近。

在主要的 Human NovelVsTrain strict external cold-start setting 上，
PaGL-PPI 相较 ESM2-GP 的 AUPRC 提高
{ext['Human NovelVsTrain']['pagl_minus_esm2gp']:+.4f}。

在更严格的 Human NovelVsAll exact-sequence novelty stress test 上，
PaGL-PPI 相较 ESM2-GP 的差值为
{ext['Human NovelVsAll']['pagl_minus_esm2gp']:+.4f}，
表明简单 global representation 在该极端条件下仍具有较强鲁棒性。

在 D-SCRIPT Yeast cross-species zero-shot benchmark 上，
PaGL-PPI 相较 ESM2-GP 的 AUPRC 提高
{ext['Yeast Full']['pagl_minus_esm2gp']:+.4f}，
显示 global-local complementarity 在跨物种迁移条件下具有更明显优势。
""".strip()


    path = (
        TEXT_DIR
        /
        "manuscript_results_zh.txt"
    )


    path.write_text(
        text,
        encoding="utf-8"
    )


    print(
        f"[Saved] {path}"
    )


# =============================================================================
# 21. Guardrails
# =============================================================================


def write_guardrails():

    text = """
FINAL MANUSCRIPT CLAIM GUARDRAILS

1. 不应声称 PaGL-PPI 在所有 external subsets 上均取得最佳性能。
   Human Full 的 ESM2-GP 平均 AUPRC 略高；
   Human NovelVsAll 中 ESM2-GP 的平均 AUPRC 明显更高。

2. 可以强调：
   PaGL-PPI 在主要 Human NovelVsTrain strict cold-start setting
   以及 Yeast cross-species zero-shot setting 中表现更优。

3. Table 3 的 literature-reported values 不是严格 controlled comparison。
   不根据 Table 3 单独声称 state-of-the-art。

4. ESM2-GP 是额外补充的 reference baseline。
   不应将其描述为最初 locked Bernett Test protocol 中预先注册的模型。

5. Validation/Test similarity analysis 属于 descriptive evidence。
   可以使用 suggests / supports / is consistent with，
   不应使用 proves / demonstrates causally。

6. Human Full 存在与 Bernett 的 exact-sequence overlap，
   因此只能称为 cross-dataset external evaluation，
   不能称为 strict cold-start。

7. Human NovelVsTrain 是主要 within-species strict external cold-start。
   Human NovelVsAll 是 ultra-strict exact-sequence novelty stress test。

8. Yeast Full 与 Bernett Train/Val/Test exact-sequence overlap 均为 0，
   可称为 cross-species zero-shot evaluation。
""".strip()


    path = (
        TEXT_DIR
        /
        "claim_guardrails.txt"
    )


    path.write_text(
        text,
        encoding="utf-8"
    )


    print(
        f"[Saved] {path}"
    )


# =============================================================================
# 22. Main
# =============================================================================


def legacy_main():

    print("=" * 120)

    print(
        "STEP 23 — FINAL PAPER RESULTS INTEGRATION"
    )

    print("=" * 120)

    print(
        f"Project : {ROOT}"
    )

    print(
        f"Output  : {OUT_ROOT}"
    )


    # =========================================================================
    # A. Required inputs
    # =========================================================================

    required_files = [

        VAL_SUMMARY,

        TEST_SUMMARY,

        ABLATION_SUMMARY,

        ESM2GP_VAL,

        ESM2GP_TEST,

        EXTERNAL_ORIGINAL,

        EXTERNAL_ORIGINAL_PROTOCOL,

        EXTERNAL_ESM2GP,

        EXTERNAL_ESM2GP_PROTOCOL,

        AUDIT_SUMMARY,

        SIM_SUMMARY,

        SIM_MODEL_SHIFT,

    ]


    for p in required_files:

        require(p)


    print(
        "\n[1/8] Required files PASS"
    )


    # =========================================================================
    # B. Protocol gates
    # =========================================================================

    (
        audit,
        sim,
        original_protocol,
        esm2gp_protocol,

    ) = validate_protocols()


    print(
        "[2/8] Protocol gates PASS"
    )


    # =========================================================================
    # C. Load data
    # =========================================================================

    val_df = load_csv(
        VAL_SUMMARY
    )


    test_df = load_csv(
        TEST_SUMMARY
    )


    ablation_df = load_csv(
        ABLATION_SUMMARY
    )


    esm2gp_test = load_esm2gp_test()


    external_original = load_csv(
        EXTERNAL_ORIGINAL
    )


    external_esm2gp = load_csv(
        EXTERNAL_ESM2GP
    )


    merged_external = merge_external(

        external_original,

        external_esm2gp,

    )


    print(
        "[3/8] Result files loaded"
    )


    # =========================================================================
    # D. Table 3
    # =========================================================================

    table03 = build_table03(

        test_df,

        esm2gp_test,

    )


    table03_path = (
        TABLE_DIR
        /
        "table03_literature_comparison.csv"
    )


    save_csv(
        table03,
        table03_path
    )


    # =========================================================================
    # E. Table 4
    # =========================================================================

    table04 = build_table04(

        test_df,

        esm2gp_test,

    )


    table04_path = (
        TABLE_DIR
        /
        "table04_bernett_controlled_test.csv"
    )


    save_csv(
        clean_hidden_columns(
            table04
        ),
        table04_path
    )


    # =========================================================================
    # F. Table 5
    # =========================================================================

    table05 = build_table05(
        ablation_df
    )


    table05_path = (
        TABLE_DIR
        /
        "table05_controlled_ablation.csv"
    )


    save_csv(
        clean_hidden_columns(
            table05
        ),
        table05_path
    )


    # =========================================================================
    # G. Table 6 Human
    # =========================================================================

    table06 = format_external_table(

        merged_external,

        [
            "Human Full",
            "Human NovelVsTrain",
            "Human NovelVsAll",
        ]

    )


    table06_path = (
        TABLE_DIR
        /
        "table06_human_external.csv"
    )


    save_csv(
        clean_hidden_columns(
            table06
        ),
        table06_path
    )


    # =========================================================================
    # H. Table 7 Yeast
    # =========================================================================

    table07 = format_external_table(

        merged_external,

        [
            "Yeast Full"
        ]

    )


    table07_path = (
        TABLE_DIR
        /
        "table07_yeast_external.csv"
    )


    save_csv(
        clean_hidden_columns(
            table07
        ),
        table07_path
    )


    print(
        "[4/8] Table 3–7 generated"
    )


    # =========================================================================
    # I. Dataset audit support
    # =========================================================================

    audit_support = build_audit_support(
        audit
    )


    save_csv(

        audit_support,

        SUPPORT_DIR
        /
        "supporting_dataset_audit.csv"

    )


    # =========================================================================
    # J. Validation/Test support
    # =========================================================================

    sim_shift = load_csv(
        SIM_MODEL_SHIFT
    )


    save_csv(

        sim_shift,

        SUPPORT_DIR
        /
        "supporting_val_test_model_shift.csv"

    )


    sim_evidence = build_similarity_support(
        sim
    )


    save_csv(

        sim_evidence,

        SUPPORT_DIR
        /
        "supporting_val_test_evidence.csv"

    )


    # =========================================================================
    # K. Full external comparison
    # =========================================================================

    save_csv(

        merged_external,

        SUPPORT_DIR
        /
        "external_auprc_comparison.csv"

    )


    print(
        "[5/8] Supporting evidence generated"
    )


    # =========================================================================
    # L. Numeric summary
    # =========================================================================

    numeric = build_numeric_summary(

        audit,

        sim,

        table04,

        merged_external,

    )


    numeric_path = (
        TEXT_DIR
        /
        "manuscript_numeric_summary.json"
    )


    numeric_path.write_text(

        json.dumps(
            numeric,
            indent=2,
            ensure_ascii=False
        ),

        encoding="utf-8"

    )


    print(
        f"[Saved] {numeric_path}"
    )


    print(
        "[6/8] Numeric summary generated"
    )


    # =========================================================================
    # M. Manuscript Markdown
    # =========================================================================

    md = []


    md.append(
        "# Final Paper Tables 3–7\n"
    )


    md.append(
        "## Table 3. Comparison with representative "
        "PPI prediction methods on the Bernett benchmark\n"
    )


    md.append(
        markdown_table(
            table03
        )
    )


    md.append(
        "\n"
        "> Note: Literature-reported values may use different "
        "preprocessing, model-selection procedures, or random-seed "
        "settings and are therefore reference comparisons rather "
        "than strictly controlled head-to-head experiments.\n"
    )


    md.append(
        "## Table 4. Performance of controlled models "
        "on the Bernett protein-disjoint Test set\n"
    )


    md.append(
        table04_markdown(
            table04
        )
    )


    md.append(
        "\n## Table 5. Controlled ablation study "
        "on Bernett Validation\n"
    )


    md.append(
        markdown_table(
            clean_hidden_columns(
                table05
            )
        )
    )


    md.append(
        "\n## Table 6. External generalization "
        "on D-SCRIPT Human\n"
    )


    md.append(
        external_markdown(
            table06
        )
    )


    md.append(
        "\n## Table 7. Cross-species zero-shot "
        "evaluation on D-SCRIPT Yeast\n"
    )


    md.append(
        external_markdown(
            table07
        )
    )


    md.append(
        "\n## Supporting Dataset Audit\n"
    )


    md.append(
        markdown_table(
            audit_support
        )
    )


    md.append(
        "\n## Supporting Validation/Test Analysis\n"
    )


    md.append(
        markdown_table(
            sim_evidence
        )
    )


    markdown_path = (
        TEXT_DIR
        /
        "table03_07_final.md"
    )


    markdown_path.write_text(

        "\n\n".join(
            md
        ),

        encoding="utf-8"

    )


    print(
        f"[Saved] {markdown_path}"
    )


    print(
        "[7/8] Final manuscript tables generated"
    )


    # =========================================================================
    # N. Text and guardrails
    # =========================================================================

    write_result_text(

        numeric,

        merged_external,

    )


    write_guardrails()


    print(
        "[8/8] Manuscript text generated"
    )


    # =========================================================================
    # Final console summary
    # =========================================================================

    print()
    print("=" * 120)

    print(
        "FINAL PAPER RESULTS"
    )

    print("=" * 120)


    print()
    print(
        "[Bernett Test]"
    )


    print(

        clean_hidden_columns(
            table04
        ).to_string(
            index=False
        )

    )


    print()
    print(
        "[Human External AUPRC]"
    )


    print(

        table06[
            [
                "Setting",
                "Model",
                "Random AUPRC",
                "AUPRC",
                "AUPRC / Random",
            ]
        ].to_string(
            index=False
        )

    )


    print()
    print(
        "[Yeast External AUPRC]"
    )


    print(

        table07[
            [
                "Setting",
                "Model",
                "Random AUPRC",
                "AUPRC",
                "AUPRC / Random",
            ]
        ].to_string(
            index=False
        )

    )


    print()
    print("-" * 120)

    print(
        f"Dataset audit : "
        f"{audit['checks']['passed']}/"
        f"{audit['checks']['total']} PASS"
    )


    print(
        f"Val→Test mean AUPRC shift : "
        f"{sim['shift_consistency']['mean_shift']:+.6f}"
    )


    print(
        f"Pair cosine gap ratio     : "
        f"{sim['pair_cosine']['test_to_validation_gap_ratio']:.3f}×"
    )


    print(
        f"Nearest-Train Test−Val    : "
        f"{sim['nearest_train_cosine']['test_minus_validation']:+.6f}"
    )


    print()
    print("=" * 120)

    print(
        "STEP 23 COMPLETED SUCCESSFULLY"
    )

    print("=" * 120)


    print()
    print(
        "Final outputs:"
    )


    print(
        f"  {TABLE_DIR}"
    )


    print(
        f"  {SUPPORT_DIR}"
    )


    print(
        f"  {TEXT_DIR}"
    )


# =============================================================================
# 23. Entry
# =============================================================================


def main():
    """Generate manuscript Tables 3–7 from freshly calculated result summaries.

    This five-CSV release does not include the original diagnostic result archive.
    Audit and validation/test similarity analyses are therefore outside this
    compact table command; the model and external metric calculations remain
    in the preceding numbered scripts.
    """
    for path in (
        TEST_SUMMARY,
        ABLATION_SUMMARY,
        ESM2GP_TEST,
        EXTERNAL_ORIGINAL,
        EXTERNAL_ESM2GP,
    ):
        require(path)

    test_df = load_csv(TEST_SUMMARY)
    ablation_df = load_csv(ABLATION_SUMMARY)
    esm2gp_test = load_esm2gp_test()
    external = merge_external(load_csv(EXTERNAL_ORIGINAL), load_csv(EXTERNAL_ESM2GP))

    table03 = build_table03(test_df, esm2gp_test)
    table04 = build_table04(test_df, esm2gp_test)
    table05 = build_table05(ablation_df)
    table06 = format_external_table(
        external, ["Human Full", "Human NovelVsTrain", "Human NovelVsAll"]
    )
    table07 = format_external_table(external, ["Yeast Full"])

    outputs = {
        "table03_literature_comparison.csv": table03,
        "table04_bernett_controlled_test.csv": clean_hidden_columns(table04),
        "table05_controlled_ablation.csv": clean_hidden_columns(table05),
        "table06_human_external.csv": clean_hidden_columns(table06),
        "table07_yeast_external.csv": clean_hidden_columns(table07),
    }
    for name, frame in outputs.items():
        save_csv(frame, TABLE_DIR / name)

    markdown = [
        "# Recomputed manuscript tables",
        "## Table 3. Literature comparison\n" + markdown_table(table03),
        "## Table 4. Controlled Bernett test\n" + table04_markdown(table04),
        "## Table 5. Bernett validation ablations\n" + markdown_table(clean_hidden_columns(table05)),
        "## Table 6. D-SCRIPT Human\n" + external_markdown(table06),
        "## Table 7. D-SCRIPT Yeast\n" + external_markdown(table07),
    ]
    (TEXT_DIR / "table03_07_final.md").write_text("\n\n".join(markdown), encoding="utf-8")
    print("Generated Tables 3–7 from five-seed model summaries:", TABLE_DIR)


if __name__ == "__main__":

    main()
