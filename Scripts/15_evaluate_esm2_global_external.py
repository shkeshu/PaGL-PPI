# -*- coding: utf-8 -*-

"""
15_evaluate_esm2_global_external.py

================================================================================
STEP 20 — ESM2-GP EXTERNAL GENERALIZATION EVALUATION
================================================================================

目的
----
使用 Step 17 已经训练并冻结的 ESM2-GP checkpoint，
在 D-SCRIPT Human / Yeast external datasets 上进行 inference-only 评估。

重要原则
--------
1. 不重新训练。
2. 不重新选择 checkpoint。
3. 不根据 external labels 调整模型结构或超参数。
4. threshold 固定为 0.5。
5. 模型结构直接复用 13_train_esm2_global_baseline.py。
6. 使用 Step 14 已经生成的 external global ESM2 embedding。
7. Step 01 data audit 必须为 PASS。
8. 五随机种子统计使用 sample standard deviation (ddof=1)。

Evaluation settings
-------------------
Human Full
    Cross-dataset external evaluation

Human NovelVsTrain
    Strict within-species external cold-start

Human NovelVsAll
    Ultra-strict exact-sequence novelty stress test

Yeast Full
    Cross-species zero-shot evaluation

Expected external embedding
---------------------------
Data/
    embeddings_external/
        D-SCRIPT_Human/
            global/
                *.pt

        D-SCRIPT_Yeast/
            global/
                *.pt

Expected canonical data
-----------------------
Data/
    external_canonical/

        D-SCRIPT_Human/
            test_pairs.csv
            test_pairs_novel_vs_bernett_train.csv
            test_pairs_novel_vs_bernett_all.csv

        D-SCRIPT_Yeast/
            test_pairs.csv

Checkpoint
----------
Results/
    esm2_global_baseline/
        best_seed_42.pt
        best_seed_123.pt
        best_seed_456.pt
        best_seed_789.pt
        best_seed_1234.pt

Output
------
Results/
    external_generalization/

        esm2_gp_external_seed_metrics.csv
        esm2_gp_external_summary.csv

        esm2_gp_predictions_human_full.csv
        esm2_gp_predictions_human_novel_train.csv
        esm2_gp_predictions_human_novel_all.csv
        esm2_gp_predictions_yeast_full.csv

        esm2_gp_external_protocol.json

================================================================================
"""


from __future__ import annotations


import hashlib
import importlib.util
import json
import platform
import sys
import time

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple


import numpy as np
import pandas as pd


import torch

from torch.utils.data import (
    Dataset,
    DataLoader,
)


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
# 1. Version
# =============================================================================

SCRIPT_VERSION = "2.0"


# =============================================================================
# 2. Project
# =============================================================================

SCRIPT_DIR = Path(
    __file__
).resolve().parent


ROOT = (
    SCRIPT_DIR
    .parent
)


DATA_DIR = (
    ROOT
    /
    "Data"
)


RESULT_DIR = (
    ROOT
    /
    "Results"
)


# =============================================================================
# 3. Step 17
# =============================================================================

STEP19_SCRIPT = (
    SCRIPT_DIR
    /
    "13_train_esm2_global_baseline.py"
)


CHECKPOINT_DIR = (
    RESULT_DIR
    /
    "esm2_global_baseline"
)


SEEDS = [
    42,
    123,
    456,
    789,
    1234,
]


# =============================================================================
# 4. Step 15 audit gate
# =============================================================================

STEP18_AUDIT = (
    RESULT_DIR
    /
    "second_round"
    /
    "dataset_audit"
    /
    "dataset_audit_summary.json"
)


# =============================================================================
# 5. External canonical data
# =============================================================================

HUMAN_CANONICAL_DIR = (
    DATA_DIR
    /
    "external_canonical"
    /
    "D-SCRIPT_Human"
)


YEAST_CANONICAL_DIR = (
    DATA_DIR
    /
    "external_canonical"
    /
    "D-SCRIPT_Yeast"
)


DATASETS = {

    "Human_Full": (
        HUMAN_CANONICAL_DIR
        /
        "test_pairs.csv"
    ),

    "Human_NovelVsTrain": (
        HUMAN_CANONICAL_DIR
        /
        "test_pairs_novel_vs_bernett_train.csv"
    ),

    "Human_NovelVsAll": (
        HUMAN_CANONICAL_DIR
        /
        "test_pairs_novel_vs_bernett_all.csv"
    ),

    "Yeast_Full": (
        YEAST_CANONICAL_DIR
        /
        "test_pairs.csv"
    ),
}


# =============================================================================
# 6. External embeddings
# =============================================================================

EXTERNAL_EMBEDDING_ROOT = (
    DATA_DIR
    /
    "embeddings_external"
)


HUMAN_EMBEDDING_ROOT = (
    EXTERNAL_EMBEDDING_ROOT
    /
    "D-SCRIPT_Human"
)


YEAST_EMBEDDING_ROOT = (
    EXTERNAL_EMBEDDING_ROOT
    /
    "D-SCRIPT_Yeast"
)


HUMAN_GLOBAL_DIR = (
    HUMAN_EMBEDDING_ROOT
    /
    "global"
)


YEAST_GLOBAL_DIR = (
    YEAST_EMBEDDING_ROOT
    /
    "global"
)


# =============================================================================
# 7. Output
# =============================================================================

OUTPUT_DIR = (
    RESULT_DIR
    /
    "external_generalization"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


SEED_METRICS_PATH = (
    OUTPUT_DIR
    /
    "esm2_gp_external_seed_metrics.csv"
)


SUMMARY_PATH = (
    OUTPUT_DIR
    /
    "esm2_gp_external_summary.csv"
)


PROTOCOL_PATH = (
    OUTPUT_DIR
    /
    "esm2_gp_external_protocol.json"
)


# =============================================================================
# 8. Evaluation config
# =============================================================================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


THRESHOLD = 0.50


# inference batch size
# 不影响模型结构或预测结果
EVAL_BATCH_SIZE = 512


EXPECTED_EMBEDDING_DIM = 1280


# =============================================================================
# 9. Expected dataset statistics
#
# 这里只用于安全核验。
# 实际指标仍从真实 CSV 重新统计。
# =============================================================================

EXPECTED_DATASET_STATS = {

    "Human_Full": {

        "pairs":
            52723,

        "positive":
            4794,

        "negative":
            47929,

    },


    "Human_NovelVsTrain": {

        "pairs":
            36845,

        "positive":
            2656,

        "negative":
            34189,

    },


    "Human_NovelVsAll": {

        "pairs":
            13957,

        "positive":
            653,

        "negative":
            13304,

    },


    "Yeast_Full": {

        "pairs":
            54964,

        "positive":
            4996,

        "negative":
            49968,

    },
}


# =============================================================================
# 10. General helpers
# =============================================================================


def require_file(
    path: Path
):

    if not path.exists():

        raise FileNotFoundError(

            f"\nMissing required file:\n"
            f"{path}\n"

        )



def require_dir(
    path: Path
):

    if not path.is_dir():

        raise FileNotFoundError(

            f"\nMissing required directory:\n"
            f"{path}\n"

        )



def load_json(
    path: Path
):

    require_file(
        path
    )


    with path.open(
        "r",
        encoding="utf-8-sig"
    ) as f:

        return json.load(
            f
        )



def sha256_file(
    path: Path
):

    h = hashlib.sha256()


    with path.open(
        "rb"
    ) as f:


        for block in iter(

            lambda: f.read(
                1024 * 1024
            ),

            b""

        ):

            h.update(
                block
            )


    return h.hexdigest()



def safe_torch_load(
    path: Path,
    map_location="cpu",
):

    """
    优先使用 weights_only=True。

    对较旧 PyTorch 或特殊 checkpoint 格式自动回退。
    """

    try:

        return torch.load(

            path,

            map_location=map_location,

            weights_only=True,

        )


    except TypeError:

        return torch.load(

            path,

            map_location=map_location,

        )


    except Exception:

        return torch.load(

            path,

            map_location=map_location,

        )


# =============================================================================
# 11. Load Step 17 model
# =============================================================================


def load_step19_module():

    require_file(
        STEP19_SCRIPT
    )


    spec = importlib.util.spec_from_file_location(

        "step19_esm2_gp",

        STEP19_SCRIPT,

    )


    if spec is None:

        raise RuntimeError(
            "Cannot create import spec for Step 17."
        )


    module = importlib.util.module_from_spec(
        spec
    )


    if spec.loader is None:

        raise RuntimeError(
            "Step 17 module loader is None."
        )


    spec.loader.exec_module(
        module
    )


    if not hasattr(
        module,
        "ESM2GlobalBaseline"
    ):

        raise RuntimeError(

            "Step 17 does not contain "
            "ESM2GlobalBaseline."

        )


    return module


# =============================================================================
# 12. Step 15 gate
# =============================================================================


def validate_prepared_data():

    report = load_json(
        STEP18_AUDIT
    )


    status = str(

        report.get(
            "status",
            ""
        )

    ).upper()


    if status != "PASS":

        raise RuntimeError(

            "Step 01 data audit is not PASS.\n"
            f"Current status = {status}"

        )


    checks = report.get(
        "checks",
        {}
    )


    failed = int(

        checks.get(
            "failed",
            -1
        )

    )


    if failed != 0:

        raise RuntimeError(

            "Step 01 data audit contains failed checks.\n"
            f"failed = {failed}"

        )


    if not bool(

        report.get(
            "bernett_protein_disjoint",
            False
        )

    ):

        raise RuntimeError(

            "Bernett protein-disjoint status "
            "is not PASS."

        )


    yeast = report.get(
        "yeast",
        {}
    )


    if not bool(

        yeast.get(
            "cross_species_zero_shot_eligible",
            False
        )

    ):

        raise RuntimeError(

            "Yeast cross-species zero-shot "
            "eligibility is not PASS."

        )


    print(
        "[PASS] Step 01 data audit"
    )


# =============================================================================
# 13. Read canonical pair CSV
# =============================================================================


def load_pair_csv(
    path: Path,
    dataset_name: str,
):

    require_file(
        path
    )


    df = pd.read_csv(
        path
    )


    required = {

        "protein_a",

        "protein_b",

        "label",

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

            f"{dataset_name} missing columns:\n"
            f"{sorted(missing)}\n"
            f"Available: {list(df.columns)}"

        )


    df = df.copy()


    df[
        "protein_a"
    ] = (

        df[
            "protein_a"
        ]

        .astype(str)

        .str.strip()

    )


    df[
        "protein_b"
    ] = (

        df[
            "protein_b"
        ]

        .astype(str)

        .str.strip()

    )


    df[
        "label"
    ] = pd.to_numeric(

        df[
            "label"
        ],

        errors="raise",

    ).astype(
        np.int64
    )


    invalid = (

        ~df[
            "label"
        ]

        .isin(
            [
                0,
                1,
            ]
        )

    )


    if invalid.any():

        raise RuntimeError(

            f"{dataset_name}: invalid labels detected."

        )


    # -------------------------------------------------------------------------
    # dataset count contract
    # -------------------------------------------------------------------------

    expected = EXPECTED_DATASET_STATS[
        dataset_name
    ]


    observed_pairs = int(
        len(df)
    )


    observed_pos = int(

        (
            df[
                "label"
            ]
            ==
            1
        ).sum()

    )


    observed_neg = int(

        (
            df[
                "label"
            ]
            ==
            0
        ).sum()

    )


    if observed_pairs != expected[
        "pairs"
    ]:

        raise RuntimeError(

            f"{dataset_name} pair count mismatch:\n"
            f"Observed = {observed_pairs}\n"
            f"Expected = {expected['pairs']}"

        )


    if observed_pos != expected[
        "positive"
    ]:

        raise RuntimeError(

            f"{dataset_name} positive count mismatch:\n"
            f"Observed = {observed_pos}\n"
            f"Expected = {expected['positive']}"

        )


    if observed_neg != expected[
        "negative"
    ]:

        raise RuntimeError(

            f"{dataset_name} negative count mismatch:\n"
            f"Observed = {observed_neg}\n"
            f"Expected = {expected['negative']}"

        )


    print(
        f"[PASS] {dataset_name:<24s} "
        f"N={observed_pairs:,} | "
        f"Pos={observed_pos:,} | "
        f"Neg={observed_neg:,}"
    )


    return df


# =============================================================================
# 14. External embedding resolver
# =============================================================================


class EmbeddingResolver:

    """
    External global embedding resolver.

    Resolve priority
    ----------------
    1. direct: global/<protein_id>.pt

    2. embedding_manifest.csv:
       supports:
           protein_id
           external_protein_id
           original_protein_id
           id

       and:
           embedding_file
           embedding_filename
           global_file
           global_filename
           global_path
           relative_path
           path
           filename

    3. automatically detect ID/path columns from manifest

    4. index protein_id stored inside .pt payload

    5. common filename normalization

    Required output
    ---------------
    protein-level ESM2 embedding:
        [1280]
    """


    def __init__(
        self,
        embedding_root: Path,
        global_dir: Path,
        species: str,
    ):

        self.embedding_root = embedding_root

        self.global_dir = global_dir

        self.species = species


        require_dir(
            self.embedding_root
        )

        require_dir(
            self.global_dir
        )


        # protein_id -> path
        self.manifest_map = {}


        # exact filename stem -> path
        self.global_stem_map = {}


        # lower-case stem -> path
        self.global_stem_lower_map = {}


        # protein_id stored inside payload -> path
        self.payload_id_map = {}


        # all files
        self.global_files = []


        self._index_global_files()

        self._load_manifest()


        print(
            f"[Embedding resolver] {self.species}: "
            f"manifest mappings={len(self.manifest_map):,}, "
            f"files={len(self.global_files):,}"
        )


    # =========================================================================
    # A. Index actual .pt files
    # =========================================================================

    def _index_global_files(self):

        files = sorted(
            self.global_dir.glob(
                "*.pt"
            )
        )


        if not files:

            files = sorted(
                self.global_dir.rglob(
                    "*.pt"
                )
            )


        if not files:

            raise RuntimeError(
                f"No .pt files found:\n"
                f"{self.global_dir}"
            )


        self.global_files = files


        for path in files:

            stem = path.stem


            if stem in self.global_stem_map:

                raise RuntimeError(
                    f"Duplicate embedding stem: {stem}"
                )


            self.global_stem_map[
                stem
            ] = path


            self.global_stem_lower_map[
                stem.lower()
            ] = path


        print(
            f"[Embedding index] "
            f"{self.species}: "
            f"{len(files):,} global files"
        )


        print(
            f"  Example files:"
        )


        for path in files[:5]:

            print(
                f"    {path.name}"
            )


    # =========================================================================
    # B. Convert manifest value to an existing file
    # =========================================================================

    def _resolve_manifest_path(
        self,
        raw_value,
    ):

        if pd.isna(
            raw_value
        ):

            return None


        raw = str(
            raw_value
        ).strip()


        if (
            not raw
            or
            raw.lower() == "nan"
        ):

            return None


        p = Path(
            raw
        )


        candidates = []


        # absolute path
        if p.is_absolute():

            candidates.append(
                p
            )


        else:

            # relative to embedding root
            candidates.append(

                self.embedding_root
                /
                p

            )


            # relative to global folder
            candidates.append(

                self.global_dir
                /
                p

            )


            # only filename
            candidates.append(

                self.global_dir
                /
                p.name

            )


            # value has no .pt extension
            if p.suffix.lower() != ".pt":

                candidates.append(

                    self.global_dir
                    /
                    f"{p.name}.pt"

                )


        for candidate in candidates:

            if candidate.exists():

                return candidate.resolve()


        # try matching filename / stem
        name = p.name


        if name.endswith(
            ".pt"
        ):

            stem = Path(
                name
            ).stem

        else:

            stem = name


        if stem in self.global_stem_map:

            return self.global_stem_map[
                stem
            ]


        if stem.lower() in self.global_stem_lower_map:

            return self.global_stem_lower_map[
                stem.lower()
            ]


        return None


    # =========================================================================
    # C. Manifest parsing
    # =========================================================================

    def _load_manifest(self):

        manifest_path = (

            self.embedding_root
            /
            "embedding_manifest.csv"

        )


        if not manifest_path.exists():

            print(
                f"[Info] {self.species}: "
                f"embedding_manifest.csv not found."
            )

            return


        manifest = pd.read_csv(
            manifest_path,
            low_memory=False,
        )


        print(
            f"[Manifest] {self.species}: "
            f"{manifest_path}"
        )


        print(
            f"  columns = "
            f"{list(manifest.columns)}"
        )


        # ---------------------------------------------------------------------
        # explicit candidates
        # ---------------------------------------------------------------------

        id_candidates = [

            "protein_id",

            "external_protein_id",

            "original_protein_id",

            "protein",

            "protein_name",

            "id",

            "name",

        ]


        path_candidates = [

            "embedding_file",

            "embedding_filename",

            "global_embedding_file",

            "global_embedding_filename",

            "global_file",

            "global_filename",

            "global_path",

            "global_embedding_path",

            "relative_path",

            "relative_file",

            "file",

            "filename",

            "file_name",

            "path",

        ]


        id_col = next(

            (
                c
                for c in id_candidates
                if c in manifest.columns
            ),

            None

        )


        path_col = next(

            (
                c
                for c in path_candidates
                if c in manifest.columns
            ),

            None

        )


        # ---------------------------------------------------------------------
        # automatic ID-column detection
        # ---------------------------------------------------------------------

        if id_col is None:

            for column in manifest.columns:

                low = str(
                    column
                ).lower()


                if (
                    "protein" in low
                    and
                    (
                        "id" in low
                        or
                        "name" in low
                    )
                ):

                    id_col = column

                    break


        # ---------------------------------------------------------------------
        # automatic path-column detection
        # ---------------------------------------------------------------------

        if path_col is None:

            # first preference:
            # column name contains global + file/path/name
            for column in manifest.columns:

                low = str(
                    column
                ).lower()


                if (
                    "global" in low
                    and
                    (
                        "file" in low
                        or
                        "path" in low
                        or
                        "name" in low
                    )
                ):

                    path_col = column

                    break


        if path_col is None:

            # second preference:
            # embedding + file/path/name
            for column in manifest.columns:

                low = str(
                    column
                ).lower()


                if (
                    "embedding" in low
                    and
                    (
                        "file" in low
                        or
                        "path" in low
                        or
                        "name" in low
                    )
                ):

                    path_col = column

                    break


        if path_col is None:

            # final generic path/file search
            for column in manifest.columns:

                low = str(
                    column
                ).lower()


                if (
                    "file" in low
                    or
                    "path" in low
                    or
                    "filename" in low
                ):

                    path_col = column

                    break


        print(
            f"  detected id column   = {id_col}"
        )

        print(
            f"  detected path column = {path_col}"
        )


        if id_col is None:

            print(
                f"[Warning] {self.species}: "
                f"cannot identify protein-ID column "
                f"from manifest."
            )

            return


        # ---------------------------------------------------------------------
        # Case 1:
        # manifest has explicit path column
        # ---------------------------------------------------------------------

        mapped = 0


        if path_col is not None:

            for _, row in manifest.iterrows():


                raw_pid = row[
                    id_col
                ]


                if pd.isna(
                    raw_pid
                ):

                    continue


                pid = str(
                    raw_pid
                ).strip()


                if not pid:

                    continue


                path = self._resolve_manifest_path(

                    row[
                        path_col
                    ]

                )


                if path is not None:

                    self.manifest_map[
                        pid
                    ] = path

                    mapped += 1


            print(
                f"  manifest path mappings = "
                f"{mapped:,}"
            )


        # ---------------------------------------------------------------------
        # Case 2:
        # manifest ID exists but explicit path field is unusable.
        #
        # Try other columns row-by-row.
        # ---------------------------------------------------------------------

        if mapped == 0:

            print(
                f"[Info] {self.species}: "
                f"explicit path column produced no valid mappings; "
                f"trying all manifest columns."
            )


            candidate_columns = [

                c

                for c in manifest.columns

                if c != id_col

            ]


            for _, row in manifest.iterrows():


                raw_pid = row[
                    id_col
                ]


                if pd.isna(
                    raw_pid
                ):

                    continue


                pid = str(
                    raw_pid
                ).strip()


                if not pid:

                    continue


                found = None


                for column in candidate_columns:

                    found = self._resolve_manifest_path(

                        row[
                            column
                        ]

                    )


                    if found is not None:

                        break


                if found is not None:

                    self.manifest_map[
                        pid
                    ] = found


            print(
                f"  fallback manifest mappings = "
                f"{len(self.manifest_map):,}"
            )


    # =========================================================================
    # D. Common filename variants
    # =========================================================================

    def _filename_variants(
        self,
        pid: str,
    ):

        pid = str(
            pid
        ).strip()


        variants = [

            pid,

            pid.replace(
                ".",
                "_"
            ),

            pid.replace(
                ".",
                "-"
            ),

            pid.replace(
                ":",
                "_"
            ),

            pid.replace(
                "/",
                "_"
            ),

            pid.replace(
                "|",
                "_"
            ),

        ]


        # D-SCRIPT IDs can contain taxonomy prefix:
        #
        # 9606.ENSP00000000233
        #
        # also try:
        #
        # ENSP00000000233
        #
        if "." in pid:

            suffix = pid.split(
                ".",
                1
            )[1]


            variants.extend(
                [
                    suffix,
                    suffix.replace(
                        ".",
                        "_"
                    ),
                ]
            )


        # unique preserve order
        output = []


        seen = set()


        for item in variants:

            if (
                item
                and
                item not in seen
            ):

                seen.add(
                    item
                )

                output.append(
                    item
                )


        return output


    # =========================================================================
    # E. Build index from payload metadata
    # =========================================================================

    def _build_payload_id_index(self):

        if self.payload_id_map:

            return


        print(
            f"[Fallback] {self.species}: "
            f"scanning .pt payload metadata for protein_id ..."
        )


        found = 0


        for i, path in enumerate(
            self.global_files,
            start=1,
        ):


            try:

                obj = safe_torch_load(

                    path,

                    map_location="cpu",

                )


            except Exception:

                continue


            if isinstance(
                obj,
                dict
            ):


                pid = None


                for key in [

                    "protein_id",

                    "external_protein_id",

                    "original_protein_id",

                    "id",

                    "protein",

                ]:


                    if (
                        key in obj
                        and
                        obj[
                            key
                        ]
                        is not None
                    ):

                        pid = str(
                            obj[
                                key
                            ]
                        ).strip()

                        break


                if pid:

                    self.payload_id_map[
                        pid
                    ] = path

                    found += 1


            if (
                i == 1
                or
                i % 2000 == 0
                or
                i == len(
                    self.global_files
                )
            ):

                print(
                    f"  metadata scan "
                    f"{i:,}/{len(self.global_files):,}"
                )


        print(
            f"  payload protein_id mappings = "
            f"{found:,}"
        )


    # =========================================================================
    # F. Resolve protein ID
    # =========================================================================

    def resolve(
        self,
        pid: str
    ) -> Path:


        pid = str(
            pid
        ).strip()


        # ---------------------------------------------------------------------
        # 1. Direct exact file
        # ---------------------------------------------------------------------

        direct = (

            self.global_dir
            /
            f"{pid}.pt"

        )


        if direct.exists():

            return direct


        # ---------------------------------------------------------------------
        # 2. Manifest exact mapping
        # ---------------------------------------------------------------------

        if pid in self.manifest_map:

            return self.manifest_map[
                pid
            ]


        # ---------------------------------------------------------------------
        # 3. Filename variants
        # ---------------------------------------------------------------------

        for variant in self._filename_variants(
            pid
        ):


            if variant in self.global_stem_map:

                return self.global_stem_map[
                    variant
                ]


            lower = variant.lower()


            if lower in self.global_stem_lower_map:

                return self.global_stem_lower_map[
                    lower
                ]


        # ---------------------------------------------------------------------
        # 4. Payload metadata
        # ---------------------------------------------------------------------

        if not self.payload_id_map:

            self._build_payload_id_index()


        if pid in self.payload_id_map:

            return self.payload_id_map[
                pid
            ]


        # payload ID variants
        for variant in self._filename_variants(
            pid
        ):


            if variant in self.payload_id_map:

                return self.payload_id_map[
                    variant
                ]


        # ---------------------------------------------------------------------
        # Fail with useful diagnostics
        # ---------------------------------------------------------------------

        examples = [

            p.name

            for p in self.global_files[
                :10
            ]

        ]


        raise FileNotFoundError(

            f"\nCannot resolve {self.species} "
            f"global embedding.\n\n"

            f"Protein ID:\n"
            f"  {pid}\n\n"

            f"Direct path checked:\n"
            f"  {direct}\n\n"

            f"Manifest mappings:\n"
            f"  {len(self.manifest_map):,}\n\n"

            f"Payload-ID mappings:\n"
            f"  {len(self.payload_id_map):,}\n\n"

            f"Example actual files:\n  "
            +
            "\n  ".join(
                examples
            )

        )


    # =========================================================================
    # G. Load embedding
    # =========================================================================

    def load(
        self,
        pid: str
    ) -> torch.Tensor:


        path = self.resolve(
            pid
        )


        obj = safe_torch_load(

            path,

            map_location="cpu",

        )


        # ---------------------------------------------------------------------
        # Tensor directly
        # ---------------------------------------------------------------------

        if isinstance(
            obj,
            torch.Tensor
        ):

            emb = obj


        # ---------------------------------------------------------------------
        # Dict payload
        # ---------------------------------------------------------------------

        elif isinstance(
            obj,
            dict
        ):


            emb = None


            possible_keys = [

                "global_embedding",

                "embedding",

                "protein_embedding",

                "esm2_embedding",

                "representation",

                "representations",

                "mean_embedding",

            ]


            for key in possible_keys:


                if key not in obj:

                    continue


                value = obj[
                    key
                ]


                if isinstance(
                    value,
                    torch.Tensor
                ):

                    emb = value

                    break


            if emb is None:

                tensor_items = [

                    value

                    for value in obj.values()

                    if isinstance(
                        value,
                        torch.Tensor
                    )

                ]


                # only one tensor -> safe fallback
                if len(
                    tensor_items
                ) == 1:

                    emb = tensor_items[
                        0
                    ]


            if emb is None:

                raise RuntimeError(

                    f"Cannot identify embedding tensor:\n"
                    f"{path}\n"
                    f"Keys = {list(obj.keys())}"

                )


        else:

            raise RuntimeError(

                f"Unsupported embedding payload:\n"
                f"{path}\n"
                f"type={type(obj)}"

            )


        emb = (

            emb

            .detach()

            .cpu()

            .float()

        )


        # ---------------------------------------------------------------------
        # [1,1280] -> [1280]
        # ---------------------------------------------------------------------

        if (
            emb.ndim == 2
            and
            emb.shape
            ==
            (
                1,
                EXPECTED_EMBEDDING_DIM
            )
        ):

            emb = emb.squeeze(
                0
            )


        # ---------------------------------------------------------------------
        # If accidentally stored residue/segment matrix,
        # DO NOT silently mean pool.
        #
        # Step16 expects the exact protein-level global vector.
        # ---------------------------------------------------------------------

        if emb.ndim != 1:

            raise RuntimeError(

                f"{pid}: expected protein-level "
                f"[1280] global embedding, "
                f"got {tuple(emb.shape)}.\n"
                f"Path: {path}\n\n"
                f"Do not automatically mean-pool here, "
                f"because Step 20 must use the same "
                f"representation contract as Step 17."

            )


        if (
            emb.shape[
                0
            ]
            !=
            EXPECTED_EMBEDDING_DIM
        ):

            raise RuntimeError(

                f"{pid}: expected "
                f"{EXPECTED_EMBEDDING_DIM} dimensions, "
                f"got {tuple(emb.shape)}.\n"
                f"Path: {path}"

            )


        if not torch.isfinite(
            emb
        ).all():

            raise RuntimeError(

                f"{pid}: embedding contains NaN/Inf.\n"
                f"{path}"

            )


        return emb

# =============================================================================
# 15. Preload unique embeddings
# =============================================================================


def preload_embeddings(

    pair_df: pd.DataFrame,

    resolver: EmbeddingResolver,

    dataset_name: str,

) -> Dict[str, torch.Tensor]:


    proteins = sorted(

        set(
            pair_df[
                "protein_a"
            ]
        )

        |

        set(
            pair_df[
                "protein_b"
            ]
        )

    )


    print()

    print(
        f"[Preload] {dataset_name}: "
        f"{len(proteins):,} unique proteins"
    )


    cache = {}


    start = time.time()


    for i, pid in enumerate(
        proteins,
        start=1,
    ):


        cache[
            pid
        ] = resolver.load(
            pid
        )


        if (
            i == 1
            or
            i % 1000 == 0
            or
            i == len(proteins)
        ):

            print(

                f"  {i:6d}"
                f"/"
                f"{len(proteins):6d}"

            )


    elapsed = time.time() - start


    print(

        f"[PASS] {dataset_name}: "
        f"loaded {len(cache):,} embeddings "
        f"in {elapsed:.1f}s"

    )


    return cache


# =============================================================================
# 16. Dataset
# =============================================================================


class ExternalPairDataset(
    Dataset
):


    def __init__(

        self,

        df: pd.DataFrame,

        embedding_cache:
            Dict[
                str,
                torch.Tensor
            ],

    ):


        self.df = (
            df
            .reset_index(
                drop=True
            )
            .copy()
        )


        self.embedding_cache = (
            embedding_cache
        )


        self.protein_a = (

            self.df[
                "protein_a"
            ]
            .astype(str)
            .tolist()

        )


        self.protein_b = (

            self.df[
                "protein_b"
            ]
            .astype(str)
            .tolist()

        )


        self.labels = (

            self.df[
                "label"
            ]
            .astype(
                np.float32
            )
            .to_numpy()

        )



    def __len__(
        self
    ):


        return len(
            self.df
        )



    def __getitem__(
        self,
        index
    ):


        pa = self.protein_a[
            index
        ]


        pb = self.protein_b[
            index
        ]


        ea = self.embedding_cache[
            pa
        ]


        eb = self.embedding_cache[
            pb
        ]


        y = torch.tensor(

            self.labels[
                index
            ],

            dtype=torch.float32,

        )


        return (
            ea,
            eb,
            y,
        )


# =============================================================================
# 17. Metrics
# =============================================================================


def calculate_metrics(

    labels: np.ndarray,

    probs: np.ndarray,

    threshold: float,

):


    labels = np.asarray(
        labels,
        dtype=np.int64,
    )


    probs = np.asarray(
        probs,
        dtype=np.float64,
    )


    preds = (

        probs
        >=
        threshold

    ).astype(
        np.int64
    )


    tn, fp, fn, tp = confusion_matrix(

        labels,

        preds,

        labels=[
            0,
            1,
        ],

    ).ravel()


    specificity = (

        tn
        /
        (
            tn
            +
            fp
        )

        if (
            tn
            +
            fp
        )
        >
        0

        else 0.0

    )


    prevalence = float(

        np.mean(
            labels
            ==
            1
        )

    )


    auprc = float(

        average_precision_score(
            labels,
            probs
        )

    )


    metrics = {

        "n":
            int(
                len(
                    labels
                )
            ),

        "positive":
            int(
                np.sum(
                    labels
                    ==
                    1
                )
            ),

        "negative":
            int(
                np.sum(
                    labels
                    ==
                    0
                )
            ),

        "positive_ratio":
            prevalence,

        "random_auprc":
            prevalence,

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

                    zero_division=0,

                )

            ),

        "recall":
            float(

                recall_score(

                    labels,

                    preds,

                    zero_division=0,

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

                    zero_division=0,

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
            auprc,

        "auprc_over_random":
            float(

                auprc
                /
                prevalence

            ),

        "tp":
            int(
                tp
            ),

        "tn":
            int(
                tn
            ),

        "fp":
            int(
                fp
            ),

        "fn":
            int(
                fn
            ),

    }


    return (
        metrics,
        preds,
    )


# =============================================================================
# 18. Load checkpoint
# =============================================================================


def load_model_for_seed(

    seed: int,

    model_class,

):


    checkpoint_path = (

        CHECKPOINT_DIR
        /
        f"best_seed_{seed}.pt"

    )


    require_file(
        checkpoint_path
    )


    checkpoint = safe_torch_load(

        checkpoint_path,

        map_location=DEVICE,

    )


    if not isinstance(
        checkpoint,
        dict
    ):

        raise RuntimeError(

            f"Unexpected checkpoint type:\n"
            f"{checkpoint_path}"

        )


    if "model" not in checkpoint:

        raise RuntimeError(

            f"Checkpoint missing 'model':\n"
            f"{checkpoint_path}"

        )


    model = model_class()


    model.load_state_dict(

        checkpoint[
            "model"
        ],

        strict=True,

    )


    model.to(
        DEVICE
    )


    model.eval()


    best_val_auprc = float(

        checkpoint.get(
            "best_val_auprc",
            np.nan
        )

    )


    return (

        model,

        checkpoint_path,

        best_val_auprc,

    )


# =============================================================================
# 19. Inference
# =============================================================================


def predict(

    model,

    loader: DataLoader,

):


    labels_list = []

    probs_list = []


    with torch.no_grad():


        for batch_index, (
            a,
            b,
            y
        ) in enumerate(
            loader,
            start=1,
        ):


            a = a.to(

                DEVICE,

                non_blocking=True,

            )


            b = b.to(

                DEVICE,

                non_blocking=True,

            )


            logits = model(
                a,
                b
            )


            probs = torch.sigmoid(
                logits
            )


            probs_list.append(

                probs
                .detach()
                .cpu()
                .numpy()

            )


            labels_list.append(

                y
                .detach()
                .cpu()
                .numpy()

            )


            if (
                batch_index == 1
                or
                batch_index % 50 == 0
                or
                batch_index == len(loader)
            ):

                print(

                    f"      batch "
                    f"{batch_index:4d}"
                    f"/"
                    f"{len(loader):4d}"

                )


    labels = np.concatenate(
        labels_list
    ).astype(
        np.int64
    )


    probs = np.concatenate(
        probs_list
    ).astype(
        np.float64
    )


    if not np.all(
        np.isfinite(
            probs
        )
    ):

        raise RuntimeError(
            "Prediction contains NaN/Inf."
        )


    return (
        labels,
        probs,
    )


# =============================================================================
# 20. Evaluate one external dataset
# =============================================================================


def evaluate_dataset(

    dataset_name: str,

    pair_df: pd.DataFrame,

    embedding_cache:
        Dict[
            str,
            torch.Tensor
        ],

    model_class,

):


    print()

    print("=" * 110)

    print(
        f"EXTERNAL DATASET: {dataset_name}"
    )

    print("=" * 110)


    dataset = ExternalPairDataset(

        pair_df,

        embedding_cache,

    )


    loader = DataLoader(

        dataset,

        batch_size=EVAL_BATCH_SIZE,

        shuffle=False,

        num_workers=0,

        pin_memory=(
            DEVICE
            ==
            "cuda"
        ),

    )


    prediction_df = (
        pair_df
        .reset_index(
            drop=True
        )
        .copy()
    )


    all_metrics = []


    reference_labels = None


    # =========================================================================
    # five seeds
    # =========================================================================

    for seed in SEEDS:


        print()
        print(
            f"  Seed {seed}"
        )


        (

            model,

            checkpoint_path,

            best_val_auprc,

        ) = load_model_for_seed(

            seed,

            model_class,

        )


        print(

            f"    checkpoint       : "
            f"{checkpoint_path.name}"

        )


        print(

            f"    internal Val AUPRC: "
            f"{best_val_auprc:.6f}"

        )


        labels, probs = predict(

            model,

            loader,

        )


        if reference_labels is None:

            reference_labels = (
                labels.copy()
            )


        else:

            if not np.array_equal(

                reference_labels,

                labels,

            ):

                raise RuntimeError(

                    f"{dataset_name}: "
                    f"label mismatch at seed {seed}"

                )


        metrics, preds = calculate_metrics(

            labels,

            probs,

            THRESHOLD,

        )


        metrics[
            "dataset"
        ] = dataset_name


        metrics[
            "seed"
        ] = int(
            seed
        )


        metrics[
            "checkpoint"
        ] = checkpoint_path.name


        metrics[
            "internal_best_val_auprc"
        ] = best_val_auprc


        metrics[
            "threshold"
        ] = THRESHOLD


        all_metrics.append(
            metrics
        )


        prediction_df[
            f"prob_seed_{seed}"
        ] = probs


        prediction_df[
            f"pred_seed_{seed}"
        ] = preds


        print(

            f"    AUPRC  : "
            f"{metrics['auprc']:.6f}"

        )


        print(

            f"    AUROC  : "
            f"{metrics['auroc']:.6f}"

        )


        print(

            f"    MCC    : "
            f"{metrics['mcc']:.6f}"

        )


        print(

            f"    F1     : "
            f"{metrics['f1']:.6f}"

        )


        print(

            f"    AUPRC/random: "
            f"{metrics['auprc_over_random']:.3f}x"

        )


        # release GPU model
        del model


        if torch.cuda.is_available():

            torch.cuda.empty_cache()


    # =========================================================================
    # prediction ensemble statistics
    #
    # 注意：
    # 这里只做结果分析。
    # 不把 mean prediction 当成新的 tuned model。
    # =========================================================================

    prob_columns = [

        f"prob_seed_{seed}"

        for seed in SEEDS

    ]


    prediction_df[
        "prob_mean"
    ] = (

        prediction_df[
            prob_columns
        ]
        .mean(
            axis=1
        )

    )


    prediction_df[
        "prob_std"
    ] = (

        prediction_df[
            prob_columns
        ]
        .std(
            axis=1,
            ddof=1
        )

    )


    prediction_df[
        "pred_from_mean_prob"
    ] = (

        prediction_df[
            "prob_mean"
        ]

        >=
        THRESHOLD

    ).astype(
        np.int64
    )


    # =========================================================================
    # save prediction
    # =========================================================================

    output_name_map = {

        "Human_Full":
            "esm2_gp_predictions_human_full.csv",

        "Human_NovelVsTrain":
            "esm2_gp_predictions_human_novel_train.csv",

        "Human_NovelVsAll":
            "esm2_gp_predictions_human_novel_all.csv",

        "Yeast_Full":
            "esm2_gp_predictions_yeast_full.csv",

    }


    prediction_path = (

        OUTPUT_DIR
        /
        output_name_map[
            dataset_name
        ]

    )


    prediction_df.to_csv(

        prediction_path,

        index=False,

        encoding="utf-8-sig",

    )


    print(
        f"\n[Saved] {prediction_path}"
    )


    return (

        all_metrics,

        prediction_path,

    )


# =============================================================================
# 21. Five-seed summary
# =============================================================================


def build_summary(
    seed_metrics_df: pd.DataFrame
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

        "auprc_over_random",

    ]


    rows = []


    dataset_order = [

        "Human_Full",

        "Human_NovelVsTrain",

        "Human_NovelVsAll",

        "Yeast_Full",

    ]


    for dataset_name in dataset_order:


        sub = seed_metrics_df.loc[

            seed_metrics_df[
                "dataset"
            ]
            ==
            dataset_name

        ].copy()


        if len(
            sub
        ) != len(
            SEEDS
        ):

            raise RuntimeError(

                f"{dataset_name}: expected "
                f"{len(SEEDS)} seeds, "
                f"got {len(sub)}"

            )


        n = int(
            sub[
                "n"
            ].iloc[0]
        )


        positive = int(
            sub[
                "positive"
            ].iloc[0]
        )


        negative = int(
            sub[
                "negative"
            ].iloc[0]
        )


        prevalence = float(

            sub[
                "positive_ratio"
            ].iloc[0]

        )


        row = {

            "dataset":
                dataset_name,

            "n_seeds":
                len(
                    sub
                ),

            "n_pairs":
                n,

            "positive":
                positive,

            "negative":
                negative,

            "positive_ratio":
                prevalence,

            "random_auprc":
                prevalence,

        }


        for metric in metric_names:


            values = (

                sub[
                    metric
                ]

                .astype(float)

                .to_numpy()

            )


            row[
                f"{metric}_mean"
            ] = float(
                np.mean(
                    values
                )
            )


            row[
                f"{metric}_std"
            ] = float(
                np.std(
                    values,
                    ddof=1
                )
            )


            row[
                f"{metric}_min"
            ] = float(
                np.min(
                    values
                )
            )


            row[
                f"{metric}_max"
            ] = float(
                np.max(
                    values
                )
            )


            row[
                f"{metric}_paper"
            ] = (

                f"{np.mean(values):.4f} "
                f"± "
                f"{np.std(values, ddof=1):.4f}"

            )


        rows.append(
            row
        )


    return pd.DataFrame(
        rows
    )


# =============================================================================
# 22. Protocol JSON
# =============================================================================


def save_protocol(

    seed_metrics_df: pd.DataFrame,

    summary_df: pd.DataFrame,

    prediction_paths: Dict[str, Path],

):


    checkpoint_info = {}


    for seed in SEEDS:


        path = (

            CHECKPOINT_DIR
            /
            f"best_seed_{seed}.pt"

        )


        checkpoint_info[
            str(seed)
        ] = {

            "path":
                str(
                    path
                ),

            "sha256":
                sha256_file(
                    path
                ),

        }


    dataset_info = {}


    for dataset_name, path in DATASETS.items():


        df = pd.read_csv(
            path,
            usecols=[
                "label"
            ]
        )


        positive = int(

            (
                df[
                    "label"
                ]
                ==
                1
            ).sum()

        )


        dataset_info[
            dataset_name
        ] = {

            "path":
                str(
                    path
                ),

            "sha256":
                sha256_file(
                    path
                ),

            "pairs":
                int(
                    len(df)
                ),

            "positive":
                positive,

            "negative":
                int(
                    len(df)
                    -
                    positive
                ),

            "positive_ratio":
                float(
                    positive
                    /
                    len(df)
                ),

        }


    protocol = {

        "script_version":
            SCRIPT_VERSION,

        "generated_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "project_root":
            str(
                ROOT
            ),

        "python_version":
            sys.version,

        "platform":
            platform.platform(),

        "torch_version":
            torch.__version__,

        "device":
            DEVICE,

        "experiment":
            (
                "Frozen ESM2-GP external "
                "generalization evaluation"
            ),

        "evaluation_type":
            "inference-only",

        "source_training_script":
            str(
                STEP19_SCRIPT
            ),

        "dataset_audit_gate":
            "PASS",

        "model":
            "ESM2-GP",

        "embedding":
            (
                "Frozen ESM2-t33-650M "
                "protein-level global embedding"
            ),

        "embedding_dimension":
            EXPECTED_EMBEDDING_DIM,

        "threshold":
            THRESHOLD,

        "checkpoint_selection":
            (
                "Best checkpoint selected solely "
                "by Bernett Validation AUPRC in Step 17"
            ),

        "external_tuning":
            False,

        "external_finetuning":
            False,

        "seeds":
            SEEDS,

        "n_seeds":
            len(
                SEEDS
            ),

        "standard_deviation":
            (
                "sample standard deviation, ddof=1"
            ),

        "primary_external_metric":
            "AUPRC",

        "secondary_metrics": [

            "AUROC",

            "MCC",

            "F1",

            "Accuracy",

            "Precision",

            "Recall",

            "Specificity",

        ],

        "evaluation_settings": {

            "Human_Full":
                (
                    "Cross-dataset external evaluation"
                ),

            "Human_NovelVsTrain":
                (
                    "Strict within-species external "
                    "cold-start; no exact-sequence "
                    "match with Bernett Training"
                ),

            "Human_NovelVsAll":
                (
                    "Ultra-strict exact-sequence "
                    "novelty stress test; no exact-sequence "
                    "match with Bernett Train/Val/Test"
                ),

            "Yeast_Full":
                (
                    "Cross-species zero-shot evaluation; "
                    "zero exact-sequence overlap with "
                    "Bernett Train/Val/Test"
                ),

        },

        "checkpoints":
            checkpoint_info,

        "datasets":
            dataset_info,

        "outputs": {

            "seed_metrics":
                str(
                    SEED_METRICS_PATH
                ),

            "summary":
                str(
                    SUMMARY_PATH
                ),

            "predictions": {

                k:
                    str(v)

                for k, v
                in prediction_paths.items()

            },

        },

    }


    # -------------------------------------------------------------------------
    # convenient paper summary
    # -------------------------------------------------------------------------

    paper_summary = {}


    for _, row in summary_df.iterrows():


        dataset_name = row[
            "dataset"
        ]


        paper_summary[
            dataset_name
        ] = {

            "random_auprc":
                float(
                    row[
                        "random_auprc"
                    ]
                ),

            "auprc_mean":
                float(
                    row[
                        "auprc_mean"
                    ]
                ),

            "auprc_std":
                float(
                    row[
                        "auprc_std"
                    ]
                ),

            "auprc_paper":
                row[
                    "auprc_paper"
                ],

            "auroc_mean":
                float(
                    row[
                        "auroc_mean"
                    ]
                ),

            "auroc_std":
                float(
                    row[
                        "auroc_std"
                    ]
                ),

            "mcc_mean":
                float(
                    row[
                        "mcc_mean"
                    ]
                ),

            "f1_mean":
                float(
                    row[
                        "f1_mean"
                    ]
                ),

            "auprc_over_random_mean":
                float(
                    row[
                        "auprc_over_random_mean"
                    ]
                ),

        }


    protocol[
        "paper_summary"
    ] = paper_summary


    with PROTOCOL_PATH.open(

        "w",

        encoding="utf-8",

    ) as f:


        json.dump(

            protocol,

            f,

            indent=2,

            ensure_ascii=False,

        )


    print(
        f"[Saved] {PROTOCOL_PATH}"
    )


# =============================================================================
# 23. Main
# =============================================================================


def main():


    print("=" * 120)

    print(
        "STEP 20 — ESM2-GP EXTERNAL GENERALIZATION"
    )

    print("=" * 120)


    print(
        f"Project     : {ROOT}"
    )


    print(
        f"Device      : {DEVICE}"
    )


    print(
        f"Batch size  : {EVAL_BATCH_SIZE}"
    )


    print(
        f"Threshold   : {THRESHOLD}"
    )


    print(
        f"Seeds       : {SEEDS}"
    )


    print(
        f"Output      : {OUTPUT_DIR}"
    )


    print("=" * 120)


    # =========================================================================
    # 1. Input gate
    # =========================================================================

    require_file(
        STEP19_SCRIPT
    )


    require_dir(
        CHECKPOINT_DIR
    )


    require_dir(
        HUMAN_GLOBAL_DIR
    )


    require_dir(
        YEAST_GLOBAL_DIR
    )


    for path in DATASETS.values():

        require_file(
            path
        )


    for seed in SEEDS:

        require_file(

            CHECKPOINT_DIR
            /
            f"best_seed_{seed}.pt"

        )


    print(
        "\n[1/8] Input files PASS"
    )


    # =========================================================================
    # 2. Step 15
    # =========================================================================

    validate_prepared_data()


    print(
        "[2/8] Dataset audit gate PASS"
    )


    # =========================================================================
    # 3. Load Step 17 model
    # =========================================================================

    step19 = load_step19_module()


    model_class = (
        step19
        .ESM2GlobalBaseline
    )


    print(
        "[3/8] Step 17 model loaded"
    )


    # =========================================================================
    # 4. Read pair files
    # =========================================================================

    pair_frames = {}


    for dataset_name, path in DATASETS.items():


        pair_frames[
            dataset_name
        ] = load_pair_csv(

            path,

            dataset_name,

        )


    print(
        "[4/8] External canonical datasets PASS"
    )


    # =========================================================================
    # 5. Embedding resolver
    # =========================================================================

    human_resolver = EmbeddingResolver(

        HUMAN_EMBEDDING_ROOT,

        HUMAN_GLOBAL_DIR,

        "Human",

    )


    yeast_resolver = EmbeddingResolver(

        YEAST_EMBEDDING_ROOT,

        YEAST_GLOBAL_DIR,

        "Yeast",

    )


    print(
        "[5/8] External embedding indices ready"
    )


    # =========================================================================
    # 6. Preload Human Full proteins
    #
    # Human Full 包含另外两个 Human subset，
    # 所以只 preload 一次。
    # =========================================================================

    human_cache = preload_embeddings(

        pair_frames[
            "Human_Full"
        ],

        human_resolver,

        "Human Full",

    )


    # =========================================================================
    # 7. Preload Yeast
    # =========================================================================

    yeast_cache = preload_embeddings(

        pair_frames[
            "Yeast_Full"
        ],

        yeast_resolver,

        "Yeast Full",

    )


    print(
        "[6/8] External embeddings preloaded"
    )


    # =========================================================================
    # 8. Evaluate
    # =========================================================================

    all_metrics = []


    prediction_paths = {}


    evaluation_plan = [

        (
            "Human_Full",
            human_cache,
        ),

        (
            "Human_NovelVsTrain",
            human_cache,
        ),

        (
            "Human_NovelVsAll",
            human_cache,
        ),

        (
            "Yeast_Full",
            yeast_cache,
        ),

    ]


    for dataset_name, cache in evaluation_plan:


        metrics, pred_path = evaluate_dataset(

            dataset_name,

            pair_frames[
                dataset_name
            ],

            cache,

            model_class,

        )


        all_metrics.extend(
            metrics
        )


        prediction_paths[
            dataset_name
        ] = pred_path


    print(
        "\n[7/8] All external evaluations complete"
    )


    # =========================================================================
    # 9. Seed metrics
    # =========================================================================

    seed_metrics_df = pd.DataFrame(
        all_metrics
    )


    preferred_columns = [

        "dataset",

        "seed",

        "checkpoint",

        "internal_best_val_auprc",

        "threshold",

        "n",

        "positive",

        "negative",

        "positive_ratio",

        "random_auprc",

        "accuracy",

        "precision",

        "recall",

        "specificity",

        "f1",

        "mcc",

        "auroc",

        "auprc",

        "auprc_over_random",

        "tp",

        "tn",

        "fp",

        "fn",

    ]


    seed_metrics_df = seed_metrics_df[
        preferred_columns
    ]


    seed_metrics_df.to_csv(

        SEED_METRICS_PATH,

        index=False,

        encoding="utf-8-sig",

    )


    print(
        f"[Saved] {SEED_METRICS_PATH}"
    )


    # =========================================================================
    # 10. Summary
    # =========================================================================

    summary_df = build_summary(
        seed_metrics_df
    )


    summary_df.to_csv(

        SUMMARY_PATH,

        index=False,

        encoding="utf-8-sig",

    )


    print(
        f"[Saved] {SUMMARY_PATH}"
    )


    # =========================================================================
    # 11. Protocol
    # =========================================================================

    save_protocol(

        seed_metrics_df,

        summary_df,

        prediction_paths,

    )


    print(
        "[8/8] Protocol and summary saved"
    )


    # =========================================================================
    # 12. Console paper summary
    # =========================================================================

    print()
    print("=" * 120)

    print(
        "FINAL ESM2-GP EXTERNAL SUMMARY"
    )

    print("=" * 120)


    display_columns = [

        "dataset",

        "n_pairs",

        "positive_ratio",

        "random_auprc",

        "auprc_mean",

        "auprc_std",

        "auroc_mean",

        "mcc_mean",

        "f1_mean",

        "auprc_over_random_mean",

    ]


    print(

        summary_df[
            display_columns
        ].to_string(
            index=False
        )

    )


    print()
    print("-" * 120)

    print(
        "Paper-ready AUPRC"
    )

    print("-" * 120)


    for _, row in summary_df.iterrows():


        print(

            f"{row['dataset']:<25s} | "
            f"Random={row['random_auprc']:.4f} | "
            f"ESM2-GP="
            f"{row['auprc_mean']:.4f}"
            f" ± "
            f"{row['auprc_std']:.4f} | "
            f"{row['auprc_over_random_mean']:.2f}x random"

        )


    print()
    print("=" * 120)

    print(
        "STEP 20 COMPLETED SUCCESSFULLY"
    )

    print("=" * 120)


    print()
    print(
        "IMPORTANT:"
    )


    print(

        "External datasets were used for inference only. "
        "Do not tune model architecture, checkpoint, "
        "threshold, or hyperparameters based on these results."

    )


# =============================================================================
# 24. Entry
# =============================================================================


if __name__ == "__main__":

    main()
