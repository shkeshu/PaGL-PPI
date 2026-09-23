from __future__ import annotations
import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, average_precision_score,
    confusion_matrix
)
from torch.utils.data import Dataset, DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calc_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
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
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }

def find_best_threshold(y_true, y_prob, metric="f1", min_t=0.10, max_t=0.90, step=0.01):
    best_t = 0.5
    best_m = calc_metrics(y_true, y_prob, 0.5)
    best_score = best_m[metric]
    for t in np.arange(min_t, max_t + 1e-12, step):
        m = calc_metrics(y_true, y_prob, float(t))
        if m[metric] > best_score:
            best_t = float(t)
            best_m = m
            best_score = m[metric]
    return best_t, best_m

def load_one_embedding(path: Path, expected_dim: int):
    x = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(x, dict):
        for key in ("embedding", "emb", "x", "representations"):
            if key in x:
                x = x[key]
                break
    x = torch.as_tensor(x).float()
    if x.ndim == 2:
        x = x.mean(dim=0)
    if x.ndim != 1 or x.numel() != expected_dim:
        raise ValueError(f"{path.name}: unexpected shape {tuple(x.shape)}")
    if not torch.isfinite(x).all():
        raise ValueError(f"{path.name}: NaN/Inf detected")
    return x.contiguous()

def preload_embeddings(protein_ids: List[str], emb_dir: Path, expected_dim: int):
    protein_ids = sorted(set(protein_ids))
    matrix = torch.empty((len(protein_ids), expected_dim), dtype=torch.float32)
    pid_to_idx = {}
    print(f"Loading {len(protein_ids):,} unique embeddings into RAM...")
    for i, pid in enumerate(protein_ids):
        path = emb_dir / f"{pid}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing embedding: {path}")
        matrix[i] = load_one_embedding(path, expected_dim)
        pid_to_idx[pid] = i
        if (i + 1) % 1000 == 0 or (i + 1) == len(protein_ids):
            print(f"  loaded {i+1:,}/{len(protein_ids):,}")
    mem_mb = matrix.numel() * matrix.element_size() / 1024**2
    print(f"Embedding cache ready: shape={tuple(matrix.shape)}, ~{mem_mb:.1f} MB")
    return matrix, pid_to_idx

class IndexedPairDataset(Dataset):
    def __init__(self, csv_path: Path, pid_to_idx: Dict[str, int]):
        df = pd.read_csv(csv_path)
        req = {"protein_a", "protein_b", "label"}
        if req - set(df.columns):
            raise ValueError(f"{csv_path}: missing columns {sorted(req - set(df.columns))}")
        self.a_idx = torch.tensor([pid_to_idx[str(x)] for x in df["protein_a"]], dtype=torch.long)
        self.b_idx = torch.tensor([pid_to_idx[str(x)] for x in df["protein_b"]], dtype=torch.long)
        self.y = torch.tensor(df["label"].astype(float).values, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.a_idx[idx], self.b_idx[idx], self.y[idx]

class StrongSymmetricPPI(nn.Module):
    def __init__(self, input_dim=1280, projection_dim=384, hidden_dim=384,
                 dropout=0.35, pair_mode="diff_product"):
        super().__init__()
        self.pair_mode = pair_mode
        self.protein_encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        if pair_mode in ("diff", "product"):
            pair_dim = projection_dim
        elif pair_mode == "diff_product":
            pair_dim = projection_dim * 2
        elif pair_mode == "sum_diff_product":
            pair_dim = projection_dim * 3
        else:
            raise ValueError(f"Unknown pair_mode={pair_mode}")

        self.pair_norm = nn.LayerNorm(pair_dim)
        self.classifier = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def pair_features(self, za, zb):
        diff = torch.abs(za - zb)
        prod = za * zb
        if self.pair_mode == "diff":
            z = diff
        elif self.pair_mode == "product":
            z = prod
        elif self.pair_mode == "diff_product":
            z = torch.cat([diff, prod], dim=-1)
        else:
            summ = za + zb
            z = torch.cat([summ, diff, prod], dim=-1)
        return self.pair_norm(z)

    def forward(self, a, b):
        za = self.protein_encoder(a)
        zb = self.protein_encoder(b)
        return self.classifier(self.pair_features(za, zb)).squeeze(-1)

@torch.inference_mode()
def predict(model, loader, emb_matrix, device):
    model.eval()
    ys, ps = [], []
    for a_idx, b_idx, y in loader:
        a = emb_matrix[a_idx].to(device, non_blocking=True)
        b = emb_matrix[b_idx].to(device, non_blocking=True)
        p = torch.sigmoid(model(a, b))
        ys.append(y.numpy())
        ps.append(p.cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)

def train_one(cfg, pair_mode, train_ds, val_ds, emb_matrix, device, out_root):
    set_seed(int(cfg["seed"]))
    tc = cfg["training"]
    mc = cfg["model"]

    exp_dir = out_root / pair_mode
    exp_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*100)
    print("Experiment:", pair_mode)
    print("="*100)

    tr = DataLoader(train_ds, batch_size=tc["batch_size"], shuffle=True,
                    num_workers=tc["num_workers"], pin_memory=(device.type == "cuda"))
    va = DataLoader(val_ds, batch_size=tc["batch_size"], shuffle=False,
                    num_workers=tc["num_workers"], pin_memory=(device.type == "cuda"))

    model = StrongSymmetricPPI(
        input_dim=mc["input_dim"],
        projection_dim=mc["projection_dim"],
        hidden_dim=mc["hidden_dim"],
        dropout=mc["dropout"],
        pair_mode=pair_mode,
    ).to(device)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss()

    best_auprc = -1.0
    best_epoch = -1
    bad = 0
    history = []

    for epoch in range(1, tc["epochs"] + 1):
        model.train()
        running = 0.0
        n = 0
        t0 = time.time()

        for a_idx, b_idx, y in tr:
            a = emb_matrix[a_idx].to(device, non_blocking=True)
            b = emb_matrix[b_idx].to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            logits = model(a, b)
            loss = loss_fn(logits, y)
            loss.backward()

            if tc["grad_clip"] > 0:
                nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            opt.step()

            running += float(loss.item()) * len(y)
            n += len(y)

        yv, pv = predict(model, va, emb_matrix, device)
        m = calc_metrics(yv, pv, 0.5)
        loss_avg = running / max(n, 1)

        history.append({
            "epoch": epoch,
            "train_loss": loss_avg,
            "val_auprc": m["auprc"],
            "val_auroc": m["auroc"],
            "val_accuracy_05": m["accuracy"],
            "val_precision_05": m["precision"],
            "val_recall_05": m["recall"],
            "val_specificity_05": m["specificity"],
            "val_f1_05": m["f1"],
            "val_mcc_05": m["mcc"],
        })

        print(
            f"Epoch {epoch:03d} | loss={loss_avg:.5f} | "
            f"AUPRC={m['auprc']:.5f} AUROC={m['auroc']:.5f} "
            f"F1={m['f1']:.5f} MCC={m['mcc']:.5f} | "
            f"{time.time()-t0:.1f}s"
        )

        if m["auprc"] > best_auprc:
            best_auprc = m["auprc"]
            best_epoch = epoch
            bad = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "pair_mode": pair_mode,
                "config": cfg,
                "best_epoch": best_epoch,
                "best_val_auprc": best_auprc,
            }, exp_dir / "best.pt")
        else:
            bad += 1
            if bad >= tc["early_stopping_patience"]:
                print(f"Early stopping. Best epoch={best_epoch}, AUPRC={best_auprc:.5f}")
                break

    pd.DataFrame(history).to_csv(exp_dir / "history.csv", index=False, encoding="utf-8-sig")

    ckpt = torch.load(exp_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    yv, pv = predict(model, va, emb_matrix, device)
    m05 = calc_metrics(yv, pv, 0.5)

    cal_cfg = cfg["threshold_calibration"]
    best_t, mcal = find_best_threshold(
        yv, pv,
        metric=cal_cfg["metric"],
        min_t=cal_cfg["min"],
        max_t=cal_cfg["max"],
        step=cal_cfg["step"],
    )

    pd.DataFrame({"label": yv.astype(int), "probability": pv}).to_csv(
        exp_dir / "val_predictions.csv", index=False, encoding="utf-8-sig"
    )

    result = {
        "pair_mode": pair_mode,
        "seed": int(cfg["seed"]),
        "best_epoch": int(ckpt["best_epoch"]),
        "trainable_parameters": int(params),
        "validation_at_threshold_0.5": m05,
        "calibrated_threshold": float(best_t),
        "validation_at_calibrated_threshold": mcal,
    }

    (exp_dir / "best_val_metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"[BEST] {pair_mode}: AUPRC={m05['auprc']:.5f}, AUROC={m05['auroc']:.5f}, "
          f"F1@0.5={m05['f1']:.5f}, MCC@0.5={m05['mcc']:.5f}, "
          f"best threshold={best_t:.2f}, F1_cal={mcal['f1']:.5f}")

    return result

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "Configs" / "strong_baseline.json"))
    ap.add_argument("--only", default="",
                    help="diff / product / diff_product / sum_diff_product")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    set_seed(int(cfg["seed"]))

    device = torch.device(
        "cuda" if cfg["device"] == "cuda" and torch.cuda.is_available() else "cpu"
    )

    print("="*100)
    print("Bernett B0_v2 Strong Symmetric Baseline Ablation")
    print("="*100)
    print("Project root:", ROOT)
    print("Device      :", device)
    print("Seed        :", cfg["seed"])

    def R(p):
        q = Path(p)
        return q if q.is_absolute() else ROOT / q

    train_csv = R(cfg["data"]["train_pairs"])
    val_csv = R(cfg["data"]["val_pairs"])
    emb_dir = R(cfg["data"]["embedding_dir"])

    train_df = pd.read_csv(train_csv, usecols=["protein_a", "protein_b", "label"])
    val_df = pd.read_csv(val_csv, usecols=["protein_a", "protein_b", "label"])

    protein_ids = sorted(
        set(train_df["protein_a"].astype(str))
        | set(train_df["protein_b"].astype(str))
        | set(val_df["protein_a"].astype(str))
        | set(val_df["protein_b"].astype(str))
    )

    emb_matrix, pid_to_idx = preload_embeddings(
        protein_ids, emb_dir, cfg["model"]["input_dim"]
    )

    train_ds = IndexedPairDataset(train_csv, pid_to_idx)
    val_ds = IndexedPairDataset(val_csv, pid_to_idx)

    print(f"Train pairs: {len(train_ds):,}")
    print(f"Val pairs  : {len(val_ds):,}")

    modes = cfg["pair_modes"]
    if args.only:
        if args.only not in modes:
            raise ValueError(f"--only={args.only} not in {modes}")
        modes = [args.only]

    out_root = R(cfg["output_dir"])
    out_root.mkdir(parents=True, exist_ok=True)

    results = []
    for mode in modes:
        results.append(
            train_one(cfg, mode, train_ds, val_ds, emb_matrix, device, out_root)
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    rows = []
    for r in results:
        m05 = r["validation_at_threshold_0.5"]
        mc = r["validation_at_calibrated_threshold"]
        rows.append({
            "pair_mode": r["pair_mode"],
            "best_epoch": r["best_epoch"],
            "parameters": r["trainable_parameters"],
            "auprc": m05["auprc"],
            "auroc": m05["auroc"],
            "accuracy_05": m05["accuracy"],
            "precision_05": m05["precision"],
            "recall_05": m05["recall"],
            "specificity_05": m05["specificity"],
            "f1_05": m05["f1"],
            "mcc_05": m05["mcc"],
            "calibrated_threshold": r["calibrated_threshold"],
            "accuracy_cal": mc["accuracy"],
            "precision_cal": mc["precision"],
            "recall_cal": mc["recall"],
            "specificity_cal": mc["specificity"],
            "f1_cal": mc["f1"],
            "mcc_cal": mc["mcc"],
        })

    summary = pd.DataFrame(rows).sort_values("auprc", ascending=False)
    summary.to_csv(out_root / "ablation_summary.csv", index=False, encoding="utf-8-sig")

    print("\n" + "="*100)
    print("FINAL VALIDATION SUMMARY")
    print("="*100)
    print(summary[
        ["pair_mode", "best_epoch", "auprc", "auroc", "f1_05", "mcc_05",
         "calibrated_threshold", "f1_cal", "mcc_cal"]
    ].to_string(index=False))
    print("\nIMPORTANT: Bernett Test was NOT accessed.")

if __name__ == "__main__":
    main()
