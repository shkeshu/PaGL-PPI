from __future__ import annotations

import argparse
import json
import random
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
    confusion_matrix,
)
from torch.utils.data import Dataset, DataLoader


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "specificity": float(tn / max(tn + fp, 1)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, pred)),
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
    }


def preload_features(
    protein_ids: List[str],
    feature_dir: Path,
):
    """
    Load train+val protein features once into RAM.

    segments are kept as float16 CPU tensors.
    global vectors are stacked into one float32 matrix.
    """
    pids = sorted(set(protein_ids))
    pid_to_idx = {}
    segment_cache = []
    globals_list = []

    print(f"Loading segment features for {len(pids):,} proteins...")

    total_segments = 0

    for i, pid in enumerate(pids):
        path = feature_dir / f"{pid}.pt"
        if not path.exists():
            raise FileNotFoundError(path)

        obj = torch.load(path, map_location="cpu", weights_only=True)
        seg = obj["segments"].half().contiguous()
        glob = obj["global"].float().contiguous()

        if seg.ndim != 2 or seg.shape[1] != 1280:
            raise ValueError(f"{pid}: invalid segment shape {tuple(seg.shape)}")
        if glob.shape != (1280,):
            raise ValueError(f"{pid}: invalid global shape {tuple(glob.shape)}")

        pid_to_idx[pid] = i
        segment_cache.append(seg)
        globals_list.append(glob)
        total_segments += seg.shape[0]

        if (i + 1) % 1000 == 0 or i + 1 == len(pids):
            print(f"  loaded {i+1:,}/{len(pids):,}")

    global_matrix = torch.stack(globals_list, dim=0)

    approx_mb = (
        total_segments * 1280 * 2
        + global_matrix.numel() * 4
    ) / 1024**2

    print(
        f"Feature cache ready: total segments={total_segments:,}, "
        f"approx={approx_mb:.1f} MB"
    )

    return segment_cache, global_matrix, pid_to_idx


class PairIndexDataset(Dataset):
    def __init__(self, csv_path: Path, pid_to_idx: Dict[str, int]):
        df = pd.read_csv(csv_path)
        self.a = torch.tensor(
            [pid_to_idx[str(x)] for x in df["protein_a"]],
            dtype=torch.long,
        )
        self.b = torch.tensor(
            [pid_to_idx[str(x)] for x in df["protein_b"]],
            dtype=torch.long,
        )
        self.y = torch.tensor(
            df["label"].astype(float).values,
            dtype=torch.float32,
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.a[idx], self.b[idx], self.y[idx]


class SegmentCollator:
    def __init__(self, segment_cache, global_matrix):
        self.segment_cache = segment_cache
        self.global_matrix = global_matrix

    def __call__(self, batch):
        a_idx = [int(x[0]) for x in batch]
        b_idx = [int(x[1]) for x in batch]
        y = torch.stack([x[2] for x in batch])

        a_seg = [self.segment_cache[i] for i in a_idx]
        b_seg = [self.segment_cache[i] for i in b_idx]

        max_len = max(
            max(x.shape[0] for x in a_seg),
            max(x.shape[0] for x in b_seg),
        )
        B = len(batch)

        A = torch.zeros((B, max_len, 1280), dtype=torch.float16)
        Bseg = torch.zeros((B, max_len, 1280), dtype=torch.float16)
        A_mask = torch.zeros((B, max_len), dtype=torch.bool)
        B_mask = torch.zeros((B, max_len), dtype=torch.bool)

        for i, x in enumerate(a_seg):
            n = x.shape[0]
            A[i, :n] = x
            A_mask[i, :n] = True

        for i, x in enumerate(b_seg):
            n = x.shape[0]
            Bseg[i, :n] = x
            B_mask[i, :n] = True

        Aglob = self.global_matrix[torch.tensor(a_idx)]
        Bglob = self.global_matrix[torch.tensor(b_idx)]

        return A, A_mask, Bseg, B_mask, Aglob, Bglob, y


def masked_mean(x, valid_mask):
    w = valid_mask.unsqueeze(-1).to(x.dtype)
    return (x * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)


def symmetric_pair(a, b):
    return torch.cat(
        [a + b, torch.abs(a - b), a * b],
        dim=-1,
    )


class CrossBlock(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, kv, q_valid, kv_valid):
        attn_out, _ = self.attn(
            q,
            kv,
            kv,
            key_padding_mask=~kv_valid,
            need_weights=False,
        )
        x = self.norm1(q + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        x = x * q_valid.unsqueeze(-1).to(x.dtype)
        return x


class SegmentPPIModel(nn.Module):
    """
    Modes:
      B1_segment_only
      B2_segment_cross
      B3_cross_global_concat
      B4_full_gate
    """

    def __init__(
        self,
        mode: str,
        input_dim=1280,
        segment_dim=256,
        heads=4,
        classifier_hidden=384,
        dropout=0.30,
    ):
        super().__init__()
        self.mode = mode
        self.segment_dim = segment_dim

        self.segment_encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, segment_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.global_encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, segment_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.cross = CrossBlock(
            segment_dim,
            heads,
            dropout,
        )

        pair_dim = segment_dim * 3

        if mode in ("B1_segment_only", "B2_segment_cross"):
            classifier_in = pair_dim

        elif mode == "B3_cross_global_concat":
            classifier_in = pair_dim * 2

        elif mode == "B4_full_gate":
            self.cross_proj = nn.Linear(pair_dim, classifier_hidden)
            self.global_proj = nn.Linear(pair_dim, classifier_hidden)

            self.gate = nn.Sequential(
                nn.Linear(classifier_hidden * 2, classifier_hidden // 2),
                nn.GELU(),
                nn.Linear(classifier_hidden // 2, 1),
                nn.Sigmoid(),
            )
            classifier_in = classifier_hidden

        else:
            raise ValueError(mode)

        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_in),
            nn.Linear(classifier_in, classifier_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, classifier_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden // 2, 1),
        )

    def forward(
        self,
        a_seg,
        a_mask,
        b_seg,
        b_mask,
        a_global,
        b_global,
        return_gate=False,
    ):
        a_seg = self.segment_encoder(a_seg.float())
        b_seg = self.segment_encoder(b_seg.float())

        if self.mode == "B1_segment_only":
            za = masked_mean(a_seg, a_mask)
            zb = masked_mean(b_seg, b_mask)
            pair = symmetric_pair(za, zb)
            logits = self.classifier(pair).squeeze(-1)
            return (logits, None) if return_gate else logits

        # Bidirectional partner-aware cross-attention.
        a_cross = self.cross(a_seg, b_seg, a_mask, b_mask)
        b_cross = self.cross(b_seg, a_seg, b_mask, a_mask)

        za = masked_mean(a_cross, a_mask)
        zb = masked_mean(b_cross, b_mask)
        cross_pair = symmetric_pair(za, zb)

        if self.mode == "B2_segment_cross":
            logits = self.classifier(cross_pair).squeeze(-1)
            return (logits, None) if return_gate else logits

        ga = self.global_encoder(a_global.float())
        gb = self.global_encoder(b_global.float())
        global_pair = symmetric_pair(ga, gb)

        if self.mode == "B3_cross_global_concat":
            fused = torch.cat([cross_pair, global_pair], dim=-1)
            logits = self.classifier(fused).squeeze(-1)
            return (logits, None) if return_gate else logits

        # B4: explicit gate, retained only as controlled ablation.
        c = self.cross_proj(cross_pair)
        g = self.global_proj(global_pair)
        gate = self.gate(torch.cat([c, g], dim=-1))
        fused = gate * c + (1.0 - gate) * g

        logits = self.classifier(fused).squeeze(-1)

        if return_gate:
            return logits, gate.squeeze(-1)
        return logits


@torch.inference_mode()
def evaluate(model, loader, device, collect_gate=False):
    model.eval()

    ys, ps = [], []
    gates = []

    for batch in loader:
        A, Am, B, Bm, Ag, Bg, y = batch

        A = A.to(device, non_blocking=True)
        Am = Am.to(device, non_blocking=True)
        B = B.to(device, non_blocking=True)
        Bm = Bm.to(device, non_blocking=True)
        Ag = Ag.to(device, non_blocking=True)
        Bg = Bg.to(device, non_blocking=True)

        if collect_gate:
            logits, gate = model(
                A, Am, B, Bm, Ag, Bg,
                return_gate=True,
            )
            if gate is not None:
                gates.append(gate.cpu().numpy())
        else:
            logits = model(A, Am, B, Bm, Ag, Bg)

        prob = torch.sigmoid(logits)

        ys.append(y.numpy())
        ps.append(prob.cpu().numpy())

    y = np.concatenate(ys)
    p = np.concatenate(ps)

    gate_stats = None
    if gates:
        g = np.concatenate(gates)
        gate_stats = {
            "mean": float(g.mean()),
            "std": float(g.std()),
            "min": float(g.min()),
            "max": float(g.max()),
        }

    return y, p, gate_stats


def train_one(
    cfg,
    mode,
    train_ds,
    val_ds,
    collator,
    device,
    out_root,
):
    set_seed(int(cfg["seed"]))

    tc = cfg["training"]
    mc = cfg["model"]

    out_dir = out_root / mode
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=tc["batch_size"],
        shuffle=True,
        num_workers=0,
        collate_fn=collator,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=tc["batch_size"],
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
        pin_memory=(device.type == "cuda"),
    )

    model = SegmentPPIModel(
        mode=mode,
        input_dim=1280,
        segment_dim=mc["segment_dim"],
        heads=mc["heads"],
        classifier_hidden=mc["classifier_hidden"],
        dropout=mc["dropout"],
    ).to(device)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n" + "=" * 110)
    print("Experiment:", mode)
    print("Trainable parameters:", f"{params:,}")
    print("=" * 110)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=tc["lr"],
        weight_decay=tc["weight_decay"],
    )
    loss_fn = nn.BCEWithLogitsLoss()

    best = -1.0
    best_epoch = -1
    bad = 0
    history = []

    for epoch in range(1, tc["epochs"] + 1):
        model.train()
        total_loss = 0.0
        total_n = 0
        t0 = time.time()

        for batch in train_loader:
            A, Am, B, Bm, Ag, Bg, y = batch

            A = A.to(device, non_blocking=True)
            Am = Am.to(device, non_blocking=True)
            B = B.to(device, non_blocking=True)
            Bm = Bm.to(device, non_blocking=True)
            Ag = Ag.to(device, non_blocking=True)
            Bg = Bg.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            logits = model(A, Am, B, Bm, Ag, Bg)
            loss = loss_fn(logits, y)
            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                tc["grad_clip"],
            )
            opt.step()

            total_loss += float(loss.item()) * len(y)
            total_n += len(y)

        yv, pv, _ = evaluate(model, val_loader, device, collect_gate=False)
        m = metrics(yv, pv, 0.5)
        avg_loss = total_loss / max(total_n, 1)

        row = {
            "epoch": epoch,
            "train_loss": avg_loss,
            **{f"val_{k}": v for k, v in m.items()},
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d} | "
            f"loss={avg_loss:.5f} | "
            f"AUPRC={m['auprc']:.5f} | "
            f"AUROC={m['auroc']:.5f} | "
            f"F1={m['f1']:.5f} | "
            f"MCC={m['mcc']:.5f} | "
            f"{time.time()-t0:.1f}s"
        )

        if m["auprc"] > best:
            best = m["auprc"]
            best_epoch = epoch
            bad = 0

            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "mode": mode,
                    "config": cfg,
                    "best_epoch": best_epoch,
                    "best_val_auprc": best,
                },
                out_dir / "best.pt",
            )
        else:
            bad += 1
            if bad >= tc["early_stopping_patience"]:
                print(
                    f"Early stopping. best_epoch={best_epoch}, "
                    f"best_AUPRC={best:.5f}"
                )
                break

    pd.DataFrame(history).to_csv(
        out_dir / "history.csv",
        index=False,
        encoding="utf-8-sig",
    )

    ckpt = torch.load(
        out_dir / "best.pt",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(ckpt["state_dict"])

    yv, pv, gate_stats = evaluate(
        model,
        val_loader,
        device,
        collect_gate=(mode == "B4_full_gate"),
    )
    m = metrics(yv, pv, 0.5)

    pd.DataFrame({
        "label": yv.astype(int),
        "probability": pv,
    }).to_csv(
        out_dir / "val_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    result = {
        "mode": mode,
        "best_epoch": int(ckpt["best_epoch"]),
        "parameters": int(params),
        "metrics": m,
        "gate_stats": gate_stats,
    }

    (out_dir / "best_val_metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(
        f"[BEST] {mode} | "
        f"AUPRC={m['auprc']:.5f} | "
        f"AUROC={m['auroc']:.5f} | "
        f"F1={m['f1']:.5f} | "
        f"MCC={m['mcc']:.5f}"
    )
    if gate_stats:
        print(
            "Gate stats:",
            json.dumps(gate_stats, ensure_ascii=False)
        )

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default=str(ROOT / "Configs" / "segment_cross_ablation.json"),
    )
    ap.add_argument(
        "--only",
        default="B4_full_gate",
        help=(
            "B1_segment_only / B2_segment_cross / "
            "B3_cross_global_concat / B4_full_gate"
        ),
    )
    args = ap.parse_args()

    cfg = json.loads(
        Path(args.config).read_text(encoding="utf-8")
    )

    set_seed(int(cfg["seed"]))

    device = torch.device(
        "cuda"
        if cfg["device"] == "cuda" and torch.cuda.is_available()
        else "cpu"
    )

    def R(p):
        q = Path(p)
        return q if q.is_absolute() else ROOT / q

    train_csv = R(cfg["data"]["train_pairs"])
    val_csv = R(cfg["data"]["val_pairs"])
    feature_dir = R(cfg["data"]["segment_embedding_dir"])

    train_df = pd.read_csv(
        train_csv,
        usecols=["protein_a", "protein_b", "label"],
    )
    val_df = pd.read_csv(
        val_csv,
        usecols=["protein_a", "protein_b", "label"],
    )

    pids = sorted(
        set(train_df["protein_a"].astype(str))
        | set(train_df["protein_b"].astype(str))
        | set(val_df["protein_a"].astype(str))
        | set(val_df["protein_b"].astype(str))
    )

    segment_cache, global_matrix, pid_to_idx = preload_features(
        pids,
        feature_dir,
    )

    train_ds = PairIndexDataset(train_csv, pid_to_idx)
    val_ds = PairIndexDataset(val_csv, pid_to_idx)
    collator = SegmentCollator(segment_cache, global_matrix)

    modes = cfg["modes"]
    if args.only:
        if args.only not in modes:
            raise ValueError(f"--only={args.only} not in {modes}")
        modes = [args.only]

    out_root = R(cfg["output_dir"])
    out_root.mkdir(parents=True, exist_ok=True)

    print("=" * 110)
    print("Bernett controlled segment/cross-attention ablation")
    print("=" * 110)
    print("Device:", device)
    print("Train pairs:", f"{len(train_ds):,}")
    print("Val pairs  :", f"{len(val_ds):,}")
    print("Modes      :", modes)
    print("TEST SET   : NOT ACCESSED")

    results = []
    for mode in modes:
        results.append(
            train_one(
                cfg,
                mode,
                train_ds,
                val_ds,
                collator,
                device,
                out_root,
            )
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    rows = []
    for r in results:
        m = r["metrics"]
        row = {
            "mode": r["mode"],
            "best_epoch": r["best_epoch"],
            "parameters": r["parameters"],
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "specificity": m["specificity"],
            "f1": m["f1"],
            "mcc": m["mcc"],
            "auroc": m["auroc"],
            "auprc": m["auprc"],
        }
        if r["gate_stats"]:
            row["gate_mean"] = r["gate_stats"]["mean"]
            row["gate_std"] = r["gate_stats"]["std"]
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values(
        "auprc",
        ascending=False,
    )
    summary.to_csv(
        out_root / "segment_cross_ablation_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 110)
    print("FINAL VALIDATION SUMMARY")
    print("=" * 110)
    print(summary.to_string(index=False))
    print("\nReference clean strong baseline:")
    print("  B0_v2 sum_diff_product AUPRC ≈ 0.66946")
    print("\nDecision rule:")
    print("  Keep a module only if ΔAUPRC >= +0.010 and later multi-seed results are consistent.")
    print("  Bernett Test remains untouched.")


if __name__ == "__main__":
    main()
