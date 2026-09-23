from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

DEFAULT_FASTA = ROOT / "Data" / "processed" / "canonical" / "proteins.fasta"
DEFAULT_OUT = ROOT / "Data" / "embeddings" / "ESM2_t33_segments"
DEFAULT_MODEL = "facebook/esm2_t33_650M_UR50D"

EXPECTED_PROTEINS = 11019
EXPECTED_DIM = 1280


def normalize_pid(header: str) -> str:
    token = header.strip().lstrip(">").split()[0]
    if "|" in token:
        parts = token.split("|")
        if len(parts) >= 2 and parts[1]:
            token = parts[1]
    return token.strip()


def read_fasta(path: Path) -> Dict[str, str]:
    seqs = {}
    pid = None
    chunks = []

    def flush():
        nonlocal pid, chunks
        if pid is None:
            return
        seq = re.sub(r"\s+", "", "".join(chunks)).upper()
        if not seq:
            raise ValueError(f"Empty sequence: {pid}")
        if pid in seqs:
            raise ValueError(f"Duplicate protein ID: {pid}")
        seqs[pid] = seq

    with path.open("r", encoding="utf-8-sig", errors="replace") as f:
        for line_no, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            if s.startswith(">"):
                flush()
                pid = normalize_pid(s)
                chunks = []
            else:
                if pid is None:
                    raise ValueError(f"Sequence before header at line {line_no}")
                chunks.append(s)
    flush()
    return seqs


@torch.inference_mode()
def residue_embeddings(
    sequence: str,
    tokenizer,
    model,
    device: torch.device,
    chunk_size: int,
    use_amp: bool,
) -> torch.Tensor:
    """
    Returns full-sequence residue embeddings [L, 1280] on CPU float32.

    Long sequences are processed in non-overlapping ESM2 chunks.
    """
    blocks = []

    for start in range(0, len(sequence), chunk_size):
        chunk = sequence[start:start + chunk_size]

        enc = tokenizer(
            chunk,
            return_tensors="pt",
            add_special_tokens=True,
            truncation=False,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        if use_amp and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            out = model(input_ids=input_ids, attention_mask=attention_mask)

        h = out.last_hidden_state[0]
        residue_h = h[1:1 + len(chunk)].float().cpu()

        if residue_h.shape[0] != len(chunk):
            raise RuntimeError(
                f"Residue-token mismatch: expected {len(chunk)}, "
                f"got {residue_h.shape[0]}"
            )

        blocks.append(residue_h)

    result = torch.cat(blocks, dim=0)

    if result.shape != (len(sequence), EXPECTED_DIM):
        raise RuntimeError(
            f"Unexpected residue embedding shape {tuple(result.shape)} "
            f"for sequence length {len(sequence)}"
        )

    if not torch.isfinite(result).all():
        raise FloatingPointError("NaN/Inf in residue embeddings")

    return result


def make_segment_embeddings(
    residue_h: torch.Tensor,
    window: int,
    stride: int,
    max_segments: int,
) -> Tuple[torch.Tensor, int]:
    """
    Overlapping segment mean embeddings.

    Example:
      window=32, stride=16
      1-32, 17-48, 33-64, ...

    If the number of segments exceeds max_segments, adaptive average pooling
    compresses the segment sequence while retaining coverage over the
    entire protein.
    """
    L, D = residue_h.shape

    if L <= window:
        seg = residue_h.mean(dim=0, keepdim=True)
        return seg, 1

    starts = list(range(0, L - window + 1, stride))

    tail_start = L - window
    if starts[-1] != tail_start:
        starts.append(tail_start)

    segments = torch.stack(
        [residue_h[s:s + window].mean(dim=0) for s in starts],
        dim=0,
    )
    raw_n = segments.shape[0]

    if raw_n > max_segments:
        # [N,D] -> [1,D,N] -> adaptive pool -> [max_segments,D]
        x = segments.transpose(0, 1).unsqueeze(0)
        x = F.adaptive_avg_pool1d(x, max_segments)
        segments = x.squeeze(0).transpose(0, 1).contiguous()

    return segments, raw_n


def valid_existing(path: Path, max_segments: int) -> bool:
    if not path.exists():
        return False
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
        seg = obj["segments"]
        glob = obj["global"]
        if seg.ndim != 2 or seg.shape[1] != EXPECTED_DIM:
            return False
        if seg.shape[0] < 1 or seg.shape[0] > max_segments:
            return False
        if glob.shape != (EXPECTED_DIM,):
            return False
        if not torch.isfinite(seg.float()).all():
            return False
        if not torch.isfinite(glob.float()).all():
            return False
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", default=str(DEFAULT_FASTA))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--chunk-size", type=int, default=800)
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--max-segments", type=int, default=64)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--local-files-only", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    fasta = Path(args.fasta)
    out_dir = Path(args.out_dir)

    if not fasta.exists():
        raise FileNotFoundError(fasta)
    if args.chunk_size > 1022:
        raise ValueError("--chunk-size must be <= 1022")
    if args.window <= 0 or args.stride <= 0:
        raise ValueError("window/stride must be positive")
    if args.max_segments <= 0:
        raise ValueError("max-segments must be positive")

    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    use_amp = device.type == "cuda" and not args.no_amp

    print("=" * 110)
    print("Bernett ESM2-t33 overlapping segment embedding generation")
    print("=" * 110)
    print("FASTA        :", fasta)
    print("Output       :", out_dir)
    print("Model        :", args.model)
    print("Device       :", device)
    print("chunk_size   :", args.chunk_size)
    print("window       :", args.window)
    print("stride       :", args.stride)
    print("max_segments :", args.max_segments)
    print()

    seqs = read_fasta(fasta)
    pids = sorted(seqs)

    print(f"Proteins: {len(pids):,}")
    if args.limit == 0 and len(pids) != EXPECTED_PROTEINS:
        raise RuntimeError(
            f"Expected {EXPECTED_PROTEINS:,} proteins, got {len(pids):,}"
        )

    if args.limit > 0:
        pids = pids[:args.limit]
        print(f"[DEBUG] Processing first {len(pids)} proteins only.")

    print("\nLoading ESM2...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
    )
    model = AutoModel.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
        torch_dtype=torch.float32,
    )
    model.eval().to(device)

    hidden = int(model.config.hidden_size)
    if hidden != EXPECTED_DIM:
        raise RuntimeError(
            f"Model hidden size={hidden}, expected {EXPECTED_DIM}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    failures = []
    manifest = []
    generated = 0
    skipped = 0
    t0 = time.time()

    for pid in tqdm(pids, desc="Segment embeddings", unit="protein"):
        save_path = out_dir / f"{pid}.pt"

        if not args.overwrite and valid_existing(save_path, args.max_segments):
            skipped += 1
            continue

        seq = seqs[pid]

        try:
            residue_h = residue_embeddings(
                seq,
                tokenizer,
                model,
                device,
                args.chunk_size,
                use_amp,
            )

            global_emb = residue_h.mean(dim=0)

            segments, raw_n = make_segment_embeddings(
                residue_h,
                window=args.window,
                stride=args.stride,
                max_segments=args.max_segments,
            )

            obj = {
                "segments": segments.half().contiguous(),
                "global": global_emb.half().contiguous(),
                "sequence_length": int(len(seq)),
                "n_segments_raw": int(raw_n),
                "n_segments_saved": int(segments.shape[0]),
                "window": int(args.window),
                "stride": int(args.stride),
                "max_segments": int(args.max_segments),
                "embedding_dim": EXPECTED_DIM,
                "model": args.model,
            }
            torch.save(obj, save_path)
            generated += 1

            manifest.append({
                "protein_id": pid,
                "sequence_length": len(seq),
                "n_segments_raw": raw_n,
                "n_segments_saved": segments.shape[0],
                "status": "generated",
            })

            del residue_h, global_emb, segments

        except torch.cuda.OutOfMemoryError as exc:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            failures.append({
                "protein_id": pid,
                "length": len(seq),
                "type": "CUDA_OutOfMemoryError",
                "error": str(exc),
            })

        except Exception as exc:
            failures.append({
                "protein_id": pid,
                "length": len(seq),
                "type": type(exc).__name__,
                "error": str(exc),
            })

    # Audit
    missing = []
    invalid = []
    for pid in pids:
        p = out_dir / f"{pid}.pt"
        if not p.exists():
            missing.append(pid)
        elif not valid_existing(p, args.max_segments):
            invalid.append(pid)

    report = {
        "model": args.model,
        "window": args.window,
        "stride": args.stride,
        "max_segments": args.max_segments,
        "chunk_size": args.chunk_size,
        "generated_this_run": generated,
        "skipped_this_run": skipped,
        "failures": failures,
        "audit": {
            "expected": len(pids),
            "valid": len(pids) - len(missing) - len(invalid),
            "missing": len(missing),
            "invalid": len(invalid),
            "missing_examples": missing[:50],
            "invalid_examples": invalid[:50],
        },
        "elapsed_seconds": time.time() - t0,
    }

    (out_dir / "segment_embedding_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 110)
    if not failures and not missing and not invalid:
        print("FINAL: PASS")
        print("=" * 110)
        print(f"Valid segment embeddings: {len(pids):,}")
        print("Next: run 06_train_segment_cross_ablation.py")
    else:
        print("FINAL: FAIL / INCOMPLETE")
        print("=" * 110)
        print(f"failures={len(failures)}, missing={len(missing)}, invalid={len(invalid)}")
        print("Re-run the same command to resume.")


if __name__ == "__main__":
    main()
