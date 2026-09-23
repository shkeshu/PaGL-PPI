from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


# =============================================================================
# Project paths
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

DEFAULT_FASTA = ROOT / "Data" / "processed" / "canonical" / "proteins.fasta"
DEFAULT_OUT_DIR = ROOT / "Data" / "embeddings" / "ESM2_t33"
DEFAULT_MODEL = "facebook/esm2_t33_650M_UR50D"

EXPECTED_PROTEINS = 11_019
EXPECTED_DIM = 1280


# =============================================================================
# FASTA
# =============================================================================

def normalize_pid(header: str) -> str:
    token = header.strip().lstrip(">").split()[0]
    if "|" in token:
        parts = token.split("|")
        if len(parts) >= 2 and parts[1]:
            token = parts[1]
    return token.strip()


def read_fasta(path: Path) -> Dict[str, str]:
    seqs: Dict[str, str] = {}
    current_id = None
    chunks: List[str] = []

    def flush():
        nonlocal current_id, chunks
        if current_id is None:
            return
        seq = "".join(chunks)
        seq = re.sub(r"\s+", "", seq).upper()
        if not seq:
            raise ValueError(f"Empty sequence: {current_id}")
        if current_id in seqs:
            raise ValueError(f"Duplicate FASTA protein ID: {current_id}")
        seqs[current_id] = seq

    with path.open("r", encoding="utf-8-sig", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            if s.startswith(">"):
                flush()
                current_id = normalize_pid(s)
                chunks = []
                if not current_id:
                    raise ValueError(f"Invalid FASTA header at line {line_no}")
            else:
                if current_id is None:
                    raise ValueError(
                        f"Sequence appears before FASTA header at line {line_no}"
                    )
                chunks.append(s)
    flush()

    return seqs


# =============================================================================
# Embedding helpers
# =============================================================================

def split_non_overlapping(sequence: str, chunk_size: int) -> List[str]:
    """
    Split long proteins into non-overlapping chunks.

    We use weighted mean pooling across chunks, so every residue contributes
    exactly once to the final protein-level vector.
    """
    return [
        sequence[i:i + chunk_size]
        for i in range(0, len(sequence), chunk_size)
    ]


def validate_existing_embedding(path: Path, expected_dim: int) -> bool:
    if not path.exists():
        return False

    try:
        x = torch.load(path, map_location="cpu")
        if isinstance(x, dict):
            for key in ("embedding", "emb", "x", "representations"):
                if key in x:
                    x = x[key]
                    break

        x = torch.as_tensor(x)

        if x.ndim != 1:
            return False
        if x.numel() != expected_dim:
            return False
        if not torch.isfinite(x.float()).all():
            return False
        return True

    except Exception:
        return False


@torch.inference_mode()
def embed_chunk(
    sequence: str,
    tokenizer,
    model,
    device: torch.device,
    use_amp: bool,
) -> torch.Tensor:
    """
    Return one mean-pooled [D] vector for a single sequence chunk.
    Excludes BOS/EOS special tokens.
    """
    encoded = tokenizer(
        sequence,
        return_tensors="pt",
        add_special_tokens=True,
        truncation=False,
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    if use_amp and device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
    else:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    h = outputs.last_hidden_state[0]

    # ESM tokenizer: [CLS] + one token per residue + [EOS].
    # We explicitly select the residue positions.
    residue_h = h[1:1 + len(sequence)]

    if residue_h.shape[0] != len(sequence):
        raise RuntimeError(
            f"Tokenizer/model residue-token mismatch: "
            f"sequence length={len(sequence)}, "
            f"residue hidden length={residue_h.shape[0]}"
        )

    pooled = residue_h.float().mean(dim=0).cpu()

    if not torch.isfinite(pooled).all():
        raise FloatingPointError("Non-finite values in pooled embedding")

    return pooled


@torch.inference_mode()
def embed_protein(
    sequence: str,
    tokenizer,
    model,
    device: torch.device,
    chunk_size: int,
    use_amp: bool,
) -> Tuple[torch.Tensor, int]:
    chunks = split_non_overlapping(sequence, chunk_size)

    weighted_sum = None
    total_residues = 0

    for chunk in chunks:
        emb = embed_chunk(
            chunk,
            tokenizer=tokenizer,
            model=model,
            device=device,
            use_amp=use_amp,
        )

        weight = len(chunk)

        if weighted_sum is None:
            weighted_sum = emb * weight
        else:
            weighted_sum += emb * weight

        total_residues += weight

    protein_emb = weighted_sum / total_residues

    return protein_emb, len(chunks)


# =============================================================================
# Audit
# =============================================================================

def audit_output(
    protein_ids: List[str],
    out_dir: Path,
    expected_dim: int,
) -> dict:
    missing = []
    invalid = []

    for pid in protein_ids:
        path = out_dir / f"{pid}.pt"

        if not path.exists():
            missing.append(pid)
            continue

        if not validate_existing_embedding(path, expected_dim):
            invalid.append(pid)

    return {
        "expected_proteins": len(protein_ids),
        "valid_embeddings": len(protein_ids) - len(missing) - len(invalid),
        "missing_embeddings": len(missing),
        "invalid_embeddings": len(invalid),
        "missing_examples": missing[:50],
        "invalid_examples": invalid[:50],
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate mean-pooled ESM2-t33 protein embeddings for Bernett."
    )

    parser.add_argument(
        "--fasta",
        type=str,
        default=str(DEFAULT_FASTA),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(DEFAULT_OUT_DIR),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Hugging Face model name or local model directory.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=800,
        help="Residues per ESM2 chunk. 800 is conservative and below model limit.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable CUDA FP16 autocast.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate valid existing .pt files.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Do not access Hugging Face network; require locally cached model.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Debug only: process first N proteins; 0 means all proteins.",
    )

    args = parser.parse_args()

    fasta_path = Path(args.fasta)
    out_dir = Path(args.out_dir)

    if not fasta_path.exists():
        raise FileNotFoundError(
            f"Canonical FASTA not found:\n{fasta_path}\n"
            f"Run 01_prepare_data.py first."
        )

    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be > 0")

    # ESM2 has a ~1022 residue usable sequence window.
    # Keep a hard safety cap here.
    if args.chunk_size > 1022:
        raise ValueError(
            f"--chunk-size={args.chunk_size} is too large for ESM2. "
            f"Use <= 1022; recommended 800."
        )

    device = torch.device(
        "cuda"
        if args.device == "cuda" and torch.cuda.is_available()
        else "cpu"
    )

    use_amp = (device.type == "cuda") and (not args.no_amp)

    print("=" * 110)
    print("Bernett ESM2-t33 protein embedding generation")
    print("=" * 110)
    print("Project root     :", ROOT)
    print("FASTA            :", fasta_path)
    print("Output directory :", out_dir)
    print("Model            :", args.model)
    print("Device           :", device)
    print("AMP              :", use_amp)
    print("Chunk size       :", args.chunk_size)
    print("Overwrite        :", args.overwrite)
    print("Local files only :", args.local_files_only)
    print()

    # -------------------------------------------------------------------------
    # 1. Read canonical FASTA
    # -------------------------------------------------------------------------
    print("[1/5] Reading canonical FASTA...")
    sequences = read_fasta(fasta_path)
    protein_ids = sorted(sequences.keys())

    print(f"  proteins: {len(protein_ids):,}")

    if len(protein_ids) != EXPECTED_PROTEINS and args.limit == 0:
        raise RuntimeError(
            f"Canonical FASTA has {len(protein_ids):,} proteins, "
            f"expected {EXPECTED_PROTEINS:,}. "
            f"Do not continue until the canonical data are correct."
        )

    lengths = [len(sequences[pid]) for pid in protein_ids]

    print(f"  min length : {min(lengths):,}")
    print(f"  max length : {max(lengths):,}")
    print(f"  mean length: {sum(lengths) / len(lengths):.2f}")
    print(
        f"  > chunk size ({args.chunk_size}) : "
        f"{sum(x > args.chunk_size for x in lengths):,}"
    )
    print()

    if args.limit > 0:
        protein_ids = protein_ids[:args.limit]
        print(f"[DEBUG] --limit={args.limit}; processing only {len(protein_ids)} proteins")
        print()

    # -------------------------------------------------------------------------
    # 2. Load ESM2
    # -------------------------------------------------------------------------
    print("[2/5] Loading tokenizer and model...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            local_files_only=args.local_files_only,
        )

        model = AutoModel.from_pretrained(
            args.model,
            local_files_only=args.local_files_only,
            torch_dtype=torch.float32,
        )

    except Exception as exc:
        print()
        print("[ERROR] Failed to load ESM2 model.")
        print("Possible reasons:")
        print("  1. First run has no Internet connection.")
        print("  2. Hugging Face model is not cached locally.")
        print("  3. transformers version is too old.")
        print("  4. --model local path is wrong.")
        print()
        raise

    model.eval()
    model.to(device)

    hidden_dim = int(getattr(model.config, "hidden_size", -1))

    print(f"  hidden size: {hidden_dim}")

    if hidden_dim != EXPECTED_DIM:
        raise RuntimeError(
            f"Model hidden size is {hidden_dim}, expected {EXPECTED_DIM}. "
            f"This script is intended for ESM2-t33-650M."
        )

    if device.type == "cuda":
        print(f"  GPU        : {torch.cuda.get_device_name(0)}")
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        print(f"  CUDA memory after model load: allocated={allocated:.2f} GB, reserved={reserved:.2f} GB")
    print()

    # -------------------------------------------------------------------------
    # 3. Generate embeddings
    # -------------------------------------------------------------------------
    print("[3/5] Generating protein embeddings...")
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "embedding_manifest.csv"
    failures_path = out_dir / "embedding_failures.json"

    rows = []
    failures = []

    n_skipped = 0
    n_generated = 0
    t0 = time.time()

    for pid in tqdm(protein_ids, desc="ESM2 proteins", unit="protein"):
        seq = sequences[pid]
        emb_path = out_dir / f"{pid}.pt"

        if (not args.overwrite) and validate_existing_embedding(
            emb_path, EXPECTED_DIM
        ):
            n_skipped += 1
            rows.append({
                "protein_id": pid,
                "sequence_length": len(seq),
                "chunks": math.ceil(len(seq) / args.chunk_size),
                "embedding_dim": EXPECTED_DIM,
                "status": "skipped_valid_existing",
                "file": emb_path.name,
            })
            continue

        try:
            emb, n_chunks = embed_protein(
                sequence=seq,
                tokenizer=tokenizer,
                model=model,
                device=device,
                chunk_size=args.chunk_size,
                use_amp=use_amp,
            )

            if emb.ndim != 1 or emb.numel() != EXPECTED_DIM:
                raise RuntimeError(
                    f"Unexpected final embedding shape: {tuple(emb.shape)}"
                )

            if not torch.isfinite(emb).all():
                raise FloatingPointError(
                    "Non-finite final protein embedding"
                )

            # Save float32 CPU vector [1280].
            torch.save(emb.contiguous().float().cpu(), emb_path)

            n_generated += 1

            rows.append({
                "protein_id": pid,
                "sequence_length": len(seq),
                "chunks": n_chunks,
                "embedding_dim": int(emb.numel()),
                "status": "generated",
                "file": emb_path.name,
            })

        except torch.cuda.OutOfMemoryError as exc:
            if device.type == "cuda":
                torch.cuda.empty_cache()

            failure = {
                "protein_id": pid,
                "sequence_length": len(seq),
                "error_type": "CUDA_OutOfMemoryError",
                "error": str(exc),
            }
            failures.append(failure)

            rows.append({
                "protein_id": pid,
                "sequence_length": len(seq),
                "chunks": math.ceil(len(seq) / args.chunk_size),
                "embedding_dim": "",
                "status": "FAILED_CUDA_OOM",
                "file": emb_path.name,
            })

            print()
            print(f"[OOM] {pid}, length={len(seq)}")
            print(
                "Re-run with a smaller chunk, for example:\n"
                "  --chunk-size 600"
            )

        except Exception as exc:
            failure = {
                "protein_id": pid,
                "sequence_length": len(seq),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            failures.append(failure)

            rows.append({
                "protein_id": pid,
                "sequence_length": len(seq),
                "chunks": math.ceil(len(seq) / args.chunk_size),
                "embedding_dim": "",
                "status": "FAILED",
                "file": emb_path.name,
            })

            print()
            print(
                f"[FAILED] {pid}, length={len(seq)}, "
                f"{type(exc).__name__}: {exc}"
            )

    elapsed = time.time() - t0

    with manifest_path.open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "protein_id",
            "sequence_length",
            "chunks",
            "embedding_dim",
            "status",
            "file",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    failures_path.write_text(
        json.dumps(failures, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print(f"  generated: {n_generated:,}")
    print(f"  skipped  : {n_skipped:,}")
    print(f"  failed   : {len(failures):,}")
    print(f"  elapsed  : {elapsed / 60:.1f} min")
    print()

    # -------------------------------------------------------------------------
    # 4. Audit all requested proteins
    # -------------------------------------------------------------------------
    print("[4/5] Auditing generated embeddings...")
    audit = audit_output(
        protein_ids=protein_ids,
        out_dir=out_dir,
        expected_dim=EXPECTED_DIM,
    )

    print(f"  expected : {audit['expected_proteins']:,}")
    print(f"  valid    : {audit['valid_embeddings']:,}")
    print(f"  missing  : {audit['missing_embeddings']:,}")
    print(f"  invalid  : {audit['invalid_embeddings']:,}")
    print()

    # -------------------------------------------------------------------------
    # 5. Report
    # -------------------------------------------------------------------------
    report = {
        "model": args.model,
        "expected_embedding_dim": EXPECTED_DIM,
        "chunk_size": args.chunk_size,
        "pooling": "non-overlapping chunk weighted residue mean",
        "device": str(device),
        "amp": bool(use_amp),
        "overwrite": bool(args.overwrite),
        "debug_limit": int(args.limit),
        "generated_this_run": n_generated,
        "skipped_valid_existing_this_run": n_skipped,
        "failures_this_run": failures,
        "audit": audit,
        "elapsed_seconds": elapsed,
    }

    report_path = out_dir / "embedding_build_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("[5/5] Final status")

    if (
        audit["missing_embeddings"] == 0
        and audit["invalid_embeddings"] == 0
        and len(failures) == 0
    ):
        print("=" * 110)
        print("FINAL: PASS")
        print("=" * 110)
        print(f"All {audit['valid_embeddings']:,} embeddings are valid.")
        print(f"Embedding dimension: {EXPECTED_DIM}")
        print(f"Output directory: {out_dir}")
        print()
        print("Next command:")
        print("  python Scripts/04_train_strong_symmetric_ablation.py")
    else:
        print("=" * 110)
        print("FINAL: FAIL / INCOMPLETE")
        print("=" * 110)
        print("Do NOT start model training yet.")
        print("Check:")
        print(" ", failures_path)
        print(" ", report_path)
        print()
        print("The script supports resume: simply run it again.")
        print("Valid existing embeddings will be skipped automatically.")


if __name__ == "__main__":
    main()
