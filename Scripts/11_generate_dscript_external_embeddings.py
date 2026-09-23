from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

CANONICAL_ROOT = ROOT / "Data" / "external_canonical"
OUT_ROOT = ROOT / "Data" / "embeddings_external"

BERNETT_FASTA = (
    ROOT
    / "Data"
    / "processed"
    / "canonical"
    / "proteins.fasta"
)

BERNETT_GLOBAL = (
    ROOT
    / "Data"
    / "embeddings"
    / "ESM2_t33"
)

BERNETT_SEGMENT = (
    ROOT
    / "Data"
    / "embeddings"
    / "ESM2_t33_segments"
)

MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
EMBED_DIM = 1280


def file_key(pid: str) -> str:
    return hashlib.sha1(
        pid.encode("utf-8")
    ).hexdigest()


def seq_hash(seq: str) -> str:
    return hashlib.sha256(
        seq.encode("ascii")
    ).hexdigest()


def read_fasta(path: Path) -> Dict[str, str]:
    seqs = {}
    current = None
    chunks = []

    def flush():
        nonlocal current, chunks
        if current is None:
            return
        seq = re.sub(
            r"\s+",
            "",
            "".join(chunks),
        ).upper()
        if current in seqs:
            raise ValueError(
                f"Duplicate FASTA key: {current}"
            )
        seqs[current] = seq

    with path.open(
        "r",
        encoding="utf-8-sig",
        errors="replace",
    ) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue

            if s.startswith(">"):
                flush()
                current = s[1:].split()[0]
                chunks = []
            else:
                chunks.append(s)

    flush()
    return seqs


def build_bernett_hash_map():
    seqs = read_fasta(
        BERNETT_FASTA
    )

    d = {}

    for pid, seq in seqs.items():
        d.setdefault(
            seq_hash(seq),
            [],
        ).append(pid)

    return d


def valid_global(path: Path) -> bool:
    if not path.exists():
        return False

    try:
        x = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )

        x = torch.as_tensor(
            x
        ).float()

        return (
            x.shape == (EMBED_DIM,)
            and torch.isfinite(x).all()
        )
    except Exception:
        return False


def valid_segment(
    path: Path,
    max_segments: int,
) -> bool:
    if not path.exists():
        return False

    try:
        obj = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )

        seg = obj["segments"]
        glob = obj["global"]

        return (
            seg.ndim == 2
            and seg.shape[1] == EMBED_DIM
            and 1 <= seg.shape[0] <= max_segments
            and tuple(glob.shape) == (EMBED_DIM,)
            and torch.isfinite(seg.float()).all()
            and torch.isfinite(glob.float()).all()
        )
    except Exception:
        return False


@torch.inference_mode()
def residue_embeddings(
    sequence: str,
    tokenizer,
    model,
    device,
    chunk_size: int,
    use_amp: bool,
):
    blocks = []

    for start in range(
        0,
        len(sequence),
        chunk_size,
    ):
        chunk = sequence[
            start:start + chunk_size
        ]

        enc = tokenizer(
            chunk,
            return_tensors="pt",
            add_special_tokens=True,
            truncation=False,
        )

        input_ids = enc[
            "input_ids"
        ].to(device)

        attention_mask = enc[
            "attention_mask"
        ].to(device)

        if (
            use_amp
            and device.type == "cuda"
        ):
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
        else:
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        h = out.last_hidden_state[0]

        residue_h = h[
            1:1 + len(chunk)
        ].float().cpu()

        if residue_h.shape[0] != len(chunk):
            raise RuntimeError(
                f"Residue-token mismatch: "
                f"expected {len(chunk)}, got {residue_h.shape[0]}"
            )

        blocks.append(
            residue_h
        )

    result = torch.cat(
        blocks,
        dim=0,
    )

    if result.shape != (
        len(sequence),
        EMBED_DIM,
    ):
        raise RuntimeError(
            f"Unexpected residue embedding shape "
            f"{tuple(result.shape)}"
        )

    return result


def make_segments(
    residue_h: torch.Tensor,
    window: int,
    stride: int,
    max_segments: int,
):
    L, D = residue_h.shape

    if L <= window:
        return (
            residue_h.mean(
                dim=0,
                keepdim=True,
            ),
            1,
        )

    starts = list(
        range(
            0,
            L - window + 1,
            stride,
        )
    )

    tail_start = L - window

    if starts[-1] != tail_start:
        starts.append(
            tail_start
        )

    seg = torch.stack(
        [
            residue_h[
                s:s + window
            ].mean(
                dim=0
            )
            for s in starts
        ],
        dim=0,
    )

    raw_n = seg.shape[0]

    if raw_n > max_segments:
        x = (
            seg
            .transpose(0, 1)
            .unsqueeze(0)
        )

        x = F.adaptive_avg_pool1d(
            x,
            max_segments,
        )

        seg = (
            x
            .squeeze(0)
            .transpose(0, 1)
            .contiguous()
        )

    return seg, raw_n


def find_reusable_bernett(
    seq: str,
    bernett_hash_map,
    max_segments: int,
):
    candidates = bernett_hash_map.get(
        seq_hash(seq),
        [],
    )

    for pid in candidates:
        gp = (
            BERNETT_GLOBAL
            / f"{pid}.pt"
        )

        sp = (
            BERNETT_SEGMENT
            / f"{pid}.pt"
        )

        if (
            valid_global(gp)
            and valid_segment(
                sp,
                max_segments,
            )
        ):
            return pid, gp, sp

    return None


def process_dataset(
    species: str,
    tokenizer,
    model,
    device,
    args,
    bernett_hash_map,
):
    canonical_dir = (
        CANONICAL_ROOT
        / f"D-SCRIPT_{species}"
    )

    fasta = (
        canonical_dir
        / "proteins.fasta"
    )

    if not fasta.exists():
        raise FileNotFoundError(
            fasta
        )

    seqs = read_fasta(
        fasta
    )

    out_dir = (
        OUT_ROOT
        / f"D-SCRIPT_{species}"
    )

    global_dir = (
        out_dir
        / "global"
    )

    segment_dir = (
        out_dir
        / "segments"
    )

    global_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    segment_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_rows = []
    failures = []

    generated = 0
    reused = 0
    skipped = 0

    pids = sorted(
        seqs
    )

    if args.limit > 0:
        pids = pids[
            :args.limit
        ]

    print()
    print("=" * 110)
    print(
        f"Embedding D-SCRIPT {species}: "
        f"{len(pids):,} proteins"
    )
    print("=" * 110)

    for pid in tqdm(
        pids,
        desc=f"{species} embeddings",
        unit="protein",
    ):
        key = file_key(pid)

        gp = (
            global_dir
            / f"{key}.pt"
        )

        sp = (
            segment_dir
            / f"{key}.pt"
        )

        seq = seqs[pid]

        if (
            not args.overwrite
            and valid_global(gp)
            and valid_segment(
                sp,
                args.max_segments,
            )
        ):
            skipped += 1

            manifest_rows.append({
                "protein_id": pid,
                "file_key": key,
                "sequence_length": len(seq),
                "sequence_sha256": seq_hash(seq),
                "source": "existing",
            })

            continue

        try:
            reuse_info = None

            if not args.no_reuse_bernett:
                reuse_info = find_reusable_bernett(
                    seq,
                    bernett_hash_map,
                    args.max_segments,
                )

            if reuse_info is not None:
                bernett_pid, bgp, bsp = (
                    reuse_info
                )

                gx = torch.load(
                    bgp,
                    map_location="cpu",
                    weights_only=True,
                )

                sx = torch.load(
                    bsp,
                    map_location="cpu",
                    weights_only=True,
                )

                torch.save(
                    torch.as_tensor(
                        gx
                    ).float().contiguous(),
                    gp,
                )

                torch.save(
                    sx,
                    sp,
                )

                reused += 1
                source = (
                    f"reused_bernett:{bernett_pid}"
                )

            else:
                if model is None:
                    raise RuntimeError(
                        "ESM2 model was not loaded "
                        "but generation is required."
                    )

                residue_h = residue_embeddings(
                    seq,
                    tokenizer,
                    model,
                    device,
                    args.chunk_size,
                    not args.no_amp,
                )

                global_emb = residue_h.mean(
                    dim=0
                )

                segments, raw_n = make_segments(
                    residue_h,
                    args.window,
                    args.stride,
                    args.max_segments,
                )

                torch.save(
                    global_emb.float().contiguous(),
                    gp,
                )

                torch.save(
                    {
                        "segments": segments.half().contiguous(),
                        "global": global_emb.half().contiguous(),
                        "sequence_length": len(seq),
                        "n_segments_raw": int(raw_n),
                        "n_segments_saved": int(
                            segments.shape[0]
                        ),
                        "window": args.window,
                        "stride": args.stride,
                        "max_segments": args.max_segments,
                        "embedding_dim": EMBED_DIM,
                        "model": args.model,
                    },
                    sp,
                )

                generated += 1
                source = "generated"

                del (
                    residue_h,
                    global_emb,
                    segments,
                )

            manifest_rows.append({
                "protein_id": pid,
                "file_key": key,
                "sequence_length": len(seq),
                "sequence_sha256": seq_hash(seq),
                "source": source,
            })

        except torch.cuda.OutOfMemoryError as exc:
            if device.type == "cuda":
                torch.cuda.empty_cache()

            failures.append({
                "protein_id": pid,
                "type": "CUDA_OutOfMemoryError",
                "error": str(exc),
            })

        except Exception as exc:
            failures.append({
                "protein_id": pid,
                "type": type(exc).__name__,
                "error": str(exc),
            })

    import pandas as pd

    manifest = pd.DataFrame(
        manifest_rows
    )

    manifest.to_csv(
        out_dir
        / "embedding_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )

    valid_count = 0
    missing = []

    for pid in pids:
        key = file_key(pid)

        gp = (
            global_dir
            / f"{key}.pt"
        )

        sp = (
            segment_dir
            / f"{key}.pt"
        )

        if (
            valid_global(gp)
            and valid_segment(
                sp,
                args.max_segments,
            )
        ):
            valid_count += 1
        else:
            missing.append(
                pid
            )

    report = {
        "dataset": f"D-SCRIPT_{species}",
        "requested_proteins": len(pids),
        "valid_embeddings": valid_count,
        "generated_this_run": generated,
        "reused_from_bernett": reused,
        "skipped_existing": skipped,
        "failures": failures,
        "missing_or_invalid": missing,
        "model": args.model,
        "window": args.window,
        "stride": args.stride,
        "max_segments": args.max_segments,
        "chunk_size": args.chunk_size,
    }

    (
        out_dir
        / "embedding_report.json"
    ).write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        )
    )

    if valid_count != len(pids):
        raise RuntimeError(
            f"{species}: external embedding generation incomplete. "
            f"Re-run the same command to resume."
        )


def generation_needed(
    species: str,
    max_segments: int,
):
    fasta = (
        CANONICAL_ROOT
        / f"D-SCRIPT_{species}"
        / "proteins.fasta"
    )

    seqs = read_fasta(
        fasta
    )

    out_dir = (
        OUT_ROOT
        / f"D-SCRIPT_{species}"
    )

    for pid in seqs:
        key = file_key(pid)

        if not (
            valid_global(
                out_dir
                / "global"
                / f"{key}.pt"
            )
            and valid_segment(
                out_dir
                / "segments"
                / f"{key}.pt",
                max_segments,
            )
        ):
            return True

    return False


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--dataset",
        choices=[
            "all",
            "Human",
            "Yeast",
        ],
        default="all",
    )

    ap.add_argument(
        "--model",
        default=MODEL_NAME,
    )

    ap.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
    )

    ap.add_argument(
        "--chunk-size",
        type=int,
        default=800,
    )

    ap.add_argument(
        "--window",
        type=int,
        default=32,
    )

    ap.add_argument(
        "--stride",
        type=int,
        default=16,
    )

    ap.add_argument(
        "--max-segments",
        type=int,
        default=64,
    )

    ap.add_argument(
        "--no-amp",
        action="store_true",
    )

    ap.add_argument(
        "--overwrite",
        action="store_true",
    )

    ap.add_argument(
        "--no-reuse-bernett",
        action="store_true",
    )

    ap.add_argument(
        "--local-files-only",
        action="store_true",
    )

    ap.add_argument(
        "--limit",
        type=int,
        default=0,
    )

    args = ap.parse_args()

    if args.chunk_size > 1022:
        raise ValueError(
            "--chunk-size must be <=1022"
        )

    datasets = (
        ["Human", "Yeast"]
        if args.dataset == "all"
        else [args.dataset]
    )

    bernett_hash_map = (
        build_bernett_hash_map()
    )

    # Reuse Bernett features first. Load ESM2 only if generation is required.
    need_model = (
        args.overwrite
        or any(
            generation_needed(
                species,
                args.max_segments,
            )
            for species in datasets
        )
    )

    device = torch.device(
        "cuda"
        if args.device == "cuda"
        and torch.cuda.is_available()
        else "cpu"
    )

    tokenizer = None
    model = None

    if need_model:
        print(
            "Loading locked ESM2-t33 model..."
        )

        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            local_files_only=args.local_files_only,
        )

        model = AutoModel.from_pretrained(
            args.model,
            local_files_only=args.local_files_only,
            torch_dtype=torch.float32,
        )

        model.eval().to(
            device
        )

        if int(
            model.config.hidden_size
        ) != EMBED_DIM:
            raise RuntimeError(
                "Unexpected ESM2 embedding dimension"
            )

    for species in datasets:
        process_dataset(
            species,
            tokenizer,
            model,
            device,
            args,
            bernett_hash_map,
        )

    print()
    print("=" * 110)
    print("STEP 14 FINAL: PASS")
    print("=" * 110)
    print(
        "Next: run 12_evaluate_locked_dscript_external_v2.py"
    )


if __name__ == "__main__":
    main()
