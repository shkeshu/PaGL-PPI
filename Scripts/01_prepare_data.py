"""Expand the five public pair CSVs using publicly sourced protein FASTAs.

The generated files use the paths expected by the original model scripts and
are ignored by Git. No sequence or label is inferred from a protein ID.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Data"
EXPECTED_ROWS = {
    "train": 163192,
    "validation": 59260,
    "test": 52048,
    "human": 52723,
    "yeast": 54964,
}
EXPECTED_MAP_HASHES = {
    "bernett": "7e6d22795c2d7f13357e4256f14e80883d1317c70ea135bad724295c18da9033",
    "human": "cf68a3a35938b05e4d3ad041d99c872c11ff38cd38625a2cd19becdb52765ea2",
    "yeast": "168b9ea6aa87b76f940463ab5076319da96959d75bcc522aa35ccd428c4d2b74",
}


def read_pairs(path: Path, expected: int) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = list(reader.fieldnames or [])
        if not {"protein_a", "protein_b", "label"}.issubset(columns):
            raise ValueError(f"Missing required pair columns: {path}")
        rows = list(reader)
    if len(rows) != expected:
        raise ValueError(f"{path}: expected {expected} pairs; found {len(rows)}")
    for row in rows:
        if row["label"] not in {"0", "1"}:
            raise ValueError(f"Invalid label in {path}: {row['label']}")
        if not row["protein_a"] or not row["protein_b"]:
            raise ValueError(f"Empty protein ID in {path}")
    return rows, columns


def read_fasta(path: Path) -> tuple[dict[str, str], dict[str, str | None]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    sequences: dict[str, str] = {}
    aliases: dict[str, str | None] = {}
    name: str | None = None
    parts: list[str] = []

    def flush() -> None:
        if name is None:
            return
        seq = "".join(parts).replace(" ", "").upper()
        if not seq:
            raise ValueError(f"Empty FASTA sequence: {name}")
        if name in sequences and sequences[name] != seq:
            raise ValueError(f"Conflicting FASTA key: {name}")
        sequences[name] = seq
        candidates = {name}
        if "|" in name:
            candidates.update(x for x in name.split("|") if x)
        for alias in candidates:
            if alias not in aliases:
                aliases[alias] = name
            elif aliases[alias] != name:
                aliases[alias] = None

    with path.open("r", encoding="utf-8-sig", errors="replace") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                name = line[1:].split()[0]
                parts = []
            else:
                if name is None:
                    raise ValueError(f"Sequence before FASTA header in {path}")
                parts.append(line)
    flush()
    return sequences, aliases


def resolve_sequences(
    path: Path, protein_ids: set[str], expected_hash: str
) -> dict[str, str]:
    sequences, aliases = read_fasta(path)
    resolved: dict[str, str] = {}
    missing: list[str] = []
    for pid in sorted(protein_ids):
        key = pid if pid in sequences else aliases.get(pid)
        if key is None:
            missing.append(pid)
        else:
            resolved[pid] = sequences[key]
    if missing:
        raise ValueError(f"{path}: missing {len(missing)} proteins; examples: {missing[:8]}")
    digest = hashlib.sha256()
    for pid, seq in sorted(resolved.items()):
        digest.update(f"{pid}\t{seq}\n".encode("ascii"))
    actual = digest.hexdigest()
    if actual != expected_hash:
        raise ValueError(
            f"Sequence mapping differs from the reported study for {path}.\n"
            f"Expected SHA-256: {expected_hash}\nObserved SHA-256: {actual}"
        )
    return resolved


def ids(rows: list[dict[str, str]]) -> set[str]:
    return {row[k] for row in rows for k in ("protein_a", "protein_b")}


def write_fasta(path: Path, sequences: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as stream:
        for pid, seq in sorted(sequences.items()):
            stream.write(f">{pid}\n")
            for offset in range(0, len(seq), 80):
                stream.write(seq[offset : offset + 80] + "\n")


def write_pairs(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bernett-fasta", type=Path, required=True)
    parser.add_argument("--human-fasta", type=Path, required=True)
    parser.add_argument("--yeast-fasta", type=Path, required=True)
    args = parser.parse_args()

    sources = {
        "train": DATA / "bernett" / "train_pairs.csv",
        "validation": DATA / "bernett" / "validation_pairs.csv",
        "test": DATA / "bernett" / "test_pairs.csv",
        "human": DATA / "external" / "human_test.csv",
        "yeast": DATA / "external" / "yeast_test.csv",
    }
    loaded = {
        key: read_pairs(path, EXPECTED_ROWS[key])
        for key, path in sources.items()
    }
    b_sets = {key: ids(loaded[key][0]) for key in ("train", "validation", "test")}
    if any(
        b_sets[a] & b_sets[b]
        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))
    ):
        raise ValueError("Bernett training, validation, and test proteins are not disjoint")

    bernett = resolve_sequences(
        args.bernett_fasta,
        set.union(*b_sets.values()),
        EXPECTED_MAP_HASHES["bernett"],
    )
    human = resolve_sequences(
        args.human_fasta, ids(loaded["human"][0]), EXPECTED_MAP_HASHES["human"]
    )
    yeast = resolve_sequences(
        args.yeast_fasta, ids(loaded["yeast"][0]), EXPECTED_MAP_HASHES["yeast"]
    )

    canonical = DATA / "processed" / "canonical"
    for key, output_name in (
        ("train", "train_pairs.csv"),
        ("validation", "val_pairs.csv"),
        ("test", "test_pairs.csv"),
    ):
        destination = canonical / output_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(sources[key], destination)
    write_fasta(canonical / "proteins.fasta", bernett)

    external = DATA / "external_canonical"
    for key, folder, mapping in (
        ("human", "D-SCRIPT_Human", human),
        ("yeast", "D-SCRIPT_Yeast", yeast),
    ):
        output = external / folder
        output.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(sources[key], output / "test_pairs.csv")
        write_fasta(output / "proteins.fasta", mapping)

    train_hashes = {
        hashlib.sha256(bernett[pid].encode("ascii")).hexdigest()
        for pid in b_sets["train"]
    }
    all_hashes = {
        hashlib.sha256(seq.encode("ascii")).hexdigest() for seq in bernett.values()
    }
    human_rows, human_columns = loaded["human"]

    def novel(rows: list[dict[str, str]], blocked: set[str]) -> list[dict[str, str]]:
        return [
            row
            for row in rows
            if all(
                hashlib.sha256(human[row[k]].encode("ascii")).hexdigest() not in blocked
                for k in ("protein_a", "protein_b")
            )
        ]

    novel_train = novel(human_rows, train_hashes)
    novel_all = novel(human_rows, all_hashes)
    if (len(novel_train), len(novel_all)) != (36845, 13957):
        raise ValueError(
            f"Human novelty subset sizes differ: {len(novel_train)}, {len(novel_all)}"
        )
    human_output = external / "D-SCRIPT_Human"
    write_pairs(
        human_output / "test_pairs_novel_vs_bernett_train.csv",
        novel_train,
        human_columns,
    )
    write_pairs(
        human_output / "test_pairs_novel_vs_bernett_all.csv",
        novel_all,
        human_columns,
    )

    yeast_hashes = {
        hashlib.sha256(seq.encode("ascii")).hexdigest() for seq in yeast.values()
    }
    if yeast_hashes & all_hashes:
        raise ValueError("Yeast and Bernett contain exact-sequence overlap")

    report = {
        "status": "PASS",
        "checks": {"total": 6, "passed": 6, "failed": 0},
        "bernett_protein_disjoint": True,
        "yeast": {"cross_species_zero_shot_eligible": True},
        "human_novel_vs_train_pairs": len(novel_train),
        "human_novel_vs_all_pairs": len(novel_all),
        "note": "Minimal audit of the five-CSV release; not the original 113-check audit",
    }
    audit_path = (
        ROOT / "Results" / "second_round" / "dataset_audit" / "dataset_audit_summary.json"
    )
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Five input CSVs and three FASTAs: PASS")
    print("Bernett split: 163,192 / 59,260 / 52,048; protein-disjoint: PASS")
    print("Human Full / NovelVsTrain / NovelVsAll: 52,723 / 36,845 / 13,957")
    print("Yeast Full: 54,964; exact-sequence overlap with Bernett: 0")
    print("Generated working data under Data/processed and Data/external_canonical")


if __name__ == "__main__":
    main()
