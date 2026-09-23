from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Data" / "processed" / "canonical"

FILES = {
    "train": DATA / "train_pairs.csv",
    "val": DATA / "val_pairs.csv",
    "test": DATA / "test_pairs.csv",
}

EXPECTED = {
    "train": 163192,
    "val": 59260,
    "test": 52048,
}

def read_pairs(path):
    df = pd.read_csv(path)
    required = {"protein_a", "protein_b", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    df["protein_a"] = df["protein_a"].astype(str)
    df["protein_b"] = df["protein_b"].astype(str)
    df["label"] = df["label"].astype(int)
    return df

def proteins(df):
    return set(df["protein_a"]) | set(df["protein_b"])

def canonical_pair(a, b):
    return tuple(sorted((a, b)))

def main():
    dfs = {}
    print("=" * 100)
    print("Bernett canonical dataset audit")
    print("=" * 100)

    for split, path in FILES.items():
        if not path.exists():
            print(f"[MISSING] {split}: {path}")
            continue

        df = read_pairs(path)
        dfs[split] = df
        pos = int((df.label == 1).sum())
        neg = int((df.label == 0).sum())
        pset = proteins(df)

        pair_keys = [canonical_pair(a, b) for a, b in zip(df.protein_a, df.protein_b)]
        n_dup = len(pair_keys) - len(set(pair_keys))

        print(f"[{split}]")
        print(f"  rows          : {len(df):,} (expected {EXPECTED[split]:,})")
        print(f"  positive      : {pos:,}")
        print(f"  negative      : {neg:,}")
        print(f"  proteins      : {len(pset):,}")
        print(f"  unordered dups: {n_dup:,}")

    if len(dfs) == 3:
        p_train = proteins(dfs["train"])
        p_val   = proteins(dfs["val"])
        p_test  = proteins(dfs["test"])

        tv = p_train & p_val
        tt = p_train & p_test
        vt = p_val & p_test

        print("-" * 100)
        print("Protein overlap audit")
        print(f"train ∩ val : {len(tv)}")
        print(f"train ∩ test: {len(tt)}")
        print(f"val ∩ test  : {len(vt)}")

        ok = (
            len(dfs["train"]) == EXPECTED["train"]
            and len(dfs["val"]) == EXPECTED["val"]
            and len(dfs["test"]) == EXPECTED["test"]
            and not tv and not tt and not vt
        )
        print("-" * 100)
        print("FINAL:", "PASS" if ok else "FAIL / CHECK REQUIRED")

if __name__ == "__main__":
    main()
