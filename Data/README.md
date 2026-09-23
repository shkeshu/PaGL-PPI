# Data in this release

Only the following five **pair and label** CSV files are uploaded:

| File | Rows | Meaning |
| --- | ---: | --- |
| `bernett/train_pairs.csv` | 163,192 | Bernett Gold Standard v4 Intra1, model training |
| `bernett/validation_pairs.csv` | 59,260 | Intra0, checkpoint/fusion selection |
| `bernett/test_pairs.csv` | 52,048 | Intra2, locked test |
| `external/human_test.csv` | 52,723 | Cleaned D-SCRIPT Human Full test |
| `external/yeast_test.csv` | 54,964 | Cleaned D-SCRIPT Yeast Full test |

The Bernett files have columns `protein_a,protein_b,label`. The external files also retain `row_index`. Labels are 0 or 1. No Bernett protein ID appears in more than one split. The two Human novelty subsets are **generated**, rather than uploaded, by `Scripts/01_prepare_data.py`: a pair is retained only if neither protein's exact amino acid sequence occurs in Bernett Training (NovelVsTrain) or in any Bernett split (NovelVsAll). Expected sizes are 36,845 and 13,957 pairs. The Yeast set has no exact-sequence overlap with Bernett.

### Fixed file checksums (SHA-256)

```text
bernett/train_pairs.csv       e9d71e0ba6a250bc75748c81b1f324c315f92027f29c873c2561ba39e4d76253
bernett/validation_pairs.csv  0664d15d4f634349462ead62ff2dc4989c19259262546b6fe37e89af85656465
bernett/test_pairs.csv        f76dd47225e8e191c9cde227f877e3b2381f4e4536afd19909411ad64ac00757
external/human_test.csv       a0a3297ba33f1f24c3be3a24dcd616ffa353b4af7c4e0c6081d236b832ac820d
external/yeast_test.csv       afba103ae8fcfe09d1c9d1e9970d45e3d0b21acfacbc726c70f5ced0c10ce03b
```

## Required public sequence sources

The five CSVs contain protein IDs and labels, not amino acid sequences. Download these three FASTA files before running the preparation script:

1. Bernett Gold Standard [Figshare Version 4](https://doi.org/10.6084/m9.figshare.21591618.v4): `human_swissprot.fasta` (the Version 4 file with the Q96PU5 correction). Source file SHA-256 in the recorded run: `82ddea56b8bf2b8c4a51f139bbbd32798ecdb01887a1d79d582bf69dc312c7f5`.
2. D-SCRIPT source repository: [`data/seqs/human.fasta`](https://github.com/samsledje/D-SCRIPT/blob/main/data/seqs/human.fasta).
3. D-SCRIPT source repository: [`data/seqs/yeast.fasta`](https://github.com/samsledje/D-SCRIPT/blob/main/data/seqs/yeast.fasta).

The source FASTAs are supplied by their original repositories. The preparation script checks all referenced IDs and a SHA-256 fingerprint of the **resolved ID-to-sequence mapping**. The expected mapping hashes are in `Scripts/01_prepare_data.py`; this detects changed source versions even if the upstream `main` branch changes. The pretrained ESM2 encoder is [`facebook/esm2_t33_650M_UR50D`](https://huggingface.co/facebook/esm2_t33_650M_UR50D), downloaded by the feature-generation scripts.

Example after downloading the FASTAs:

```bash
python Scripts/01_prepare_data.py \
  --bernett-fasta /path/to/human_swissprot.fasta \
  --human-fasta /path/to/human.fasta \
  --yeast-fasta /path/to/yeast.fasta
```

This creates ignored working files under `Data/processed/` and `Data/external_canonical/`. It does not change the five tracked CSVs. The original Bernett data are attributed to Judith Bernett (CC BY 4.0); D-SCRIPT is attributed to its source repository. The manuscript should cite both original datasets and identify the exact source versions used.
