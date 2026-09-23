# Ordered reproduction workflow

Run all commands from the repository root. Python 3.10 and a PyTorch build compatible with the local GPU are recommended. Install the dependencies in `requirements.txt`. The original recorded protocol uses seeds 42, 123, 456, 789, and 1234, a locked Bernett Intra2 test, and fixed 0.70 B2 / 0.30 B0 logit fusion.

## Prepare the five datasets

Download the FASTA files listed in `Data/README.md`, then run:

```bash
python Scripts/00_check_environment.py
python Scripts/01_prepare_data.py --bernett-fasta /path/to/human_swissprot.fasta --human-fasta /path/to/human.fasta --yeast-fasta /path/to/yeast.fasta
python Scripts/02_audit_canonical_dataset.py
```

Step 01 checks sequence fingerprints and creates ignored working FASTAs, copied split files, and exact-sequence Human novelty subsets. Its minimal audit is a new six-check gate for this compact package. It does not reproduce the original full 113-check diagnostic audit.

## Train and evaluate the Bernett model

```bash
python Scripts/03_generate_esm2_embeddings.py
python Scripts/04_train_strong_symmetric_ablation.py
python Scripts/05_generate_esm2_segment_embeddings.py
python Scripts/06_train_segment_cross_ablation.py
python Scripts/07_run_multiseed_confirmation.py
python Scripts/08_evaluate_b0_b2_late_fusion.py
python Scripts/09_run_fixed_fusion_5seed_confirmation.py
python Scripts/10_evaluate_locked_bernett_test.py
```

The feature generators download the pretrained ESM2 encoder. The training steps create checkpoints and validation predictions under ignored `Results/` paths. Steps 08–09 freeze the fusion weight before Step 10 reads the locked test.

## External evaluation and ESM2-GP reference

```bash
python Scripts/11_generate_dscript_external_embeddings.py
python Scripts/12_evaluate_locked_dscript_external_v2.py
python Scripts/13_train_esm2_global_baseline.py
python Scripts/14_evaluate_esm2_global_baseline_test.py
python Scripts/15_evaluate_esm2_global_external.py
```

External scripts use the generated Human Full, NovelVsTrain, NovelVsAll, and Yeast Full working sets. ESM2-GP is a later reference baseline; it was not used to select the original PaGL-PPI model.

## Calculate the manuscript tables

```bash
python Scripts/16_finalize_paper_results.py
```

Step 16 reads the newly produced five-seed summaries under `Results/` and writes Tables 3–7 to `Results/paper_results_final_v2/tables/`. The five archived table CSVs are included as small reference outputs. Literature comparison values in Table 3 are reference values and may follow different protocols. The package does not contain pretrained study checkpoints or archived prediction tables, so the table command runs only after the preceding training and evaluation steps have produced their summaries.
