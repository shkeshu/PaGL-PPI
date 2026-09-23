# Code sequence

| Step | Script | Output |
| --- | --- | --- |
| 00 | `00_check_environment.py` | Environment check |
| 01 | `01_prepare_data.py` | Working FASTAs, fixed splits, Human novelty subsets, minimal audit |
| 02 | `02_audit_canonical_dataset.py` | Bernett split counts and disjointness |
| 03 | `03_generate_esm2_embeddings.py` | Global ESM2 features |
| 04 | `04_train_strong_symmetric_ablation.py` | B0 global model and ablations |
| 05 | `05_generate_esm2_segment_embeddings.py` | Local ESM2 segment features |
| 06 | `06_train_segment_cross_ablation.py` | B1–B4 local models |
| 07 | `07_run_multiseed_confirmation.py` | Multiseed B0/B2 results |
| 08 | `08_evaluate_b0_b2_late_fusion.py` | Validation fusion selection |
| 09 | `09_run_fixed_fusion_5seed_confirmation.py` | Five-seed fixed fusion results |
| 10 | `10_evaluate_locked_bernett_test.py` | Bernett locked test metrics |
| 11 | `11_generate_dscript_external_embeddings.py` | External ESM2 features |
| 12 | `12_evaluate_locked_dscript_external_v2.py` | External B0/B2/fusion metrics |
| 13 | `13_train_esm2_global_baseline.py` | ESM2-GP reference checkpoints |
| 14 | `14_evaluate_esm2_global_baseline_test.py` | ESM2-GP Bernett metrics |
| 15 | `15_evaluate_esm2_global_external.py` | ESM2-GP external metrics |
| 16 | `16_finalize_paper_results.py` | Manuscript Tables 3–7 |

Model and utility scripts retain the architecture and hyperparameters of the original study. Only the data adapter and the compact main-table entry point were added for this five-CSV release.
