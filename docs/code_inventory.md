# Code inventory

This inventory describes project-owned code copied from the active `GeneJEPA-main` workspace. “Formal” means it is an entry point or dependency of a frozen experiment; it does not imply a biological claim.

| Area | Directory / entry point | Purpose | Status |
| --- | --- | --- | --- |
| GeneJEPA core | `genejepa/models.py`, `tokenizer.py` | Perceiver encoder, JEPA wiring, continuous-expression tokenizer | Formal |
| GeneJEPA data | `genejepa/data.py` | Tahoe streaming, gene mapping, sentinel and normalization path | Formal |
| GeneJEPA training | `genejepa/train.py`, `callbacks.py`, `configs.py` | Training, EMA Teacher, validation, logging, checkpointing | Formal |
| Training operations | `auto_resume_genejepa.sh` | Local GeneJEPA resume wrapper | Operational |
| Training plots | `plot_train_sim_loss.py`, `plot_val_loss.py` | Read local logs and produce diagnostic curves | Diagnostic |
| HLCA benchmarks | `benchmark_scripts/` | Frozen split, embedding extraction/merge, linear and MLP probes | Historical benchmark |
| Tahoe acquisition | `download Tahoe datasets scripts/` | Metadata audit, local download watchdog, shard verification | Operational |
| Experiment 0 | `make_tahoe_latent_audit_cohort.py`, `make_tahoe_latent_audit_cell_index_plan.py`, `extract_tahoe_latent_audit_embeddings.py`, `analyze_tahoe_latent_experiment0.py` | Frozen cohort/index construction, Epoch25 extraction, latent audit | Formal completed workflow |
| Experiment 1 manifests | `prepare_tahoe_experiment1_manifests.py`, `plan_tahoe_experiment1_full_cache.py` | Condition universe, edge split, cache population and worker plans | Formal |
| Epoch25 cache | `extract_tahoe_experiment1_full_cache_worker.py`, `merge_tahoe_experiment1_full_cache.py`, `status_tahoe_experiment1_full_cache.py`, `audit_tahoe_experiment1_cache_resume.py` | Independent resumable workers, scatter merge, status and resume audit | Formal |
| Latent DataLoader | `tahoe_experiment1_latent_data.py`, `audit_tahoe_experiment1_merged_cache_consumer.py` | Mmap cache consumer and deterministic S=256 set sampling | Formal |
| Evaluation protocol | `freeze_tahoe_experiment1_evaluation_sampling.py` | Five deterministic evaluation repeats with shared indices across models | Formal |
| B0 | `evaluate_tahoe_experiment1_b0.py` | Identity/no-change baseline | Formal |
| B1 | `audit_tahoe_experiment1_b1_coverage.py`, `build_tahoe_experiment1_b1_mean_delta.py`, `evaluate_tahoe_experiment1_b1.py` | Train-only macro condition-shift baseline | Formal |
| B2 | `train_tahoe_experiment1_b2_v2.py`, `run_tahoe_experiment1_b2_v2_formal.py`, `evaluate_tahoe_experiment1_b2_v2.py` | Pooled MLP with condition-shift MSE | Formal |
| B2 v1 | `train_tahoe_experiment1_b2.py`, `run_tahoe_experiment1_b2_formal.py` | Earlier Energy-loss implementation retained for history | Superseded |
| STATE compatibility | `smoke_tahoe_experiment1_state.py`, `patches/state_genejepa_latent_compat.patch` | Signed latent output, ST-A absolute mode, ST-R raw output-space residual | Formal dependency |
| ST training | `run_tahoe_experiment1_st_formal.py` | Two-GPU ST-A/ST-R formal training and resume | Formal |
| ST evaluation | `evaluate_tahoe_experiment1_st_raw.py`, `finalize_tahoe_experiment1.py` | Frozen test evaluation and final comparison assembly | Formal |
| Decoder data | `tahoe_decoder_v1_data.py` | Exact cached-latent/raw-count pairing and log1p(CP10K) targets | Formal |
| Decoder v1 | `freeze_genejepa_decoder_v1_contract.py`, `audit_genejepa_decoder_v1.py`, `run_genejepa_decoder_v1.py`, `run_genejepa_decoder_v1_demo.py` | Frozen 5,000-gene decoder contract, training, audit and demo | Formal |
| Gene panel audit | `audit_genejepa_decoder_gene_sparsity.py` | Train-only expression/detection statistics and panel audit | Formal support |
| HD100 | `run_genejepa_decoder_hd100.py` | High-detection Top100 decoder training | Formal |
| ARC7 | `run_genejepa_decoder_v1_arc7_100.py`, `run_genejepa_decoder_v1_arc7_abc.py`, `run_genejepa_decoder_hd100_arc7.py` | Frozen same-cell A/B/C construction and Cell-Eval metrics | Formal diagnostic |
| Author checkpoint | `audit_author_genejepa_checkpoint.py` | Official checkpoint/config alias loader and EMA-Teacher audit | Formal |
| Author cache | `author_genejepa_epoch49_hd100_cache.py` | Deterministic, resumable Author Epoch49 embedding extraction and merge | Prepared; long run paused |
| Author HD100 | `run_author_genejepa_hd100.py`, `run_author_genejepa_hd100_arc7_c.py` | Author-latent HD100 training and Decoder-only C comparison | Prepared; not formally run |
| Historical configs | `experiment_configs/` | Snapshots retained to explain earlier GeneJEPA runs | Historical |

All Python files in `perturbation_scripts/` are included. Large local inputs referenced by these scripts are intentionally not included; see `docs/provenance.md`.
