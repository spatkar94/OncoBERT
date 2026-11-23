# Ablation studies

- `rank_vs_set/`: compares ESM2-, GenePT-, and STRING-diffusion rank encoders
  with the OncoBERT set Transformer for broad/detailed GENIE retrieval and
  MSK-CHORD survival prediction.
- `embedding_dimension/`: compares 32, 64, 128, and 256 dimensional OncoBERT
  variants using three-state and binary masked reconstruction AUPRC.

Patient-bootstrap confidence intervals are used for both reconstruction tasks;
the rank/set retrieval panels use patient-cluster bootstrap confidence
intervals while scoring held-out samples.
