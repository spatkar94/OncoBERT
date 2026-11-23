# Data layout

Only small, redistributable reference assets are committed. Raw cohorts,
patient-level representations, and intermediate training checkpoints are
excluded because they are large or governed by source-specific terms. The
single final manuscript checkpoint is included under `saved_checkpoints/`.

`shared200_genes.txt` is the ordered vocabulary expected by the included
three-state Set Transformer checkpoint. State matrices supplied to the core
training and inference scripts must use these columns in this exact order and
encode `0=WT`, `1=mutated-not-LOF`, and `2=LOF`.

```text
data/
  raw/                 # source cohort downloads (gitignored)
  prepared/            # shared-200 matrices and fixed masks (gitignored)
  derived/             # embeddings and analysis intermediates (gitignored)
  reference_networks/  # small redistributable graph assets
  manifests/           # provenance for committed reference assets
  example_outputs/     # final figure snapshots for regression checks
```

Download the public cBioPortal cohorts with:

```bash
python data/download_cbioportal.py --extract
```

By default this downloads every study in `public_studies.txt`, including
MSK-CHORD and the public immunotherapy validation studies. Pass
`--studies data/immunotherapy_studies.txt` to retrieve only the response and
external-validation collection.

AACR-GENIE and TCGA should be downloaded from the source pages recorded in
`datasets.json`. The analysis scripts accept these environment roots:

```bash
export ONCOBERT_WORK_ROOT=/path/to/analysis_workspace
export ONCOBERT_EXPERIMENT_ROOT=/path/to/OncoBERT_experiments
export ONCOBERT_GENIE_ROOT=/path/to/AACR_GENIE/v20_public
export ONCOBERT_GENIE_CLINICAL=$ONCOBERT_GENIE_ROOT/data_clinical_sample.txt
export ONCOBERT_TCGA_ROOT=/path/to/TCGA
export ONCOBERT_TCGA_CLINICAL=$ONCOBERT_TCGA_ROOT/tcga_pancan_clinical_data.csv
export ONCOBERT_TCGA_MODEL_ROOT=/path/to/tcga_mc3_three_state_set_transformer
export ONCOBERT_CHORD_BENCHMARK_ROOT=/path/to/chord_os_benchmark
export ONCOBERT_ICI_RESPONSE_ROOT=/path/to/response_benchmark
```

`ONCOBERT_WORK_ROOT` contains the manuscript-style `artifacts/` tree. If it is
omitted, scripts use the repository root. Exact intermediate files are listed
in each workflow README.

`mutation_projector_network_edges.pt` contains the eight supplied
MutationProjector networks restricted to the shared-200 gene vocabulary. Its
provenance and graph audit are recorded in `manifests/`.
