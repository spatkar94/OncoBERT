# Benchmarks

Each subfolder contains the analysis code for a final manuscript figure. The
PNG files in `data/example_outputs/` are regression references, not analysis
inputs. Large embeddings and clinical tables are staged through the data roots
described in `data/README.md`.

| Workflow | Final reference figure |
|---|---|
| `genie_chord_os/` | `benchmark_genie_chord_os.png` |
| `tcga/` | `benchmark_tcga.png` |
| `nsclc_ici_os/` | `benchmark_nsclc_ici_os.png` |
| `representation_diagnostics/` | `overlay_{cancer_type,institution,panel_source}.png` |

Retrieval benchmarks use held-out sample-level representations. CHORD survival
benchmarks operate at the patient level and mean-pool eligible baseline sample
representations when a patient has multiple baseline specimens.
