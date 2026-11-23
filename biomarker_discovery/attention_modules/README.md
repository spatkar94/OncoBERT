# Cluster 2 attention-module interpretability

The workflow is:

1. `extract_attention.py`: average final-layer gene-gene attention over Cluster
   2 patients.
2. `sweep_spectral_k.py`: spectral clustering and normalized-Laplacian eigengap
   analysis.
3. `pathway_enrichment.py`: Reactome enrichment of K=6 modules.
4. `analyze_module_mutations.py`: module-level mutation and co-mutation
   summaries in Cluster 2 and matched controls.
5. `plot_attention_modules.py` and `build_figure.py`: full matrix, module
   insets, integrated-gradient attribution, and mutation-frequency panels.

The attention matrix is averaged across patients and heads and is interpreted
as model routing, not a physical interaction network.

Final reference: `data/example_outputs/biomarker_attention_modules.png`.
