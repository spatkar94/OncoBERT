# Rank versus set encoder

`export_rank_embeddings.py` exports full-context rank-encoder representations,
`compute_retrieval_bootstrap.py` recomputes sample-level top-k retrieval with
patient-resampled confidence intervals, and `build_figure.py` combines broad
retrieval, detailed retrieval, and CHORD OS.

Final reference: `data/example_outputs/ablation_rank_vs_set.png`.
