# Cross-cancer generalizability

## Analysis overview

The composite below evaluates Cluster 2 survival and objective-response
associations beyond the discovery NSCLC cohort, including non-overlapping
MSK-IMPACT cancers and harmonized immunotherapy-response cohorts.

<div align="center">
  <img src="../../data/example_outputs/biomarker_cross_cancer.png"
       width="100%"
       alt="Cross-cancer generalizability of Cluster 2 outcome associations">
</div>

`build_figure.py` combines the non-overlapping MSK-IMPACT pan-cancer ICI cohort,
cancer-specific Cox estimates, and harmonized response cohorts. NSCLC is
excluded from the MSK-IMPACT analysis to avoid overlap with MSK-CHORD.

Final reference: `data/example_outputs/biomarker_cross_cancer.png`.
