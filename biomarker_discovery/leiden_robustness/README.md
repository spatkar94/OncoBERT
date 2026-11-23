# Leiden robustness

## Analysis overview

The figure below tests whether the Cluster 2–ICI treatment interaction remains
consistent across Leiden resolutions and random initializations.

<div align="center">
  <img src="../../data/example_outputs/biomarker_leiden_robustness.png"
       width="100%"
       alt="Robustness of the Cluster 2 ICI interaction across Leiden solutions">
</div>

`analyze.py` evaluates 100 Leiden solutions (five resolutions x twenty seeds)
and identifies the cluster corresponding to the reference Cluster 2 by maximum
patient-membership overlap. `build_figure.py` reports the interaction-HR and
interaction-p-value distributions across resolutions.

Final reference: `data/example_outputs/biomarker_leiden_robustness.png`.
