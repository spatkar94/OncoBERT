# NMF baseline

`train.py` fits NMF on training profiles only and uses validation
reconstruction error to select the rank. `consensus_rank_sweep.py` evaluates
cophenetic stability over repeated fits. `evaluate_masked_reconstruction.py`
fits coefficients from visible entries and scores the same fixed masked entries
used by the neural models. `collect_masked_metrics.py` consolidates ranks.

The expected `profiles.npz` contains a binary `mutations` matrix and the same
patient-disjoint `train`, `validation`, and `test` split used by OncoBERT.
Test samples are never used to fit NMF or choose K.
