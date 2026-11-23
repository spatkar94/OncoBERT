# Baseline models

The baseline implementations use the same shared-200 cohort, patient split,
and fixed 15% test masks as OncoBERT.

- `mutation_projector/`: the two-block, eight-network GATv2 encoder adapted
  from MutationProjector. The cancer-type and mutation-burden covariates from
  the original MutationProjector task are intentionally omitted. Gene and
  mutation-state tokens are trained with the controlled masked-state objective.
- `nmf/`: non-negative matrix factorization with train-only fitting, validation
  rank selection/consensus analysis, and masked-entry reconstruction.

Run Python entry points as modules from the repository root, for example:

```bash
python -m baselines.mutation_projector.train_three_state \
  --prepared data/prepared/shared200 --output artifacts/mutation_projector
python -m baselines.nmf.train \
  --prepared data/prepared/shared200 --output artifacts/nmf
```

See the subfolder READMEs for input contracts and evaluation details.
