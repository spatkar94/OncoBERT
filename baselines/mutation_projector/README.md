# MutationProjector-inspired GAT

Install `requirements-analysis.txt` before running this baseline; it adds
PyTorch Geometric to the core OncoBERT environment.

`model.py` implements two residual GAT/FF blocks over the eight supplied
biological networks: GRN, E3, phosphorylation, physical PPI, genetic
interaction, DDRAM, STRING, and PCNet. Each network has an independent GATv2
operator; network outputs are concatenated and projected with gene-specific
layers. The patient representation is the mean of the 200 final gene nodes.

Inputs under `--prepared`:

- `profiles.npz`: `states`, `mutations`, `genes`, `samples`, `patients`, and
  patient-disjoint `split` arrays.
- `network_edges.pt`: ordered edge dictionaries for the eight networks. A
  shared-200 copy is included at
  `data/reference_networks/mutation_projector_network_edges.pt`.
- `functional_states.npy`: required by `train_three_state.py`; target IDs are
  WT, mutated-not-LOF, and LOF.

The scripts train only with uniform 15% state masking. They do not use cancer
type, TMB, outcome, or treatment information.
