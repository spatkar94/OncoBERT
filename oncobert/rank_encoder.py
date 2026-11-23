"""Diffusion-rank encoder used in the sequence/rank ablation experiments."""

from __future__ import annotations

from .model import OncoBERT


class RankEncoder(OncoBERT):
    """Predict masked, precomputed diffusion ranks for the full gene set.

    Inputs are zero-based ranks. Gene identities remain visible when ranks are
    masked. The architecture is otherwise identical to the Set Transformer,
    enabling controlled comparisons of state and diffusion-rank objectives.
    """

    def __init__(
        self,
        num_genes: int = 200,
        dim: int = 128,
        layers: int = 8,
        heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(
            num_classes=num_genes,
            num_genes=num_genes,
            dim=dim,
            layers=layers,
            heads=heads,
            ff_dim=ff_dim,
            dropout=dropout,
        )
