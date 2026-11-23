"""Core OncoBERT three-state, position-free Set Transformer.

Every sample supplies the same ordered gene vocabulary. Each input token is the
sum of a learned gene-identity embedding and a learned mutation-state
embedding. State masking never hides gene identity, and the model does not use
positional embeddings.
"""

from __future__ import annotations

import torch
from torch import nn


STATE_NAMES = ("WILD_TYPE", "MUTATED_NOT_LOF", "LOF")


def uniform_mask(
    values: torch.Tensor,
    fraction: float = 0.15,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Select the same number of distinct gene states in every sample."""
    if values.ndim != 2 or not 0 < fraction <= 1:
        raise ValueError("Expected a samples-by-genes tensor and 0 < fraction <= 1")
    count = max(1, int(values.shape[1] * fraction))
    scores = torch.rand(values.shape, generator=generator, device=values.device)
    return scores.argsort(dim=1).argsort(dim=1) < count


class OncoBERT(nn.Module):
    """Full-gene Set Transformer for masked mutation-state prediction."""

    def __init__(
        self,
        num_classes: int = 3,
        num_genes: int = 200,
        dim: int = 128,
        layers: int = 8,
        heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("Embedding dimension must be divisible by attention heads")
        self.num_genes = num_genes
        self.num_classes = num_classes
        self.dim = dim

        # Attribute names exactly match the released manuscript checkpoint.
        self.gene_embedding = nn.Embedding(num_genes, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    dim, heads, ff_dim, dropout, batch_first=True
                )
                for _ in range(layers)
            ]
        )
        self.norm = nn.LayerNorm(dim)
        self.value_embedding = nn.Embedding(num_classes + 1, dim)
        self.head = nn.Linear(dim, num_classes)

    @property
    def mask_state_id(self) -> int:
        return self.num_classes

    def tokenize(
        self,
        values: torch.Tensor,
        mask: torch.Tensor | None = None,
        gene_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Combine gene identity with visible state or input-only MASK embeddings."""
        if values.ndim != 2 or values.shape[1] != self.num_genes:
            raise ValueError(f"Expected samples x {self.num_genes} state IDs")
        if values.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError("State IDs must use an integer tensor dtype")
        if values.numel() and (values.min() < 0 or values.max() >= self.num_classes):
            raise ValueError("State IDs fall outside the model vocabulary")
        if mask is None:
            mask = torch.zeros_like(values, dtype=torch.bool)
        if mask.dtype != torch.bool or mask.shape != values.shape:
            raise ValueError("Mask must be a Boolean tensor matching values")
        if gene_ids is None:
            gene_ids = torch.arange(self.num_genes, device=values.device)
        identities = self.gene_embedding(gene_ids)
        if identities.ndim == 2:
            identities = identities.unsqueeze(0)
        visible_values = values.masked_fill(mask, self.mask_state_id)
        return identities + self.value_embedding(visible_values)

    def forward(
        self,
        values: torch.Tensor,
        mask: torch.Tensor | None = None,
        gene_ids: torch.Tensor | None = None,
        return_nodes: bool = False,
    ):
        nodes = self.tokenize(values, mask, gene_ids)
        nodes = torch.cat([self.cls_token.expand(len(values), -1, -1), nodes], 1)
        for block in self.blocks:
            nodes = block(nodes)
        nodes = self.norm(nodes)
        logits = self.head(nodes[:, 1:])
        patient_embedding = nodes[:, 0]
        if return_nodes:
            return logits, patient_embedding, nodes[:, 1:]
        return logits, patient_embedding

    @classmethod
    def from_checkpoint(cls, path, map_location="cpu"):
        """Instantiate and strictly load an OncoBERT checkpoint."""
        checkpoint = torch.load(path, map_location=map_location, weights_only=True)
        model = cls(**checkpoint["architecture"])
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        return model, checkpoint


def masked_cross_entropy(
    logits: torch.Tensor, values: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Unweighted cross-entropy evaluated exclusively at masked genes."""
    if not mask.any():
        raise ValueError("No reconstruction targets were selected")
    return nn.functional.cross_entropy(logits[mask].float(), values[mask])


OncoBERTSetTransformer = OncoBERT
CleanMaskedSetModel = OncoBERT
masked_ce = masked_cross_entropy
    
