"""Matched rank/state Transformers with learned gene identity and uniform masking.

ESM2/GenePT/STRING vectors are deliberately absent: those networks determine
precomputed rank targets only. All 200 genes have trainable identity vectors.
"""
import torch
from torch import nn


def uniform_mask(values, fraction=.15, generator=None):
    """Select exactly floor(fraction * genes) distinct genes per sample, uniformly."""
    if values.ndim != 2 or not 0 < fraction <= 1:
        raise ValueError('Expected samples x genes and a valid mask fraction')
    count = max(1, int(values.shape[1] * fraction))
    scores = torch.rand(values.shape, generator=generator, device=values.device)
    return scores.argsort(1).argsort(1) < count


class CleanMaskedSetModel(nn.Module):
    """Gene embedding + masked rank/state embedding, then a shared Transformer.

    Values are zero-based target IDs. The last value-embedding row is input-only
    MASK. Gene IDs are never hidden, and there are no input-order embeddings.
    Construct shared parameters before the task-specific tables so seed-matched
    models start with identical gene embeddings and Transformer weights.
    """
    def __init__(self, num_classes, num_genes=200, dim=256, layers=8,
                 heads=8, ff_dim=2048, dropout=.1):
        super().__init__()
        self.num_genes = num_genes
        self.num_classes = num_classes
        self.dim = dim
        # The gene identity representation is this trainable lookup table ONLY.
        self.gene_embedding = nn.Embedding(num_genes, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(dim, heads, ff_dim, dropout, batch_first=True)
            for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)
        self.value_embedding = nn.Embedding(num_classes + 1, dim)
        self.head = nn.Linear(dim, num_classes)

    def tokenize(self, values, mask=None, gene_ids=None):
        if values.ndim != 2 or values.shape[1] != self.num_genes:
            raise ValueError('All genes must be supplied')
        if mask is None:
            mask = torch.zeros_like(values, dtype=torch.bool)
        if mask.dtype != torch.bool or mask.shape != values.shape:
            raise ValueError('Invalid mask')
        if gene_ids is None:
            gene_ids = torch.arange(self.num_genes, device=values.device)
        identities = self.gene_embedding(gene_ids)
        if identities.ndim == 2:
            identities = identities.unsqueeze(0)
        # Replace rank/state IDs BEFORE embedding: hidden values cannot enter attention.
        visible_ids = values.masked_fill(mask, self.num_classes)
        return identities + self.value_embedding(visible_ids)

    def forward(self, values, mask=None, gene_ids=None):
        nodes = self.tokenize(values, mask, gene_ids)
        nodes = torch.cat([self.cls_token.expand(len(values), -1, -1), nodes], 1)
        for block in self.blocks:
            nodes = block(nodes)
        nodes = self.norm(nodes)
        return self.head(nodes[:, 1:]), nodes[:, 0]


def masked_ce(logits, values, mask):
    """Standard unweighted CE, exclusively on masked genes."""
    if not mask.any():
        raise ValueError('No reconstruction targets')
    return nn.functional.cross_entropy(logits[mask].float(), values[mask])
