"""MutationProjector-style eight-network GATv2 encoder for shared-200 profiles.

Only the encoder operator is adapted from MutationProjector. Inputs and the
masked-state objective remain those of the controlled shared-200 baseline:
learned gene identities plus target-state/MASK embeddings.
"""
from collections.abc import Sequence
import math

import torch
from torch import nn
from torch_geometric.nn import GATv2Conv


class GeneWiseLinear(nn.Module):
    """Independent linear transformation per gene in one batched operation.

    This is algebraically identical to applying one ``nn.Linear`` module to
    each gene in a Python loop, but avoids hundreds of small GPU kernel
    launches per layer.
    """

    def __init__(self, num_genes, in_features, out_features):
        super().__init__()
        self.num_genes = int(num_genes)
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight = nn.Parameter(torch.empty(
            self.num_genes, self.out_features, self.in_features))
        self.bias = nn.Parameter(torch.empty(self.num_genes, self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        # Match the initialization of 200 independent nn.Linear modules.
        bound = 1 / math.sqrt(self.in_features)
        for gene in range(self.num_genes):
            nn.init.kaiming_uniform_(self.weight[gene], a=math.sqrt(5))
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, nodes):
        if nodes.ndim != 3 or nodes.shape[1:] != (self.num_genes, self.in_features):
            raise ValueError("Invalid gene-wise linear input")
        return torch.einsum("bgi,goi->bgo", nodes, self.weight) + self.bias


class MutationProjectorEnsembleBlock(nn.Module):
    """Two-stage residual GAT/FF block used by MutationProjector.

    Each biological network has its own GATv2 operator in every block. Their
    outputs are concatenated and projected separately for every gene. As in the
    reference code, the gene-specific projection and FF layers are shared by
    all GAT blocks, while GAT and layer-normalization parameters are not.
    """

    def __init__(self, num_genes, dim, network_edges, blocks=2, heads=1,
                 d_ff=100, dropout=.1, vectorized_gene_layers=False,
                 cache_batched_edges=False):
        super().__init__()
        if not isinstance(network_edges, Sequence) or not network_edges:
            raise ValueError("At least one network edge list is required")
        self.num_genes = int(num_genes)
        self.dim = int(dim)
        self.num_networks = len(network_edges)
        self.num_blocks = int(blocks)
        self.vectorized_gene_layers = bool(vectorized_gene_layers)
        self.cache_batched_edges = bool(cache_batched_edges)
        self._batched_edge_cache = {}
        for index, edge_index in enumerate(network_edges):
            edge_index = torch.as_tensor(edge_index, dtype=torch.long)
            if edge_index.ndim != 2 or edge_index.shape[0] != 2:
                raise ValueError("Every edge list must have shape [2, edges]")
            if edge_index.numel() and (
                    int(edge_index.min()) < 0 or int(edge_index.max()) >= num_genes):
                raise ValueError("Network endpoint outside the gene vocabulary")
            self.register_buffer(f"edge_index_{index}", edge_index.clone())

        self.gat_layers = nn.ModuleList([
            nn.ModuleList([
                GATv2Conv(dim, dim, heads=heads, concat=False,
                          add_self_loops=False, dropout=dropout)
                for _ in network_edges
            ])
            for _ in range(blocks)
        ])
        # These layers deliberately match the reference's per-gene modules.
        if self.vectorized_gene_layers:
            self.network_projections = GeneWiseLinear(
                num_genes, dim * self.num_networks, dim)
            self.ff1 = GeneWiseLinear(num_genes, dim, d_ff)
            self.ff2 = GeneWiseLinear(num_genes, d_ff, dim)
        else:
            self.network_projections = nn.ModuleList([
                nn.Linear(dim * self.num_networks, dim) for _ in range(num_genes)
            ])
            self.ff1 = nn.ModuleList([nn.Linear(dim, d_ff) for _ in range(num_genes)])
            self.ff2 = nn.ModuleList([nn.Linear(d_ff, dim) for _ in range(num_genes)])
        self.norm1 = nn.ModuleList([nn.LayerNorm(dim) for _ in range(blocks)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(dim) for _ in range(blocks)])
        self.dropout = nn.Dropout(dropout)

    def network_edge(self, index):
        return getattr(self, f"edge_index_{index}")

    def _batched_edges(self, edge_index, batch_size):
        """Make disconnected graph copies; no message can cross samples."""
        offsets = (torch.arange(batch_size, device=edge_index.device)
                   * self.num_genes).view(batch_size, 1, 1)
        return (edge_index.unsqueeze(0) + offsets).permute(1, 0, 2).reshape(2, -1)

    @staticmethod
    def _gene_wise(layers, nodes):
        if isinstance(layers, GeneWiseLinear):
            return layers(nodes)
        return torch.stack([layer(nodes[:, gene])
                            for gene, layer in enumerate(layers)], dim=1)

    def _all_batched_edges(self, batch_size, device):
        key = (device.type, device.index, int(batch_size))
        if self.cache_batched_edges and key in self._batched_edge_cache:
            return self._batched_edge_cache[key]
        result = [self._batched_edges(self.network_edge(i), batch_size)
                  for i in range(self.num_networks)]
        if self.cache_batched_edges:
            self._batched_edge_cache[key] = result
        return result

    def forward(self, nodes):
        if nodes.ndim != 3 or nodes.shape[1:] != (self.num_genes, self.dim):
            raise ValueError("Expected nodes with shape [batch, genes, dim]")
        batch_size = nodes.shape[0]
        batched_edges = self._all_batched_edges(batch_size, nodes.device)
        for block_index, gat_group in enumerate(self.gat_layers):
            flat = nodes.reshape(batch_size * self.num_genes, self.dim)
            network_outputs = [
                gat(flat, edge_index).reshape(batch_size, self.num_genes, self.dim)
                for gat, edge_index in zip(gat_group, batched_edges)
            ]
            concatenated = torch.cat(network_outputs, dim=-1)
            projected = self._gene_wise(self.network_projections, concatenated)
            nodes = self.norm1[block_index](nodes + self.dropout(projected))
            ff = self._gene_wise(self.ff1, nodes).relu()
            ff = self._gene_wise(self.ff2, ff)
            nodes = self.norm2[block_index](nodes + self.dropout(ff))
        return nodes


class MutationProjectorBinaryGAT(nn.Module):
    """Learned gene/state tokenizer plus the official eight-network GAT design.

    ``num_states`` defaults to the original binary controlled comparator.  The
    input-only MASK ID is always the row immediately after the target states,
    allowing otherwise identical multiclass functional-state experiments.
    """

    def __init__(self, network_edges, num_genes=200, dim=128, blocks=2,
                 heads=1, d_ff=100, dropout=.1, num_states=2,
                 vectorized_gene_layers=False, cache_batched_edges=False):
        super().__init__()
        self.num_genes = int(num_genes)
        self.dim = int(dim)
        self.num_states = int(num_states)
        if self.num_states < 2:
            raise ValueError("At least two target states are required")
        self.gene_embedding = nn.Embedding(num_genes, dim)
        self.state_embedding = nn.Embedding(self.num_states + 1, dim)
        self.encoder = MutationProjectorEnsembleBlock(
            num_genes, dim, network_edges, blocks, heads, d_ff, dropout,
            vectorized_gene_layers, cache_batched_edges)
        # Keep the binary baseline's shared decoder; only the encoder is replaced.
        self.state_head = nn.Linear(dim, self.num_states)

    def tokenize(self, values, mask=None, gene_ids=None):
        if values.ndim != 2 or values.shape[1] != self.num_genes:
            raise ValueError("All prepared shared genes must be supplied")
        if mask is None:
            mask = torch.zeros_like(values, dtype=torch.bool)
        if mask.dtype != torch.bool or mask.shape != values.shape:
            raise ValueError("Invalid state mask")
        if gene_ids is None:
            gene_ids = torch.arange(self.num_genes, device=values.device)
        identities = self.gene_embedding(gene_ids)
        if identities.ndim == 2:
            identities = identities.unsqueeze(0)
        if values.numel() and (values.min() < 0 or values.max() >= self.num_states):
            raise ValueError("State ID outside the target vocabulary")
        visible_states = values.long().masked_fill(mask, self.num_states)
        return identities + self.state_embedding(visible_states)

    def encode(self, values, mask=None):
        nodes = self.encoder(self.tokenize(values, mask))
        # No synthetic CLS/covariate node is introduced into the supplied graphs.
        return nodes, nodes.mean(dim=1)

    def forward(self, values, mask=None):
        nodes, tumors = self.encode(values, mask)
        return self.state_head(nodes), tumors
