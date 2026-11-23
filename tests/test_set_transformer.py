import unittest

import torch

from oncobert.model import OncoBERT, masked_cross_entropy, uniform_mask


def small_model():
    torch.manual_seed(7)
    return OncoBERT(
        num_classes=3, num_genes=10, dim=16, layers=2, heads=4, ff_dim=32, dropout=0.0
    ).eval()


class SetTransformerTests(unittest.TestCase):
    def test_uniform_mask_is_deterministic_and_exact(self):
        values = torch.zeros((4, 10), dtype=torch.long)
        first = uniform_mask(values, 0.3, torch.Generator().manual_seed(9))
        second = uniform_mask(values, 0.3, torch.Generator().manual_seed(9))
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(first.sum(1), torch.full((4,), 3)))

    def test_mask_hides_state_but_retains_gene_identity(self):
        model = small_model()
        first = torch.zeros((1, 10), dtype=torch.long)
        second = first.clone()
        second[0, 3] = 2
        mask = torch.zeros_like(first, dtype=torch.bool)
        mask[0, 3] = True
        logits_a, embedding_a = model(first, mask)
        logits_b, embedding_b = model(second, mask)
        self.assertTrue(torch.allclose(logits_a, logits_b))
        self.assertTrue(torch.allclose(embedding_a, embedding_b))

    def test_position_free_permutation_equivariance(self):
        model = small_model()
        values = torch.tensor([[0, 1, 2, 0, 1, 0, 2, 0, 0, 1]])
        permutation = torch.tensor([6, 0, 4, 9, 2, 8, 1, 7, 5, 3])
        logits, cls = model(values)
        permuted_logits, permuted_cls = model(values[:, permutation], gene_ids=permutation)
        self.assertTrue(torch.allclose(cls, permuted_cls, atol=1e-6))
        self.assertTrue(torch.allclose(logits[:, permutation], permuted_logits, atol=1e-6))

    def test_masked_cross_entropy_uses_selected_targets_only(self):
        model = small_model()
        values = torch.randint(0, 3, (2, 10))
        mask = uniform_mask(values, 0.2, torch.Generator().manual_seed(3))
        logits, _ = model(values, mask)
        loss = masked_cross_entropy(logits, values, mask)
        expected = torch.nn.functional.cross_entropy(logits[mask], values[mask])
        self.assertTrue(torch.allclose(loss, expected))


if __name__ == "__main__":
    unittest.main()
