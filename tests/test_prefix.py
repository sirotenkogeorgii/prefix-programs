"""Soft and variational prefixes."""
import math

import pytest
import torch

from prefixlab.prefix import SoftPrefix, VariationalPrefix, gaussian_prior_from_embeddings


def test_soft_prefix_from_vocab_has_shape_and_rows_of_embedding_matrix(tiny_model):
    E = tiny_model.get_input_embeddings().weight.detach()
    p = SoftPrefix.from_vocab(E, k=4, seed=0)
    out = p(batch_size=2)
    assert out.shape == (2, 4, E.shape[1])
    for row in p.weight.detach():
        assert torch.isclose(E, row).all(dim=1).any()


def test_soft_prefix_is_trainable_and_batch_broadcast():
    p = SoftPrefix(k=2, d=8)
    assert p.weight.requires_grad
    out = p(batch_size=3)
    assert out.shape == (3, 2, 8)
    assert torch.equal(out[0], out[2])


def test_variational_mean_mode_returns_mu():
    vp = VariationalPrefix(k=3, d=8, prior_mu=torch.zeros(8), prior_log_sigma=torch.zeros(8))
    assert torch.equal(vp(1, sample=False)[0], vp.mu)


def test_variational_sample_differs_from_mean():
    vp = VariationalPrefix(k=3, d=8, prior_mu=torch.zeros(8), prior_log_sigma=torch.zeros(8))
    torch.manual_seed(0)
    s = vp(1, sample=True)[0]
    assert s.shape == vp.mu.shape
    assert not torch.allclose(s, vp.mu)


def test_kl_closed_form_matches_monte_carlo():
    torch.manual_seed(0)
    vp = VariationalPrefix(k=2, d=8, prior_mu=torch.randn(8), prior_log_sigma=0.3 * torch.randn(8))
    with torch.no_grad():
        vp.mu.copy_(torch.randn(2, 8))
        vp.log_sigma.copy_(-0.5 + 0.2 * torch.randn(2, 8))
    kl = vp.kl_nats().item()
    mc = vp.kl_nats_mc(n=200_000).item()
    assert kl > 0
    assert abs(kl - mc) / kl < 0.02
    assert vp.kl_bits().item() == pytest.approx(kl / math.log(2), rel=1e-6)


def test_kl_is_zero_when_posterior_equals_prior():
    mu0, ls0 = torch.randn(8), 0.1 * torch.randn(8)
    vp = VariationalPrefix(k=2, d=8, prior_mu=mu0, prior_log_sigma=ls0)
    with torch.no_grad():
        vp.mu.copy_(mu0.expand(2, 8))
        vp.log_sigma.copy_(ls0.expand(2, 8))
    assert vp.kl_nats().item() == pytest.approx(0.0, abs=1e-6)


def test_gaussian_prior_from_embeddings_matches_column_moments(tiny_model):
    E = tiny_model.get_input_embeddings().weight.detach()
    mu, log_sigma = gaussian_prior_from_embeddings(E)
    assert mu.shape == (E.shape[1],)
    assert torch.allclose(mu, E.mean(0), atol=1e-6)
    assert torch.allclose(log_sigma.exp(), E.std(0), atol=1e-6)
