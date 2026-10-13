"""Soft prefixes: k free slot vectors, or a diagonal-Gaussian posterior over them.

`forward(batch_size, sample)` returns [B, k, d]. `kl_nats()` is the description
length of the prefix under the prior (bits-back); SoftPrefix returns 0 so the
training loop can treat both uniformly.
"""
from __future__ import annotations

import math

import torch
from torch import nn

LN2 = math.log(2.0)


def gaussian_prior_from_embeddings(embed_matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Diagonal Gaussian fitted to the token-embedding matrix: (mu [d], log_sigma [d])."""
    E = embed_matrix.detach().float()
    return E.mean(0), E.std(0).log()


def _sample_vocab_rows(embed_matrix: torch.Tensor, k: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, embed_matrix.shape[0], (k,), generator=g)
    return embed_matrix.detach()[idx].float().clone()


class SoftPrefix(nn.Module):
    def __init__(self, k: int, d: int, init: torch.Tensor | None = None):
        super().__init__()
        w = init.clone().float() if init is not None else torch.randn(k, d) * 0.02
        assert w.shape == (k, d)
        self.weight = nn.Parameter(w)

    @classmethod
    def from_vocab(cls, embed_matrix: torch.Tensor, k: int, seed: int) -> "SoftPrefix":
        return cls(k, embed_matrix.shape[1], init=_sample_vocab_rows(embed_matrix, k, seed))

    @property
    def k(self) -> int:
        return self.weight.shape[0]

    def forward(self, batch_size: int, sample: bool = False) -> torch.Tensor:
        return self.weight.unsqueeze(0).expand(batch_size, -1, -1)

    def kl_nats(self) -> torch.Tensor:
        return self.weight.new_zeros(())

    def kl_bits(self) -> torch.Tensor:
        return self.kl_nats() / LN2


class VariationalPrefix(nn.Module):
    def __init__(
        self,
        k: int,
        d: int,
        prior_mu: torch.Tensor,
        prior_log_sigma: torch.Tensor,
        init_mu: torch.Tensor | None = None,
        init_log_sigma: float = -2.0,
    ):
        super().__init__()
        mu = init_mu.clone().float() if init_mu is not None else prior_mu.detach().float().expand(k, d).clone()
        assert mu.shape == (k, d)
        self.mu = nn.Parameter(mu)
        self.log_sigma = nn.Parameter(torch.full((k, d), float(init_log_sigma)))
        self.register_buffer("prior_mu", prior_mu.detach().float().clone())
        self.register_buffer("prior_log_sigma", prior_log_sigma.detach().float().clone())

    @classmethod
    def from_vocab(
        cls, embed_matrix: torch.Tensor, k: int, seed: int, prior: tuple[torch.Tensor, torch.Tensor], **kw
    ) -> "VariationalPrefix":
        return cls(k, embed_matrix.shape[1], prior[0], prior[1], init_mu=_sample_vocab_rows(embed_matrix, k, seed), **kw)

    @property
    def k(self) -> int:
        return self.mu.shape[0]

    def forward(self, batch_size: int, sample: bool = False) -> torch.Tensor:
        z = self.mu
        if sample:
            z = z + torch.exp(self.log_sigma) * torch.randn_like(self.mu)
        return z.unsqueeze(0).expand(batch_size, -1, -1)

    def kl_nats(self) -> torch.Tensor:
        """KL( N(mu, sigma^2) || N(prior_mu, prior_sigma^2) ), summed over slots and coordinates."""
        s2 = torch.exp(2 * self.log_sigma)
        s02 = torch.exp(2 * self.prior_log_sigma)
        kl = (self.prior_log_sigma - self.log_sigma) + (s2 + (self.mu - self.prior_mu) ** 2) / (2 * s02) - 0.5
        return kl.sum()

    def kl_bits(self) -> torch.Tensor:
        return self.kl_nats() / LN2

    @torch.no_grad()
    def kl_nats_mc(self, n: int = 100_000) -> torch.Tensor:
        q = torch.distributions.Normal(self.mu, torch.exp(self.log_sigma))
        p = torch.distributions.Normal(self.prior_mu.expand_as(self.mu), torch.exp(self.prior_log_sigma).expand_as(self.mu))
        z = q.sample((n,))
        return (q.log_prob(z) - p.log_prob(z)).sum(dim=(-1, -2)).mean()


class HierarchicalPrefix(nn.Module):
    """Shared centre with per-member deviations: prefix(m) = centre + dev[m].

    centre ~ q(centre) with KL to the global prior; dev[m] ~ q(dev[m]) with KL to N(0, tau^2 I).
    The centre's KL is the shared (e.g. task) bits, each deviation's KL the member-specific
    (e.g. language) bits. Reverse nesting is the same class with members = tasks.
    """

    def __init__(
        self,
        k: int,
        d: int,
        members: list[str],
        prior: tuple[torch.Tensor, torch.Tensor],
        tau: float,
        init_centre: torch.Tensor | None = None,
        init_log_sigma: float = -2.0,
    ):
        super().__init__()
        prior_mu, prior_log_sigma = prior
        centre = init_centre.clone().float() if init_centre is not None else prior_mu.detach().float().expand(k, d).clone()
        self.centre_mu = nn.Parameter(centre)
        self.centre_log_sigma = nn.Parameter(torch.full((k, d), float(init_log_sigma)))
        self.dev_mu = nn.ParameterDict({m: nn.Parameter(torch.zeros(k, d)) for m in members})
        self.dev_log_sigma = nn.ParameterDict({m: nn.Parameter(torch.full((k, d), float(init_log_sigma))) for m in members})
        self.register_buffer("prior_mu", prior_mu.detach().float().clone())
        self.register_buffer("prior_log_sigma", prior_log_sigma.detach().float().clone())
        self.register_buffer("log_tau", torch.tensor(math.log(float(tau))))
        self.members = list(members)

    @property
    def k(self) -> int:
        return self.centre_mu.shape[0]

    def member_mean(self, member: str) -> torch.Tensor:
        return self.centre_mu + self.dev_mu[member]

    def forward(self, member: str, batch_size: int, sample: bool = False) -> torch.Tensor:
        z = self.member_mean(member)
        if sample:
            z = z + torch.exp(self.centre_log_sigma) * torch.randn_like(z) + torch.exp(self.dev_log_sigma[member]) * torch.randn_like(z)
        return z.unsqueeze(0).expand(batch_size, -1, -1)

    @staticmethod
    def _kl(mu, log_sigma, prior_mu, prior_log_sigma) -> torch.Tensor:
        s2 = torch.exp(2 * log_sigma)
        s02 = torch.exp(2 * prior_log_sigma)
        return ((prior_log_sigma - log_sigma) + (s2 + (mu - prior_mu) ** 2) / (2 * s02) - 0.5).sum()

    def kl_centre_nats(self) -> torch.Tensor:
        return self._kl(self.centre_mu, self.centre_log_sigma, self.prior_mu, self.prior_log_sigma)

    def kl_dev_nats(self) -> dict[str, torch.Tensor]:
        zero = torch.zeros((), device=self.centre_mu.device)
        return {m: self._kl(self.dev_mu[m], self.dev_log_sigma[m], zero, self.log_tau) for m in self.members}

    def kl_centre_bits(self) -> torch.Tensor:
        return self.kl_centre_nats() / LN2

    def kl_dev_bits(self) -> dict[str, torch.Tensor]:
        return {m: v / LN2 for m, v in self.kl_dev_nats().items()}
