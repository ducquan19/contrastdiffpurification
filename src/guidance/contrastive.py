"""Contrastive guidance for the reverse diffusion process -- the method of the paper.

Bai et al., ICML 2024, "Diffusion Models Demand Contrastive Guidance for Adversarial
Purification to Advance". Algorithm 1 and Eq. (8):

    eps_tilde(x(t)) = eps(x(t)) + lambda * grad_x(t) loss(x(t)_a, x(t)_p; tau)

Correspondence with the paper:

* ``lambda``  -> ``guidance_strength``;  ``tau`` -> ``temperature``;  ``tau^-`` -> ``tau_minus``
* the anchor ``x(t)_a`` is the current reverse step, and its positive pair ``x(t)_p`` is the
  previous step ``x(t + dt)`` (``xt_cache``);
* the negative pairs are the other samples of the same minibatch, which is why the attack batch
  size is a hyperparameter of this method and not just an infrastructure setting;
* ``F(.)`` is ``embed_net``, trained online on every minibatch (one Adam step per call) and
  re-initialised for each purified batch by ``reset_drift_counter``;
* ``estimator='hard'`` is the hard-negative-mining loss of Eq. (10), ``'soft'`` the plain
  InfoNCE of Eq. (9).

``drift_min`` / ``drift_max`` restrict the guidance to a window of reverse steps. Algorithm 1
guides every step; the window comes from the released code.

Reference implementation of the contrastive loss:
- https://github.com/YiDongOuYang/Contrastive-Guided-Diffusion-Process.git
"""

from enum import Enum
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as torchF


class MLPBlock(nn.Module):
    """Gated (SwiGLU-style) MLP block: ``d3(silu(d1(x)) * d2(x))``, same input and output width."""

    def __init__(self, input_size=640, hidden_size=128):
        super().__init__()
        self.d1 = nn.Linear(input_size, hidden_size, bias=False)
        self.d2 = nn.Linear(input_size, hidden_size, bias=False)
        self.d3 = nn.Linear(hidden_size, input_size, bias=False)

    def reset_parameters(self):
        self.d1.reset_parameters()
        self.d2.reset_parameters()
        self.d3.reset_parameters()

    def forward(self, input):
        out = self.d3(torchF.silu(self.d1(input)) * self.d2(input))
        return out


class MLP(nn.Module):
    """The feature extractor F(.): three residual ``MLPBlock``s applied to classifier logits."""

    def __init__(self, input_size=640, hidden_size=128):
        super(MLP, self).__init__()
        self.layer0 = MLPBlock(input_size, hidden_size)
        self.layer1 = MLPBlock(input_size, hidden_size)
        self.layer2 = MLPBlock(input_size, hidden_size)

    def reset_parameters(self):
        self.layer0.reset_parameters()
        self.layer1.reset_parameters()
        self.layer2.reset_parameters()

    def forward(self, input: torch.Tensor):
        out = input
        out = self.layer0(out) + out
        out = self.layer1(out) + out
        out = self.layer2(out) + out
        return out


class Estimator(Enum):
    """Negative term of the contrastive loss: ``hard`` = HNM (Eq. 10), ``soft`` = InfoNCE (Eq. 9)."""
    hard = 0
    soft = 1


class ContrastiveLoss(nn.Module):
    """Debiased / hard-negative contrastive loss between two batches of embeddings.

    ``x_1[i]`` and ``x_2[i]`` are two views of sample ``i``; every other sample of either batch
    is a negative.
    """

    def __init__(
        self,
        tau_minus: float = 0.1,
        beta: float = 1.0,
        temperature: float = 10,
        estimator: Union[Estimator, str] = Estimator.hard,
    ):
        """``tau_minus`` is the paper's tau^- (assumed probability of another class)."""
        super().__init__()

        self.tau_minus = tau_minus
        self.beta = beta
        self.temperature = temperature

        if isinstance(estimator, str):
            estimator = estimator.lower()
            if estimator == "hard":
                estimator = Estimator.hard
            elif estimator == "soft":
                estimator = Estimator.soft
            else:
                raise ValueError(f"Unknown estimator: {estimator}")

        self.estimator = estimator

    @torch.no_grad()
    def negative_mask(self, batch_size):
        """Boolean ``(2B, 2B)`` mask over ``cat([x_1, x_2])`` that is False on the self- and
        positive-pair entries of each row, i.e. True exactly on the negatives."""
        mask = torch.ones((batch_size, 2 * batch_size), dtype=torch.bool)
        if batch_size != 1:
            for i in range(batch_size):
                mask[i, i] = 0
                mask[i, i + batch_size] = 0
        else:
            mask[0, 1] = 0
        mask = torch.cat([mask, mask], dim=0)
        return mask

    def forward(self, x_1: torch.Tensor, x_2: torch.Tensor) -> torch.Tensor:
        """Mean loss over both views; raises ``ValueError`` if it is NaN."""
        batch_size = x_1.size(0)

        # negatives: exp(cos / tau) between all 2B embeddings, minus self and positive pairs
        out = torch.cat([x_1, x_2], dim=0)
        neg = torch.exp(torchF.cosine_similarity(out.unsqueeze(1), out.unsqueeze(0), dim=-1) / self.temperature)
        mask = self.negative_mask(batch_size).to(x_1.device)
        neg = neg.masked_select(mask).view(2 * batch_size, -1)

        # positives: note this is the full (B, B) similarity matrix between the two views, not
        # only the matched diagonal pairs (kept as in the released code).
        pos = torch.exp(torchF.cosine_similarity(x_1.unsqueeze(1), x_2.unsqueeze(0), dim=-1) / self.temperature)
        pos = torch.cat([pos, pos], dim=0)

        if self.estimator == Estimator.hard:
            N = max(batch_size * 2 - 2, 2)
            neg = torch.clamp(neg, min=1e-6)
            # importance weights neg^beta: harder (more similar) negatives weigh more
            imp = (self.beta * neg.log()).exp()
            reweight_neg = (imp * neg).sum(dim=-1, keepdim=True) / \
                imp.mean(dim=-1, keepdim=True)
            # debiased negative term, Eq. (10)
            Ng = (-self.tau_minus * N * pos + reweight_neg) / (1 - self.tau_minus)
            # lower bound at its theoretical minimum N * e^(-1/tau)
            Ng = torch.clamp(Ng, min=N*np.e**(-1 / self.temperature))
        elif self.estimator == Estimator.soft:
            # known shape mismatch with ``pos`` (B, B) -> see tests/test_contrastive.py
            Ng = neg.sum(dim=-1)
        else:
            raise NotImplementedError(
                f"Estimator {self.estimator} is not supported currently.")

        ratio = torch.clamp(pos / (pos + Ng), min=1e-6)
        loss = - torch.log(ratio).mean()

        if torch.isnan(loss):
            raise ValueError("NaN loss")

        return loss


class ContrastiveGuidedDiffusion(nn.Module):
    """Score network wrapper that adds the contrastive gradient of Eq. (8) to its output.

    Stateful across calls: it caches the previous input ``x(t + dt)`` as the positive view,
    counts reverse steps in ``drift_counter`` and trains F(.) online. Call
    ``reset_drift_counter`` before purifying a new batch.
    """

    def __init__(
        self,
        model: nn.Module,
        classifier: nn.Module,
        n_classes: int,
        embed_lr=1e-2,
        drift_min=2,
        drift_max=10000,
        tau_minus: float = 0.1,
        beta: float = 1.0,
        temperature: float = 10,
        guidance_strength: float = 5000,
        estimator: Union[Estimator, str] = Estimator.hard,
    ):
        """Wrap a score network so that its output carries the contrastive guidance.

        Args:
            model: the score network eps_theta; called as ``model(x, t)``.
            classifier: frozen backbone whose logits F(.) embeds.
            n_classes: width of the classifier output, i.e. the input width of F(.).
            embed_lr: Adam learning rate of F(.) (``embed_net``).
            drift_min / drift_max: add the guidance only while
                ``drift_min <= drift_counter < drift_max``, counting calls
                from 0. The paper guides every step.
            tau_minus, beta, temperature, estimator: contrastive loss, see ContrastiveLoss.
            guidance_strength: lambda in Eq. (8), the scale of the contrastive gradient.

        The number of negative pairs follows the batch passed to ``forward`` at run time.
        """
        super().__init__()
        self.model = model
        self.embed_lr = embed_lr

        self.classifier = classifier.eval()
        self.embed_net = MLP(n_classes, 2*n_classes)
        self.optimizer = torch.optim.Adam(self.embed_net.parameters(), lr=self.embed_lr)

        self.contrastive_loss = ContrastiveLoss(
            tau_minus=tau_minus,
            beta=beta,
            temperature=temperature,
            estimator=estimator,
        )

        self.xt_cache = None

        self.drift_counter = 0
        self.drift_min = drift_min
        self.drift_max = drift_max

        self.guidance_strength = guidance_strength

    def reset_drift_counter(self):
        """Restart the step counter and re-initialise F(.) and its optimizer.

        ``xt_cache`` is kept; it is dropped automatically when the next batch has another shape.
        """
        self.drift_counter = 0
        self.embed_net.reset_parameters()
        del self.optimizer
        self.optimizer = torch.optim.Adam(self.embed_net.parameters(), lr=self.embed_lr)

    def acquisition_grad(self, x: torch.Tensor, x0: torch.Tensor):
        """Return ``lambda * grad_x loss(F(f(x)), F(f(x0)))`` and take one Adam step on F(.).

        ``x`` is the current reverse step (anchor), ``x0`` the previous one (positive view);
        ``f`` is the frozen classifier. Neither input is modified.
        """
        x = x.detach().clone()
        x0 = x0.detach().clone()

        with torch.no_grad():
            _x_0 = self.classifier(x0)

        with torch.enable_grad():
            # gradient w.r.t. the input is read from a zero perturbation ``delta``
            delta = torch.zeros_like(x, requires_grad=True)
            _x_0 = self.embed_net(_x_0) # type: torch.Tensor
            _x_d = self.classifier(x + delta)  # type: torch.Tensor
            _x_d = self.embed_net(_x_d)

            loss = self.contrastive_loss(_x_0, _x_d)

            # the same backward pass trains F(.) and yields grad w.r.t. delta
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return delta.grad.detach() * self.guidance_strength

    def forward(self, x: torch.Tensor, t):
        """``model(x, t)`` plus the guidance term, when the current step is inside the window."""
        out = self.model.forward(x, t)

        # a batch of another shape is a new purification run: drop the cached positive view
        if self.xt_cache is not None:
            if self.xt_cache.shape != x.shape:
                self.xt_cache = None
                self.reset_drift_counter()

        # the first step has no previous x(t + dt) to contrast with
        if self.xt_cache is not None and self.drift_min <= self.drift_counter < self.drift_max:
            delta = self.acquisition_grad(x, self.xt_cache)
            # only the first C channels are eps; models with learned variance output 2C channels
            out[:, 0:x.size(1)] = out[:, 0:x.size(1)] + delta

        self.xt_cache = x.detach().clone()

        self.drift_counter += 1
        return out
