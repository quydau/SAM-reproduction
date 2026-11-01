"""SGD and Sharpness-Aware Minimization (SAM) optimizers.

SAM update follows the reference implementation by Davide A. (davda54):
https://github.com/davda54/sam/blob/main/sam.py

This is a PyTorch adaptation for this repository. The two calls to
``first_step`` and ``second_step`` expose the reference algorithm to the
training loop, which must run forward/backward twice for each SAM batch.

Comparison with the pasted ``flax_training.py``: both normalize the full
gradient tree and perturb weights by ``rho * normalized_gradient``. This
optimizer also treats ``l2_reg`` as the JAX loss penalty
``0.5 * l2_reg * sum(parameter**2)`` and includes its gradient in both SAM
passes. Distributed synchronization, post-synchronization gradient clipping,
shared PRNG keys, and preserving BatchNorm state from the first forward still
belong in the training loop.
"""

from __future__ import annotations

from typing import Type

import torch
from torch.optim import Optimizer


class SAM(Optimizer):
    """Sharpness-Aware Minimization wrapped around a base optimizer.

    The first backward pass computes the perturbation direction. After
    ``first_step``, run a second forward/backward pass at the perturbed
    weights, then call ``second_step`` to restore the original weights and
    apply the base optimizer update using the second gradient.

    ``adaptive=True`` implements ASAM's parameter-scaled perturbation, as in
    the cited reference. Use the default ``False`` for standard SAM.

    ``l2_reg`` sets the loss penalty ``0.5 * l2_reg * sum(parameter**2)``;
    its gradient is included in both SAM passes. ``weight_decay`` in
    ``kwargs`` is passed to the base optimizer and remains independent of
    that loss penalty. ``l2_skip_1d`` leaves parameters with ``ndim <= 1``
    (biases, normalization scales) out of the penalty, as JAX's
    ``no_weight_decay_on_bn`` does.
    """

    def __init__(
        self,
        params,
        base_optimizer: Type[Optimizer] = torch.optim.SGD,
        rho: float = 0.05,
        adaptive: bool = False,
        l2_reg: float = 0.0,
        l2_skip_1d: bool = False,
        **kwargs,
    ) -> None:
        if rho < 0.0:
            raise ValueError(f"Invalid rho, should be non-negative: {rho}")
        if l2_reg < 0.0:
            raise ValueError(f"Invalid l2_reg, should be non-negative: {l2_reg}")

        # Keep the loss penalty separate from the base optimizer's weight_decay.
        defaults = dict(rho=rho, adaptive=adaptive, l2_reg=l2_reg,
                        l2_skip_1d=l2_skip_1d, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        """Move weights to ``w + e(w)`` using gradients at ``w``."""
        self.add_l2_gradient()
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            # Match JAX dual_vector for non-zero gradients. Keep the zero
            # gradient case well-defined instead of dividing by zero.
            scale = torch.where(
                grad_norm > 0,
                group["rho"] / grad_norm,
                torch.zeros_like(grad_norm),
            )
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                self.state[parameter]["old_p"] = parameter.data.clone()
                perturbation = (
                    parameter.pow(2) if group["adaptive"] else 1.0
                ) * parameter.grad * scale.to(parameter)
                parameter.add_(perturbation)

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False,
                    *, l2_already_added: bool = False) -> None:
        """Restore ``w`` and update it with gradients at ``w + e(w)``."""
        # At this point parameters are still perturbed, matching the JAX
        # second loss evaluation at ``noised_model``.
        if not l2_already_added:
            self.add_l2_gradient()
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                parameter.data = self.state[parameter]["old_p"]

        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def add_l2_gradient(self) -> None:
        for group in self.param_groups:
            coefficient = group["l2_reg"]
            if coefficient == 0.0:
                continue
            pairs = [(p.grad, p) for p in group["params"]
                     if p.grad is not None
                     and not (group["l2_skip_1d"] and p.ndim <= 1)]
            if pairs:
                grads, params = zip(*pairs)
                torch._foreach_add_(list(grads), list(params), alpha=coefficient)

    @torch.no_grad()
    def step(self, closure=None):
        """Run SAM with a closure that performs a full forward/backward pass.

        Training loops that manage the two passes explicitly should call
        ``first_step`` and ``second_step`` instead. A closure is required here
        because SAM needs two forward/backward evaluations.
        """
        if closure is None:
            raise ValueError(
                "Sharpness-Aware Minimization requires a closure, but none was provided"
            )
        closure = torch.enable_grad()(closure)
        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self) -> torch.Tensor:
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict) -> None:
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


class RMSProp(Optimizer):
    """TF-style RMSProp with momentum (efficientnet_optim.RMSProp analogue).

    ``v = beta2*v + (1-beta2)*g^2`` (initialised to ones), then
    ``m = beta*m + lr*g/sqrt(v+eps)`` and ``w -= m``. Reconstructed from the
    TensorFlow convention; verify against ``efficientnet/optim.py``.
    """

    def __init__(self, params, lr=0.1, beta=0.9, beta2=0.9, eps=0.001):
        super().__init__(params, dict(lr=lr, beta=beta, beta2=beta2, eps=eps))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state["v"] = torch.ones_like(p)
                    state["momentum"] = torch.zeros_like(p)
                state["v"].mul_(group["beta2"]).addcmul_(
                    p.grad, p.grad, value=1 - group["beta2"])
                state["momentum"].mul_(group["beta"]).add_(
                    group["lr"] * p.grad / (state["v"] + group["eps"]).sqrt())
                p.sub_(state["momentum"])
        return loss


class ExponentialMovingAverage:
    """EMA over all model weights and buffers (JAX ``ema_decay``).

    Uses the TF warm-up ``min(decay, (1+step)/(10+step))``; the JAX
    ``end_warmup_step`` rule could not be checked without its source.
    """

    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module, step: int) -> None:
        decay = min(self.decay, (1 + step) / (10 + step))
        for key, value in model.state_dict().items():
            if value.is_floating_point():
                self.shadow[key].mul_(decay).add_(value, alpha=1 - decay)
            else:
                self.shadow[key].copy_(value)


def make_sgd(params, **kwargs) -> torch.optim.SGD:
    """Construct the baseline SGD optimizer."""
    return torch.optim.SGD(params, **kwargs)
