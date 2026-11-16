"""Sliced Epps-Pulley statistic used by the state and effect SIGReg losses."""

from __future__ import annotations

import torch
from torch import Tensor, distributed as dist, nn
from torch.distributed.nn.functional import all_reduce


def _average_across_ranks(value: Tensor, world_size: int) -> Tensor:
    if world_size == 1:
        return value
    if dist.get_backend() == "nccl":
        return all_reduce(value, op=dist.ReduceOp.AVG)
    # Gloo supports SUM and its backward all-reduce, but has no AVG reduction.
    return all_reduce(value, op=dist.ReduceOp.SUM) / world_size


class _EppsPulley(nn.Module):
    """Compare a projected sample's characteristic function to a unit Gaussian."""

    def __init__(self, *, n_points: int, t_max: float) -> None:
        super().__init__()
        if n_points < 3 or n_points % 2 != 1:
            raise ValueError("n_points must be an odd integer of at least three")
        if t_max <= 0:
            raise ValueError("t_max must be positive")

        grid = torch.linspace(0, t_max, n_points, dtype=torch.float32)
        normal_cf = grid.square().mul_(0.5).neg_().exp_()
        spacing = t_max / (n_points - 1)
        # Integrate over nonnegative frequencies and use symmetry for the other half.
        quadrature = torch.full((n_points,), 2 * spacing, dtype=torch.float32)
        quadrature[0] = quadrature[-1] = spacing
        self.register_buffer("t", grid)
        self.register_buffer("phi", normal_cf)
        self.register_buffer("weights", quadrature * normal_cf)

    def forward(self, projections: Tensor, world_size: int) -> Tensor:
        angles = projections.unsqueeze(-1) * self.t
        real = _average_across_ranks(angles.cos().mean(dim=-3), world_size)
        imaginary = _average_across_ranks(angles.sin().mean(dim=-3), world_size)
        discrepancy = (real - self.phi).square() + imaginary.square()
        return (discrepancy @ self.weights) * projections.shape[-2] * world_size


class SlicedSIGReg(nn.Module):
    """Average the Epps-Pulley statistic over unit Gaussian directions."""

    def __init__(self, *, num_slices: int, n_points: int, t_max: float) -> None:
        super().__init__()
        if num_slices < 1:
            raise ValueError("num_slices must be positive")
        self.num_slices = num_slices
        # Keep the counter and univariate buffers in the original checkpoint layout.
        self.register_buffer("global_step", torch.zeros((), dtype=torch.long))
        self.univariate_test = _EppsPulley(n_points=n_points, t_max=t_max)
        self._generator = None

    def forward(self, samples: Tensor) -> Tensor:
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        with torch.no_grad():
            counter = self.global_step.clone()
            if world_size > 1:
                dist.all_reduce(counter, op=dist.ReduceOp.MAX)
            if self._generator is None or self._generator.device != samples.device:
                self._generator = torch.Generator(device=samples.device)
            self._generator.manual_seed(counter.item())
            directions = torch.randn(
                (samples.shape[-1], self.num_slices),
                device=samples.device,
                dtype=samples.dtype,
                generator=self._generator,
            )
            directions /= directions.norm(p=2, dim=0)
            self.global_step.add_(1)

        return self.univariate_test(samples @ directions, world_size).mean()
