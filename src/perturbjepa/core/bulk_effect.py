"""Raw-expression population effects independent of the learned gene tokenizer."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) <= 0:
        raise ValueError("{} must be a positive integer".format(name))
    return int(value)


def build_bulk_difference(
    gene_ids: Tensor,
    gene_values: Tensor,
    cell_offsets: Tensor,
    target_cell_map: Tensor,
    control_cell_map: Tensor,
    *,
    num_genes: int,
) -> Tensor:
    """Return ``mean(X_target) - mean(X_control)`` with shape ``[B, G]``.

    The expression values are used exactly as supplied by the sparse dataset
    (including its fixed expression transform). No learned gene embeddings or
    per-cell normalization participate. Both maps index the physical cell bank;
    repeated indices retain their sampling multiplicity, and empty cells still
    count toward their population's mean denominator.

    Sparse ``X.T @ membership`` needs only O(nnz + U*B + B*G) storage, where U
    counts physical cells. It never materializes a dense cell-by-gene matrix.
    Values, membership weights, and accumulation stay FP32 under autocast.
    """
    num_genes = _positive_integer(num_genes, "num_genes")
    tensors = (
        ("gene_ids", gene_ids),
        ("gene_values", gene_values),
        ("cell_offsets", cell_offsets),
        ("target_cell_map", target_cell_map),
        ("control_cell_map", control_cell_map),
    )
    for name, value in tensors:
        if not isinstance(value, Tensor):
            raise TypeError("{} must be a torch.Tensor".format(name))
    if gene_ids.ndim != 1 or gene_values.ndim != 1:
        raise ValueError("gene_ids and gene_values must be one-dimensional")
    if gene_ids.shape != gene_values.shape:
        raise ValueError("gene_ids and gene_values must have identical shapes")
    if cell_offsets.ndim != 1 or cell_offsets.numel() < 2:
        raise ValueError("cell_offsets must describe at least one physical cell")
    if target_cell_map.ndim != 2 or control_cell_map.ndim != 2:
        raise ValueError("target_cell_map and control_cell_map must be two-dimensional")
    if target_cell_map.shape[0] != control_cell_map.shape[0]:
        raise ValueError("target and control maps must have the same batch size")
    if min(*target_cell_map.shape, *control_cell_map.shape) <= 0:
        raise ValueError("population maps must contain at least one group and cell")
    for name, value in tensors:
        if value.device != gene_values.device:
            raise ValueError("all sparse batch tensors must be on the same device")
        if name != "gene_values" and value.dtype not in (torch.int32, torch.int64):
            raise TypeError("{} must have int32 or int64 dtype".format(name))
    if not gene_values.is_floating_point():
        raise TypeError("gene_values must have floating-point dtype")

    batch_size = target_cell_map.shape[0]
    num_cells = cell_offsets.numel() - 1
    # The logical maps are tiny compared with the expression entries. Accumulate
    # their signed multiplicities before touching expression, including cells
    # reused across different groups or on both sides of a pair.
    membership = torch.zeros(
        num_cells * batch_size, dtype=torch.float32, device=gene_values.device
    )
    group_ids = torch.arange(batch_size, device=gene_values.device).unsqueeze(1)
    for cell_map, sign in ((target_cell_map, 1.0), (control_cell_map, -1.0)):
        positions = cell_map.long() * batch_size + group_ids
        counts = torch.zeros_like(membership, dtype=torch.int32)
        counts.scatter_add_(
            0,
            positions.flatten(),
            torch.ones(cell_map.numel(), dtype=torch.int32, device=gene_values.device),
        )
        # Count in integers first: identical sampled populations cancel exactly,
        # including repetitions whose floating-point 1/N weights would drift.
        membership.add_(counts.float() * (sign / cell_map.shape[1]))
    membership = membership.view(num_cells, batch_size)
    cell_ids = torch.repeat_interleave(
        torch.arange(num_cells, device=gene_values.device),
        (cell_offsets[1:] - cell_offsets[:-1]).long(),
        output_size=gene_ids.numel(),
    )
    expression_transpose = torch.sparse_coo_tensor(
        torch.stack((gene_ids.long(), cell_ids)),
        gene_values.float(),
        size=(num_genes, num_cells),
        device=gene_values.device,
        dtype=torch.float32,
    )
    with torch.autocast(device_type=gene_values.device.type, enabled=False):
        return torch.sparse.mm(expression_transpose, membership).transpose(0, 1).contiguous()


class RawBulkEffectEncoder(nn.Module):
    """A small odd MLP mapping a raw bulk difference ``[B,G]`` to ``[B,D]``.

    Bias-free linear layers and tanh guarantee f(-x)=-f(x) and f(0)=0 without
    computing two forward passes. There is no per-sample output normalization,
    so an arbitrarily small difference is not forced to have unit magnitude.
    ``input_scale`` is a fixed positive multiplier, never a learned escape gate.
    """

    def __init__(
        self,
        num_genes: int,
        hidden_size: int,
        intermediate_size: int = 128,
        *,
        input_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_genes = _positive_integer(num_genes, "num_genes")
        self.hidden_size = _positive_integer(hidden_size, "hidden_size")
        self.intermediate_size = _positive_integer(intermediate_size, "intermediate_size")
        self.input_scale = float(input_scale)
        if not math.isfinite(self.input_scale) or self.input_scale <= 0:
            raise ValueError("input_scale must be finite and positive")
        self.input_projection = nn.Linear(self.num_genes, self.intermediate_size, bias=False)
        self.output_projection = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        # HF initializes Linear leaves itself. These hints let its initializer
        # preserve fan-in scaling without reinitializing already-loaded children
        # from a parent-module initializer.
        self.input_projection._bulk_effect_init_std = self.num_genes ** -0.5
        self.output_projection._bulk_effect_init_std = self.intermediate_size ** -0.5
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize each projection using its own fan-in, including HF reloads."""
        nn.init.normal_(self.input_projection.weight, std=self.num_genes ** -0.5)
        nn.init.normal_(self.output_projection.weight, std=self.intermediate_size ** -0.5)

    def forward(self, bulk_difference: Tensor) -> Tensor:
        if bulk_difference.ndim != 2 or bulk_difference.shape[1] != self.num_genes:
            raise ValueError("bulk_difference must have shape [batch, num_genes]")
        if bulk_difference.device != self.input_projection.weight.device:
            raise ValueError("bulk_difference and effect encoder must be on the same device")
        values = bulk_difference.to(dtype=self.input_projection.weight.dtype) * self.input_scale
        return self.output_projection(torch.tanh(self.input_projection(values)))


__all__ = ["RawBulkEffectEncoder", "build_bulk_difference"]
