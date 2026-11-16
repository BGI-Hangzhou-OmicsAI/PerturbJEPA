"""Expression-weighted gene bags that produce one token per cell."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

DEFAULT_NUM_GENES = 62_710


def _batch_field(batch: Any, name: str) -> Tensor:
    if isinstance(batch, Mapping):
        value = batch[name]
    else:
        try:
            value = getattr(batch, name)
        except AttributeError as exc:
            raise TypeError("batch must be a mapping or expose {!r}".format(name)) from exc
    if not isinstance(value, Tensor):
        raise TypeError("batch field {!r} must be a torch.Tensor".format(name))
    return value


class ExpressionWeightedGeneBag(nn.Module):
    """Encode sparse expression rows as weighted gene-embedding bags.

    ``cell_offsets`` uses the ``include_last_offset=True`` convention and describes
    a shared bank of unique cell bags. ``target_cell_map`` and ``control_cell_map``
    select from that bank and may repeat entries. Empty bags use a learned cell
    embedding, initialized to zero, instead of dividing by zero.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_embeddings: int = DEFAULT_NUM_GENES,
        *,
        pooling_normalization: str = "mean",
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        if isinstance(embedding_dim, bool) or int(embedding_dim) != embedding_dim:
            raise ValueError("embedding_dim must be a positive integer")
        if isinstance(num_embeddings, bool) or int(num_embeddings) != num_embeddings:
            raise ValueError("num_embeddings must be a positive integer")
        embedding_dim = int(embedding_dim)
        num_embeddings = int(num_embeddings)
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be a positive integer")
        if num_embeddings <= 0:
            raise ValueError("num_embeddings must be a positive integer")
        pooling_normalization = str(pooling_normalization).lower()
        if pooling_normalization not in ("mean", "l2", "none"):
            raise ValueError("pooling_normalization must be 'mean', 'l2', or 'none'")

        self.pooling_normalization = pooling_normalization
        self.embedding_bag = nn.EmbeddingBag(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            mode="sum",
            include_last_offset=True,
            device=device,
            dtype=dtype,
        )
        self.empty_cell_embedding = nn.Parameter(
            torch.zeros(embedding_dim, device=device, dtype=dtype)
        )
        self.register_buffer(
            "_normalization_embedding",
            torch.ones(num_embeddings, 1, device=device, dtype=dtype),
            persistent=False,
        )

    def _l2_value_sums(self, gene_ids: Tensor, gene_values: Tensor, cell_offsets: Tensor) -> Tensor:
        values = F.embedding_bag(
            gene_ids,
            self._normalization_embedding,
            cell_offsets,
            mode="sum",
            per_sample_weights=gene_values.square(),
            include_last_offset=True,
        )
        return values.squeeze(-1)

    def forward(self, batch: Any) -> Tuple[Tensor, Tensor]:
        gene_ids = _batch_field(batch, "gene_ids")
        gene_values = _batch_field(batch, "gene_values")
        cell_offsets = _batch_field(batch, "cell_offsets")
        cell_value_sums = _batch_field(batch, "cell_value_sums")
        target_cell_map = _batch_field(batch, "target_cell_map")
        control_cell_map = _batch_field(batch, "control_cell_map")

        if gene_ids.ndim != 1 or gene_values.ndim != 1:
            raise ValueError("gene_ids and gene_values must be one-dimensional")
        if gene_ids.shape != gene_values.shape:
            raise ValueError("gene_ids and gene_values must have identical shapes")
        if cell_offsets.ndim != 1 or cell_value_sums.ndim != 1:
            raise ValueError("cell_offsets and cell_value_sums must be one-dimensional")
        if cell_offsets.numel() != cell_value_sums.numel() + 1:
            raise ValueError("cell_offsets must contain one more item than cell_value_sums")
        if target_cell_map.ndim != 2 or control_cell_map.ndim != 2:
            raise ValueError("target_cell_map and control_cell_map must be two-dimensional")
        if target_cell_map.shape != control_cell_map.shape:
            raise ValueError("target_cell_map and control_cell_map must have identical shapes")

        weight = self.embedding_bag.weight
        if gene_ids.device != weight.device:
            raise ValueError("sparse batch and gene embeddings must be on the same device")
        for name, value in (
            ("gene_values", gene_values),
            ("cell_offsets", cell_offsets),
            ("cell_value_sums", cell_value_sums),
            ("target_cell_map", target_cell_map),
            ("control_cell_map", control_cell_map),
        ):
            if value.device != weight.device:
                raise ValueError(
                    "batch field {!r} and gene embeddings must be on the same device".format(name)
                )

        gene_values = gene_values.to(dtype=weight.dtype)
        cell_value_sums = cell_value_sums.to(dtype=weight.dtype)
        weighted_sums = self.embedding_bag(
            gene_ids,
            cell_offsets,
            per_sample_weights=gene_values,
        )
        if self.pooling_normalization == "none":
            nonempty = cell_offsets[1:] > cell_offsets[:-1]
            unique_tokens = weighted_sums
        else:
            normalization = (
                cell_value_sums
                if self.pooling_normalization == "mean"
                else self._l2_value_sums(gene_ids, gene_values, cell_offsets).sqrt()
            )
            nonempty = normalization > 0
            safe_normalization = torch.where(
                nonempty, normalization, torch.ones_like(normalization)
            )
            unique_tokens = weighted_sums / safe_normalization.unsqueeze(-1)
        unique_tokens = torch.where(
            nonempty.unsqueeze(-1),
            unique_tokens,
            self.empty_cell_embedding.expand_as(unique_tokens),
        )

        target = unique_tokens[target_cell_map.to(dtype=torch.long)]
        control = unique_tokens[control_cell_map.to(dtype=torch.long)]
        return target, control


__all__ = ["DEFAULT_NUM_GENES", "ExpressionWeightedGeneBag"]
