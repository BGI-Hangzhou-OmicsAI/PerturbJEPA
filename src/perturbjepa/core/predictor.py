"""Action-conditioned per-cell predictor for perturbed populations."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .set_transformer import RMSNorm


class ActionConditionedMLPBlock(nn.Module):
    """Apply the same drug-conditioned residual MLP to every cell."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        hidden_dropout: float = 0.0,
        rms_norm_eps: float = 1e-6,
        layer_scale_init: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.input_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.input_projection = nn.Linear(2 * hidden_size, intermediate_size, bias=False)
        self.output_projection = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(hidden_dropout)
        self.layer_scale = nn.Parameter(torch.full((hidden_size,), float(layer_scale_init)))

    def forward(self, hidden_states: Tensor, action_embedding: Tensor) -> Tensor:
        action = action_embedding.unsqueeze(1).expand(-1, hidden_states.shape[1], -1)
        conditioned = torch.cat((self.input_norm(hidden_states), action), dim=-1)
        update = self.output_projection(F.silu(self.input_projection(conditioned)))
        return hidden_states + self.layer_scale * self.dropout(update)


class ActionConditionedMLPPredictor(nn.Module):
    """Predict each perturbed cell from its control latent and the drug action.

    Cells share weights but never exchange values, so the module is permutation
    equivariant without attention or any other operation over the cell axis.
    """

    def __init__(
        self,
        hidden_size: int,
        depth: int,
        intermediate_size: int,
        *,
        hidden_dropout: float = 0.0,
        rms_norm_eps: float = 1e-6,
        layer_scale_init: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.layers = nn.ModuleList(
            [
                ActionConditionedMLPBlock(
                    hidden_size,
                    intermediate_size,
                    hidden_dropout=hidden_dropout,
                    rms_norm_eps=rms_norm_eps,
                    layer_scale_init=layer_scale_init,
                )
                for _ in range(int(depth))
            ]
        )
        self.final_norm = RMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(self, hidden_states: Tensor, action_embedding: Tensor) -> Tensor:
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError("hidden_states must have shape [batch, cells, hidden_size]")
        if action_embedding.ndim != 2:
            raise ValueError("action_embedding must have shape [batch, hidden_size]")
        if (
            action_embedding.shape[0] != hidden_states.shape[0]
            or action_embedding.shape[1] != self.hidden_size
        ):
            raise ValueError("action_embedding does not align with hidden_states")
        for layer in self.layers:
            hidden_states = layer(hidden_states, action_embedding)
        return self.final_norm(hidden_states)


__all__ = ["ActionConditionedMLPBlock", "ActionConditionedMLPPredictor"]
