"""Permutation-equivariant Transformer blocks for unordered cell populations."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class RMSNorm(nn.Module):
    """LLaMA-style RMS normalization with float32 variance accumulation."""

    def __init__(
        self, hidden_size: int, eps: float = 1e-6, elementwise_affine: bool = True
    ) -> None:
        super().__init__()
        self.eps = float(eps)
        if elementwise_affine:
            self.weight: Optional[nn.Parameter] = nn.Parameter(torch.ones(hidden_size))
        else:
            self.register_parameter("weight", None)

    def forward(self, hidden_states: Tensor) -> Tensor:
        input_dtype = hidden_states.dtype
        normalized = hidden_states.float()
        variance = normalized.square().mean(dim=-1, keepdim=True)
        normalized = normalized * torch.rsqrt(variance + self.eps)
        normalized = normalized.to(dtype=input_dtype)
        if self.weight is not None:
            normalized = normalized * self.weight
        return normalized


class SwiGLU(nn.Module):
    """Bias-free gated feed-forward layer used by LLaMA-like blocks."""

    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        return self.dropout(self.down_proj(hidden_states))


class SetSelfAttention(nn.Module):
    """Full self-attention with no positional or causal mechanism."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        attention_dropout: float = 0.0,
        output_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_size // self.num_heads
        self.attention_dropout = float(attention_dropout)
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.output_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.output_dropout = nn.Dropout(output_dropout)

    def forward(self, hidden_states: Tensor) -> Tensor:
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError("hidden_states must have shape [batch, cells, hidden_size]")
        batch_size, num_cells, _ = hidden_states.shape
        qkv = self.qkv_proj(hidden_states).view(
            batch_size, num_cells, 3, self.num_heads, self.head_dim
        )
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False,
        )
        attended = (
            attended.transpose(1, 2).contiguous().view(batch_size, num_cells, self.hidden_size)
        )
        return self.output_dropout(self.output_proj(attended))


class SetTransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        rms_norm_eps: float = 1e-6,
        layer_scale_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.attention = SetSelfAttention(
            hidden_size,
            num_heads,
            attention_dropout=attention_dropout,
            output_dropout=hidden_dropout,
        )
        self.feed_forward_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.feed_forward = SwiGLU(hidden_size, intermediate_size, dropout=hidden_dropout)
        self.attention_scale = nn.Parameter(torch.full((hidden_size,), float(layer_scale_init)))
        self.feed_forward_scale = nn.Parameter(torch.full((hidden_size,), float(layer_scale_init)))

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = hidden_states + self.attention_scale * self.attention(
            self.attention_norm(hidden_states)
        )
        return hidden_states + self.feed_forward_scale * self.feed_forward(
            self.feed_forward_norm(hidden_states)
        )


class SetTransformer(nn.Module):
    """A LLaMA-like stack that is equivariant to cell permutations."""

    def __init__(
        self,
        hidden_size: int,
        depth: int,
        num_heads: int,
        intermediate_size: int,
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        rms_norm_eps: float = 1e-6,
        layer_scale_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.layers = nn.ModuleList(
            [
                SetTransformerBlock(
                    hidden_size,
                    num_heads,
                    intermediate_size,
                    attention_dropout=attention_dropout,
                    hidden_dropout=hidden_dropout,
                    rms_norm_eps=rms_norm_eps,
                    layer_scale_init=layer_scale_init,
                )
                for _ in range(int(depth))
            ]
        )
        self.final_norm = RMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(self, hidden_states: Tensor) -> Tensor:
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError("hidden_states must have shape [batch, cells, hidden_size]")
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.final_norm(hidden_states)


def _modulate(hidden_states: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return hidden_states * (1.0 + scale) + shift


class AdaLNZeroSetBlock(nn.Module):
    """Full-attention predictor block conditioned by one action per population."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        rms_norm_eps: float = 1e-6,
        modulation_weight_std: float = 0.0,
        gate_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.modulation_weight_std = float(modulation_weight_std)
        self.gate_init = float(gate_init)
        self.attention_norm = RMSNorm(hidden_size, eps=rms_norm_eps, elementwise_affine=False)
        self.attention = SetSelfAttention(
            hidden_size,
            num_heads,
            attention_dropout=attention_dropout,
            output_dropout=hidden_dropout,
        )
        self.feed_forward_norm = RMSNorm(hidden_size, eps=rms_norm_eps, elementwise_affine=False)
        self.feed_forward = SwiGLU(hidden_size, intermediate_size, dropout=hidden_dropout)
        self.adaln_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.reset_adaln_parameters()

    def reset_adaln_parameters(self) -> None:
        modulation = self.adaln_modulation[-1]
        if self.modulation_weight_std == 0:
            nn.init.zeros_(modulation.weight)
        else:
            nn.init.normal_(modulation.weight, mean=0.0, std=self.modulation_weight_std)
        nn.init.zeros_(modulation.bias)
        with torch.no_grad():
            bias = modulation.bias.view(6, self.hidden_size)
            bias[2].fill_(self.gate_init)
            bias[5].fill_(self.gate_init)

    def forward(self, hidden_states: Tensor, action_embedding: Tensor) -> Tensor:
        if action_embedding.ndim != 2:
            raise ValueError("action_embedding must have shape [batch, hidden_size]")
        if (
            action_embedding.shape[0] != hidden_states.shape[0]
            or action_embedding.shape[1] != self.hidden_size
        ):
            raise ValueError("action_embedding does not align with hidden_states")
        modulation = self.adaln_modulation(action_embedding).unsqueeze(1)
        shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn = modulation.chunk(
            6, dim=-1
        )
        hidden_states = hidden_states + gate_attn * self.attention(
            _modulate(self.attention_norm(hidden_states), shift_attn, scale_attn)
        )
        return hidden_states + gate_ffn * self.feed_forward(
            _modulate(self.feed_forward_norm(hidden_states), shift_ffn, scale_ffn)
        )


class ActionConditionedPredictor(nn.Module):
    """Predict a perturbed set while preserving its input cell permutation."""

    def __init__(
        self,
        hidden_size: int,
        depth: int,
        num_heads: int,
        intermediate_size: int,
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        rms_norm_eps: float = 1e-6,
        modulation_weight_std: float = 0.0,
        gate_init: float = 0.0,
        action_input_scale: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.action_input_scale = float(action_input_scale)
        self.action_input_projection: Optional[nn.Linear] = (
            nn.Linear(hidden_size, hidden_size, bias=False)
            if self.action_input_scale > 0
            else None
        )
        self.layers = nn.ModuleList(
            [
                AdaLNZeroSetBlock(
                    hidden_size,
                    num_heads,
                    intermediate_size,
                    attention_dropout=attention_dropout,
                    hidden_dropout=hidden_dropout,
                    rms_norm_eps=rms_norm_eps,
                    modulation_weight_std=modulation_weight_std,
                    gate_init=gate_init,
                )
                for _ in range(int(depth))
            ]
        )
        self.final_norm = RMSNorm(hidden_size, eps=rms_norm_eps)

    def reset_adaln_parameters(self) -> None:
        for layer in self.layers:
            layer.reset_adaln_parameters()

    def forward(self, hidden_states: Tensor, action_embedding: Tensor) -> Tensor:
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError("hidden_states must have shape [batch, cells, hidden_size]")
        if self.action_input_projection is not None:
            hidden_states = hidden_states + self.action_input_scale * self.action_input_projection(
                action_embedding
            ).unsqueeze(1)
        for layer in self.layers:
            hidden_states = layer(hidden_states, action_embedding)
        return self.final_norm(hidden_states)


class ActionTokenAdaLNZeroPredictor(ActionConditionedPredictor):
    """Condition AdaLN-Zero on the fixed mean of encoded action tokens."""

    def __init__(
        self,
        hidden_size: int,
        depth: int,
        num_heads: int,
        intermediate_size: int,
        *,
        num_action_tokens: int = 2,
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        rms_norm_eps: float = 1e-6,
        modulation_weight_std: float = 0.0,
        gate_init: float = 0.0,
        action_input_scale: float = 0.0,
    ) -> None:
        if (
            isinstance(num_action_tokens, bool)
            or not isinstance(num_action_tokens, int)
            or num_action_tokens <= 0
        ):
            raise ValueError("num_action_tokens must be a positive integer")
        super().__init__(
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            attention_dropout=attention_dropout,
            hidden_dropout=hidden_dropout,
            rms_norm_eps=rms_norm_eps,
            modulation_weight_std=modulation_weight_std,
            gate_init=gate_init,
            action_input_scale=action_input_scale,
        )
        self.num_action_tokens = num_action_tokens

    def forward(self, hidden_states: Tensor, action_tokens: Tensor) -> Tensor:
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError("hidden_states must have shape [batch, cells, hidden_size]")
        if action_tokens.ndim != 3:
            raise ValueError("action_tokens must have shape [batch, action_tokens, hidden_size]")
        if (
            action_tokens.shape[0] != hidden_states.shape[0]
            or action_tokens.shape[1] != self.num_action_tokens
            or action_tokens.shape[2] != self.hidden_size
        ):
            raise ValueError("action_tokens do not align with hidden_states")
        return super().forward(hidden_states, action_tokens.mean(dim=1))


class ActionTokenSetTransformerPredictor(nn.Module):
    """Predict cells by jointly attending over drug, dose, and control tokens."""

    def __init__(
        self,
        hidden_size: int,
        depth: int,
        num_heads: int,
        intermediate_size: int,
        *,
        num_action_tokens: int = 2,
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        rms_norm_eps: float = 1e-6,
        layer_scale_init: float = 0.1,
    ) -> None:
        super().__init__()
        if num_action_tokens <= 0:
            raise ValueError("num_action_tokens must be positive")
        self.hidden_size = int(hidden_size)
        self.num_action_tokens = int(num_action_tokens)
        self.transformer = SetTransformer(
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            attention_dropout=attention_dropout,
            hidden_dropout=hidden_dropout,
            rms_norm_eps=rms_norm_eps,
            layer_scale_init=layer_scale_init,
        )

    def forward(self, hidden_states: Tensor, action_tokens: Tensor) -> Tensor:
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError("hidden_states must have shape [batch, cells, hidden_size]")
        if action_tokens.ndim != 3:
            raise ValueError("action_tokens must have shape [batch, action_tokens, hidden_size]")
        if (
            action_tokens.shape[0] != hidden_states.shape[0]
            or action_tokens.shape[1] != self.num_action_tokens
            or action_tokens.shape[2] != self.hidden_size
        ):
            raise ValueError("action_tokens do not align with hidden_states")
        combined = torch.cat((action_tokens, hidden_states), dim=1)
        transformed = self.transformer(combined)
        return transformed[:, self.num_action_tokens :]


__all__ = [
    "ActionConditionedPredictor",
    "ActionTokenAdaLNZeroPredictor",
    "ActionTokenSetTransformerPredictor",
    "AdaLNZeroSetBlock",
    "RMSNorm",
    "SetSelfAttention",
    "SetTransformer",
    "SetTransformerBlock",
    "SwiGLU",
]
