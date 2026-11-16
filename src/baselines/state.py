"""STATE baseline on Parse10M, ported from Perturb's project-owned implementation.

Model and transformer classes retain the reference equations and initialization.
The Parse10M runner below shares JEPA's data, collator, sampler and retrieval.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from geomloss import SamplesLoss
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import PretrainedConfig, PreTrainedModel, TrainingArguments, set_seed
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import ModelOutput

from data import Parse10MDataset, condition_labels
from data.visit import visit_sampling_config
from perturbjepa.batch import SparseBatchCollator
from perturbjepa.retrieval import sha256, write_json
from perturbjepa.selection import finish_training
from perturbjepa.trainer import Parse10MTrainer
from perturbjepa.visit_trainer import VisitTrainerMixin, last_visit_checkpoint


class StateRMSNorm(nn.Module):
    """RMS normalization used by the released STATE backbone."""

    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = float(eps)

    def forward(self, hidden_states: Tensor) -> Tensor:
        input_dtype = hidden_states.dtype
        normalized = hidden_states.float()
        variance = normalized.square().mean(dim=-1, keepdim=True)
        normalized = normalized * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * normalized.to(dtype=input_dtype)

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class StateSelfAttention(nn.Module):
    """Full bidirectional attention without positional encodings or a causal mask."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        attention_dropout: float,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_attention_heads = int(num_attention_heads)
        self.num_key_value_heads = int(num_key_value_heads)
        self.head_dim = int(head_dim)
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.attention_dropout = float(attention_dropout)
        self.is_causal = False

        query_size = self.num_attention_heads * self.head_dim
        key_value_size = self.num_key_value_heads * self.head_dim
        self.q_proj = nn.Linear(self.hidden_size, query_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, key_value_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, key_value_size, bias=False)
        self.o_proj = nn.Linear(query_size, self.hidden_size, bias=False)

    def _split_heads(self, value: Tensor, num_heads: int) -> Tensor:
        batch_size, sequence_length, _ = value.shape
        return value.view(batch_size, sequence_length, num_heads, self.head_dim).transpose(1, 2)

    def forward(self, hidden_states: Tensor) -> Tensor:
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError("hidden_states must have shape [batch, cells, hidden_size]")

        queries = self._split_heads(self.q_proj(hidden_states), self.num_attention_heads)
        keys = self._split_heads(self.k_proj(hidden_states), self.num_key_value_heads)
        values = self._split_heads(self.v_proj(hidden_states), self.num_key_value_heads)
        if self.num_key_value_groups != 1:
            keys = keys.repeat_interleave(self.num_key_value_groups, dim=1)
            values = values.repeat_interleave(self.num_key_value_groups, dim=1)

        attended = F.scaled_dot_product_attention(
            queries,
            keys,
            values,
            attn_mask=None,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False,
        )
        batch_size, _, sequence_length, _ = attended.shape
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size,
            sequence_length,
            self.num_attention_heads * self.head_dim,
        )
        return self.o_proj(attended)


class StateMLP(nn.Module):
    """SwiGLU feed-forward block used by STATE's transformer."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class StateTransformerLayer(nn.Module):
    """Pre-norm attention and MLP residual block."""

    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        rms_norm_eps: float,
        attention_dropout: float,
    ) -> None:
        super().__init__()
        self.self_attn = StateSelfAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            attention_dropout=attention_dropout,
        )
        self.mlp = StateMLP(hidden_size, intermediate_size)
        self.input_layernorm = StateRMSNorm(hidden_size, rms_norm_eps)
        self.post_attention_layernorm = StateRMSNorm(hidden_size, rms_norm_eps)

    def forward(self, hidden_states: Tensor) -> Tensor:
        residual = hidden_states
        hidden_states = residual + self.self_attn(self.input_layernorm(hidden_states))
        residual = hidden_states
        return residual + self.mlp(self.post_attention_layernorm(hidden_states))


class StateTransformerBackbone(nn.Module):
    """Permutation-equivariant transformer over a fixed-size set of cells."""

    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        max_sequence_length: int,
        rms_norm_eps: float,
        attention_dropout: float,
        initializer_range: float,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.max_sequence_length = int(max_sequence_length)
        self.initializer_range = float(initializer_range)
        self.layers = nn.ModuleList(
            [
                StateTransformerLayer(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    num_attention_heads=num_attention_heads,
                    num_key_value_heads=num_key_value_heads,
                    head_dim=head_dim,
                    rms_norm_eps=rms_norm_eps,
                    attention_dropout=attention_dropout,
                )
                for _ in range(num_hidden_layers)
            ]
        )
        self.norm = StateRMSNorm(hidden_size, rms_norm_eps)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=self.initializer_range)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, StateRMSNorm):
                nn.init.ones_(module.weight)

    def forward(self, hidden_states: Tensor) -> Tensor:
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError("hidden_states must have shape [batch, cells, hidden_size]")
        if hidden_states.shape[1] > self.max_sequence_length:
            raise ValueError(
                f"cell sequence length {hidden_states.shape[1]} exceeds configured maximum "
                f"{self.max_sequence_length}"
            )
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.norm(hidden_states)


class StateModelConfig(PretrainedConfig):
    """Serializable architecture and data identity for STATE-on-Parse10M."""

    model_type = "state_transition"

    def __init__(
        self,
        *,
        num_genes: int = 2000,
        num_conditions: int = 1138,
        cell_set_size: int = 256,
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        num_hidden_layers: int = 8,
        num_attention_heads: int = 12,
        num_key_value_heads: int = 12,
        head_dim: int | None = 64,
        max_position_embeddings: int = 256,
        rms_norm_eps: float = 1.0e-6,
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        initializer_range: float = 0.02,
        energy_blur: float = 0.05,
        predict_residual: bool = True,
        output_relu: bool = True,
        condition_labels: Sequence[str] | None = None,
        gene_indices: Sequence[int] | None = None,
        gene_names: Sequence[str] | None = None,
        action_vocabulary_sha256: str = "",
        gene_vocabulary_sha256: str = "",
        expression_transform: str = "normalize_total_log1p",
        normalize_target_sum: float = 10_000.0,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("tie_word_embeddings", False)
        super().__init__(**kwargs)
        self.num_genes = int(num_genes)
        self.num_conditions = int(num_conditions)
        self.cell_set_size = int(cell_set_size)
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.num_key_value_heads = int(num_key_value_heads)
        self.head_dim = (
            self.hidden_size // self.num_attention_heads if head_dim is None else int(head_dim)
        )
        self.max_position_embeddings = int(max_position_embeddings)
        self.rms_norm_eps = float(rms_norm_eps)
        self.attention_dropout = float(attention_dropout)
        self.hidden_dropout = float(hidden_dropout)
        self.initializer_range = float(initializer_range)
        self.energy_blur = float(energy_blur)
        self.predict_residual = bool(predict_residual)
        self.output_relu = bool(output_relu)
        self.condition_labels = (
            [str(index) for index in range(self.num_conditions)]
            if condition_labels is None
            else [str(value) for value in condition_labels]
        )
        self.gene_indices = (
            list(range(self.num_genes))
            if gene_indices is None
            else [int(value) for value in gene_indices]
        )
        self.gene_names = (
            [f"gene_{index}" for index in range(self.num_genes)]
            if gene_names is None
            else [str(value) for value in gene_names]
        )
        self.action_vocabulary_sha256 = str(action_vocabulary_sha256)
        self.gene_vocabulary_sha256 = str(gene_vocabulary_sha256)
        self.expression_transform = str(expression_transform)
        self.normalize_target_sum = float(normalize_target_sum)
        self._validate()

    def _validate(self) -> None:
        positive_integers = (
            "num_genes",
            "num_conditions",
            "cell_set_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "max_position_embeddings",
        )
        for name in positive_integers:
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError("hidden_size must equal num_attention_heads * head_dim")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.max_position_embeddings < self.cell_set_size:
            raise ValueError("max_position_embeddings must cover cell_set_size")
        if not math.isfinite(self.rms_norm_eps) or self.rms_norm_eps <= 0.0:
            raise ValueError("rms_norm_eps must be finite and positive")
        if not math.isfinite(self.initializer_range) or self.initializer_range <= 0.0:
            raise ValueError("initializer_range must be finite and positive")
        if not math.isfinite(self.energy_blur) or self.energy_blur <= 0.0:
            raise ValueError("energy_blur must be finite and positive")
        if self.attention_dropout != 0.0 or self.hidden_dropout != 0.0:
            raise ValueError("released STATE uses zero attention and hidden dropout")
        if len(self.condition_labels) != self.num_conditions:
            raise ValueError("condition_labels must contain one label per condition")
        if len(set(self.condition_labels)) != len(self.condition_labels):
            raise ValueError("condition_labels must not contain duplicates")
        if len(self.gene_indices) != self.num_genes:
            raise ValueError("gene_indices must contain one index per model gene")
        if len(self.gene_names) != self.num_genes:
            raise ValueError("gene_names must contain one name per model gene")
        if len(set(self.gene_indices)) != len(self.gene_indices):
            raise ValueError("gene_indices must not contain duplicates")
        if len(set(self.gene_names)) != len(self.gene_names):
            raise ValueError("gene_names must not contain duplicates")
        if any(index < 0 for index in self.gene_indices):
            raise ValueError("gene_indices must be nonnegative")
        if self.expression_transform != "normalize_total_log1p":
            raise ValueError("expression_transform must be 'normalize_total_log1p'")
        if not math.isfinite(self.normalize_target_sum):
            raise ValueError("normalize_target_sum must be finite")
        if self.normalize_target_sum != 10_000.0:
            raise ValueError("normalize_target_sum must be 10000 for CP10K inputs")


@dataclass
class StateModelOutput(ModelOutput):
    """Training output plus the expression populations used by the objective."""

    loss: Tensor | None = None
    distribution_loss: Tensor | None = None
    per_set_distribution_loss: Tensor | None = None
    predicted_perturbed: Tensor | None = None
    encoded_target: Tensor | None = None
    encoded_control: Tensor | None = None
    diagnostics: dict[str, Tensor] | None = None


def _as_long_tensor(value: Any, *, device: torch.device, name: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    return value.to(device=device, dtype=torch.long)


class _DenseGeneTokenizer(nn.Module):
    """Densify the selected-gene rows while preserving logical population maps."""

    def __init__(self, num_genes: int, cell_set_size: int) -> None:
        super().__init__()
        self.num_genes = int(num_genes)
        self.cell_set_size = int(cell_set_size)

    def forward(self, population_inputs: Mapping[str, Any]) -> tuple[Tensor, Tensor]:
        if not isinstance(population_inputs, Mapping):
            raise TypeError("population_inputs must be a mapping")
        try:
            gene_ids = population_inputs["gene_ids"]
            gene_values = population_inputs["gene_values"]
            cell_offsets = population_inputs["cell_offsets"]
            target_cell_map = population_inputs["target_cell_map"]
            control_cell_map = population_inputs["control_cell_map"]
        except KeyError as exc:
            raise KeyError(f"missing Parse10M sparse field: {exc.args[0]}") from exc
        if not isinstance(gene_values, Tensor) or gene_values.ndim != 1:
            raise ValueError("gene_values must be a one-dimensional torch.Tensor")
        device = gene_values.device
        gene_ids = _as_long_tensor(gene_ids, device=device, name="gene_ids")
        cell_offsets = _as_long_tensor(cell_offsets, device=device, name="cell_offsets")
        target_cell_map = _as_long_tensor(target_cell_map, device=device, name="target_cell_map")
        control_cell_map = _as_long_tensor(control_cell_map, device=device, name="control_cell_map")
        if gene_ids.ndim != 1:
            raise ValueError("gene_ids must be one-dimensional")
        if gene_ids.shape != gene_values.shape:
            raise ValueError("gene_ids and gene_values must have identical shapes")
        if cell_offsets.ndim != 1 or cell_offsets.numel() < 2:
            raise ValueError("cell_offsets must describe at least one physical cell")
        if target_cell_map.ndim != 2 or control_cell_map.ndim != 2:
            raise ValueError("target_cell_map and control_cell_map must be two-dimensional")
        if target_cell_map.shape != control_cell_map.shape:
            raise ValueError("target_cell_map and control_cell_map must have identical shapes")
        if target_cell_map.shape[1] != self.cell_set_size:
            raise ValueError("population maps do not match config.cell_set_size")
        unique_cells = int(cell_offsets.numel() - 1)
        # Content validation is useful for CPU-side collation tests. On CUDA,
        # these reductions would synchronize every training step; invalid sparse
        # indices still fail in the native scatter/index kernels below.
        if device.type == "cpu":
            if int(cell_offsets[0]) != 0 or int(cell_offsets[-1]) != gene_ids.numel():
                raise ValueError("cell_offsets do not delimit gene_ids")
            if bool((cell_offsets[1:] < cell_offsets[:-1]).any()):
                raise ValueError("cell_offsets must be monotonic")
            if gene_ids.numel() and (
                bool((gene_ids < 0).any()) or bool((gene_ids >= self.num_genes).any())
            ):
                raise ValueError("gene_ids contain an out-of-range selected-gene index")
            if gene_values.numel() and not bool(torch.isfinite(gene_values).all()):
                raise ValueError("gene_values contain non-finite values")
            if gene_values.numel() and bool((gene_values < 0).any()):
                raise ValueError("gene_values must be nonnegative CP10K+log1p values")
            for name, cell_map in (
                ("target_cell_map", target_cell_map),
                ("control_cell_map", control_cell_map),
            ):
                if cell_map.numel() and (
                    bool((cell_map < 0).any()) or bool((cell_map >= unique_cells).any())
                ):
                    raise ValueError(f"{name} references a missing physical cell")

        lengths = cell_offsets[1:] - cell_offsets[:-1]
        cell_ids = torch.repeat_interleave(
            torch.arange(unique_cells, device=device, dtype=torch.long), lengths
        )
        flat_indices = cell_ids * self.num_genes + gene_ids
        dense = gene_values.new_zeros(unique_cells * self.num_genes)
        if gene_values.numel():
            dense = dense.scatter_add(0, flat_indices, gene_values)
        dense = dense.view(unique_cells, self.num_genes)
        return dense[target_cell_map], dense[control_cell_map]


class _ConditionLinearEmbedding(nn.Linear):
    """A Linear(one_hot(condition)) represented without materializing one-hot rows."""

    def __init__(self, num_conditions: int, hidden_size: int) -> None:
        self.num_conditions = int(num_conditions)
        self.hidden_size = int(hidden_size)
        # Keep the released nn.Linear(num_conditions, hidden_size) parameter
        # layout and initialization so official checkpoints load directly.
        super().__init__(self.num_conditions, self.hidden_size, bias=True)

    def forward(self, condition_ids: Tensor) -> Tensor:
        if not isinstance(condition_ids, Tensor) or condition_ids.ndim != 1:
            raise ValueError("condition_ids must have shape [batch]")
        condition_ids = condition_ids.to(dtype=torch.long)
        if (
            condition_ids.device.type == "cpu"
            and condition_ids.numel()
            and (
                bool((condition_ids < 0).any())
                or bool((condition_ids >= self.num_conditions).any())
            )
        ):
            raise IndexError("condition_ids contains an out-of-range condition")
        return F.embedding(condition_ids, self.weight.transpose(0, 1)) + self.bias


class _StateTransitionPredictor(nn.Module):
    """Released STATE encoders, project-owned transformer, and residual head."""

    def __init__(self, config: StateModelConfig) -> None:
        super().__init__()
        self.num_genes = config.num_genes
        self.hidden_size = config.hidden_size
        self.cell_set_size = config.cell_set_size
        self.predict_residual = config.predict_residual
        self.output_relu = config.output_relu

        # These outer Linear modules deliberately retain torch.nn.Linear's
        # default initialization, matching the official STATE implementation.
        self.basal_encoder = nn.Linear(config.num_genes, config.hidden_size)
        self.transformer_backbone = StateTransformerBackbone(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            max_sequence_length=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            initializer_range=config.initializer_range,
            attention_dropout=config.attention_dropout,
        )
        self.project_out = nn.Linear(config.hidden_size, config.num_genes)

    def forward(self, controls: Tensor, action: Tensor) -> Tensor:
        if not isinstance(controls, Tensor) or controls.ndim != 3:
            raise ValueError("controls must have shape [batch, cells, genes]")
        if controls.shape[1:] != (self.cell_set_size, self.num_genes):
            raise ValueError("controls do not match the configured cell set and gene panel")
        if not isinstance(action, Tensor):
            raise TypeError("action must be a torch.Tensor")
        if action.ndim == 2:
            action = action.unsqueeze(1)
        if action.ndim != 3 or action.shape[0] != controls.shape[0]:
            raise ValueError("action must align with the control population batch")
        if action.shape[-1] != self.hidden_size or action.shape[1] not in (
            1,
            self.cell_set_size,
        ):
            raise ValueError("action has an invalid STATE hidden shape")

        control_hidden = self.basal_encoder(controls)
        hidden = control_hidden + action
        residual = self.transformer_backbone(hidden)
        decoded = self.project_out(residual + control_hidden if self.predict_residual else residual)
        return F.relu(decoded) if self.output_relu else decoded


class StateTransitionModel(PreTrainedModel):
    """STATE set-to-set model over PerturbJEPA's sparse Parse10M populations."""

    config_class = StateModelConfig
    base_model_prefix = "state_transition"
    main_input_name = "gene_ids"
    # Loss is a mean over population sets, not a token-count-normalized loss.
    # Tell HF to scale each microbatch for gradient accumulation even though
    # forward accepts unrelated sparse-batch metadata through **kwargs.
    accepts_loss_kwargs = False

    def __init__(self, config: StateModelConfig) -> None:
        super().__init__(config)
        self.gene_tokenizer = _DenseGeneTokenizer(config.num_genes, config.cell_set_size)
        self.encoder = nn.Identity()
        self.action_encoder = _ConditionLinearEmbedding(config.num_conditions, config.hidden_size)
        self.predictor = _StateTransitionPredictor(config)
        self.energy_loss = SamplesLoss(loss="energy", blur=config.energy_blur)
        # Do not call outer post_init(): official STATE leaves basal/action/output
        # Linear modules at torch.nn.Linear's default initialization.
        self._register_hf_runtime_metadata()

    def _register_hf_runtime_metadata(self) -> None:
        """Install HF bookkeeping normally created by ``post_init`` only."""

        self._tp_plan: dict[str, Any] = {}
        self._ep_plan: dict[str, Any] = {}
        self._pp_plan: dict[str, Any] = {}
        expanded_ties = getattr(self, "get_expanded_tied_weights_keys", None)
        self.all_tied_weights_keys = (
            expanded_ties(all_submodels=False) if callable(expanded_ties) else {}
        )
        if self.base_model is self:
            for destination, source in (
                ("_tp_plan", "base_model_tp_plan"),
                ("_ep_plan", "base_model_ep_plan"),
                ("_pp_plan", "base_model_pp_plan"),
            ):
                plan = getattr(self.config, source, None)
                if plan is not None:
                    setattr(self, destination, plan.copy())
        for name, module in self.named_children():
            for plan_name in ("_tp_plan", "_ep_plan", "_pp_plan"):
                if plan := getattr(module, plan_name, None):
                    getattr(self, plan_name).update(
                        {f"{name}.{key}": value for key, value in plan.copy().items()}
                    )
            if tied_keys := getattr(module, "all_tied_weights_keys", None):
                self.all_tied_weights_keys.update(
                    {
                        f"{name}.{target}": f"{name}.{source}"
                        for target, source in tied_keys.copy().items()
                    }
                )

    @property
    def basal_encoder(self) -> nn.Linear:
        return self.predictor.basal_encoder

    @property
    def pert_encoder(self) -> _ConditionLinearEmbedding:
        return self.action_encoder

    @property
    def transformer_backbone(self) -> StateTransformerBackbone:
        return self.predictor.transformer_backbone

    @property
    def project_out(self) -> nn.Linear:
        return self.predictor.project_out

    def get_input_embeddings(self) -> None:
        return None

    def set_input_embeddings(self, value: nn.Module) -> None:
        del value
        raise NotImplementedError("STATE accepts continuous cell embeddings, not token IDs")

    def predict(self, control_expression: Tensor, condition_ids: Tensor) -> Tensor:
        """Predict one perturbed population per control set and condition."""

        action = self.action_encoder(condition_ids)
        return self.predictor(self.encoder(control_expression), action)

    def forward(
        self,
        gene_ids: Tensor,
        gene_values: Tensor,
        cell_offsets: Tensor,
        cell_value_sums: Tensor | None = None,
        target_cell_map: Tensor | None = None,
        control_cell_map: Tensor | None = None,
        condition_ids: Tensor | None = None,
        return_dict: bool | None = None,
        **_: Any,
    ) -> StateModelOutput | tuple[Tensor, Tensor]:
        del cell_value_sums
        if target_cell_map is None or control_cell_map is None:
            raise ValueError("target_cell_map and control_cell_map are required")
        if condition_ids is None:
            raise ValueError("condition_ids is required")
        target, control = self.gene_tokenizer(
            {
                "gene_ids": gene_ids,
                "gene_values": gene_values,
                "cell_offsets": cell_offsets,
                "target_cell_map": target_cell_map,
                "control_cell_map": control_cell_map,
            }
        )
        target = self.encoder(target)
        control = self.encoder(control)
        if condition_ids.shape != (control.shape[0],):
            raise ValueError("condition_ids must have one entry per population")
        predicted = self.predict(control, condition_ids)
        per_set_loss = self.energy_loss(predicted, target)
        distribution_loss = torch.nanmean(per_set_loss)
        with torch.no_grad():
            diagnostics = {
                "prediction_mean": predicted.float().mean(),
                "prediction_rms": predicted.float().square().mean().sqrt(),
                "target_mean": target.float().mean(),
                "control_mean": control.float().mean(),
            }
        output = StateModelOutput(
            loss=distribution_loss,
            distribution_loss=distribution_loss,
            per_set_distribution_loss=per_set_loss,
            predicted_perturbed=predicted,
            encoded_target=target,
            encoded_control=control,
            diagnostics=diagnostics,
        )
        use_return_dict = self.config.use_return_dict if return_dict is None else return_dict
        return output if use_return_dict else (distribution_loss, predicted)


class StateTrainer(Parse10MTrainer):
    """Share Parse10M sampling and metrics with the reference STATE optimizer."""

    def create_optimizer(self):
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(
                [parameter for parameter in self.model.parameters() if parameter.requires_grad],
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
                weight_decay=0.0,
            )
        return self.optimizer


class StateVisitTrainer(VisitTrainerMixin, StateTrainer):
    pass


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECIPE = PROJECT_ROOT / "configs/parse10m_state_tiny_epoch_35400.yaml"
MODEL_FIELDS = {
    "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
    "num_key_value_heads", "head_dim", "rms_norm_eps", "attention_dropout",
    "hidden_dropout", "initializer_range", "energy_blur", "predict_residual", "output_relu",
}


def compile_experiment(recipe_path, *, protocol=None, fold=None, output_dir=None):
    """Bind a recipe to the same manifests and ordered vocabularies as JEPA."""
    recipe_path = Path(recipe_path).resolve()
    recipe = yaml.safe_load(recipe_path.read_text())
    unknown = set(recipe) - {"data", "model", "training", "swanlab", "retrieval"}
    if unknown:
        raise ValueError(f"Unknown STATE recipe sections: {sorted(unknown)}")
    data = recipe["data"]
    sampling = visit_sampling_config(data, recipe["training"])
    protocol = protocol or data["protocol"]
    fold = data.get("fold") if fold is None else fold
    root = (recipe_path.parent / Path(data["dataset_root"]).expanduser()).resolve()
    options = {
        "root": root, "protocol": protocol, "fold": fold,
        "cells_per_group": data["cells_per_group"], "seed": data["seed"],
    }
    train = Parse10MDataset(**options, split="train")
    options["store"] = train.store
    validation = Parse10MDataset(**options, split="val")
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["config"].get("target_sum") != 10000:
        raise ValueError("STATE expects the shared CP10K+log1p Parse10M expression space")
    labels = condition_labels(manifest)
    genes = np.load(root / "genes.npy", allow_pickle=False).tolist()
    if len(genes) != manifest["stats"]["retained_genes"] or len(set(genes)) != len(genes):
        raise ValueError("Parse10M gene vocabulary disagrees with the source manifest")
    if not (train.conditions | validation.conditions) <= set(labels):
        raise ValueError("Split conditions are absent from the source vocabulary")
    if not len(train) or not len(validation):
        raise ValueError("Training and validation splits must be nonempty")
    unknown = set(recipe["model"]) - MODEL_FIELDS
    if unknown:
        raise ValueError(f"Unknown STATE model fields: {sorted(unknown)}")
    training = dict(recipe["training"])
    base = PROJECT_ROOT / training.pop("output_dir")
    split_name = f"cell_type_fold_{fold:02d}" if protocol == "cell_type" else protocol
    destination = Path(output_dir).resolve() if output_dir else (base / split_name).resolve()
    run_name = training.pop("run_name", "parse10m_state") + "_" + split_name
    for field in ("report_to", "remove_unused_columns", "ignore_data_skip"):
        if field in training:
            raise ValueError(f"The STATE entry point manages {field}")
    if training.get("bf16", False) or training.get("fp16", False):
        raise ValueError("The reference STATE recipe uses FP32")
    if training.get("weight_decay", 0) != 0 or training.get("lr_scheduler_type", "constant") != "constant":
        raise ValueError("STATE uses Adam without weight decay and a constant learning rate")
    if training.get("warmup_steps", 0) or training.get("warmup_ratio", 0):
        raise ValueError("STATE uses zero warmup")
    retrieval = recipe.get("retrieval", {})
    if training.get("load_best_model_at_end", False):
        if not retrieval.get("enabled", False):
            raise ValueError("Validation MRR selection requires retrieval.enabled=true")
        if training.get("metric_for_best_model") != "eval_retrieval_mrr":
            raise ValueError("Select validation macro MRR with metric_for_best_model=eval_retrieval_mrr")
        if training.get("greater_is_better") is not True:
            raise ValueError("Validation MRR selection requires greater_is_better=true")
    if retrieval.get("enabled", False):
        if retrieval.get("cells_per_group", data["cells_per_group"]) != data["cells_per_group"]:
            raise ValueError("STATE retrieval cell sets must match the model cell_set_size")
        if retrieval.get("candidate_scope", "all_actions") not in {"all_actions", "target_split"}:
            raise ValueError("Invalid retrieval candidate_scope")
        if retrieval.get("candidate_batch_size", 2) < 1:
            raise ValueError("retrieval.candidate_batch_size must be positive")
    sources = [Path(__file__), Path(__file__).with_name("state_reference.json")]
    sources += [PROJECT_ROOT / "src" / p for p in (
        "data/parse10m.py", "perturbjepa/batch.py", "perturbjepa/trainer.py", "perturbjepa/retrieval.py",
        "perturbjepa/selection.py",
    )]
    if sampling is not None:
        sources += [PROJECT_ROOT / "src" / p for p in (
            "data/visit.py", "data/_visit_sampling.py", "perturbjepa/visit_trainer.py",
        )]
    contract = {
        "source_code_sha256": {str(p.relative_to(PROJECT_ROOT)): sha256(p) for p in sources},
        "source_manifest_sha256": sha256(root / "manifest.json"),
        "train_split_sha256": sha256(train.split_path),
        "validation_split_sha256": sha256(validation.split_path),
        "genes_sha256": sha256(root / "genes.npy"),
        "action_vocabulary_sha256": hashlib.sha256(
            json.dumps(labels, ensure_ascii=False, separators=(",", ":")).encode()
        ).hexdigest(),
        "protocol": protocol, "fold": fold,
        "data_seed": data["seed"], "cells_per_group": data["cells_per_group"],
        "model": recipe["model"], "training": training,
        **({"train_sampling": data["train_sampling"]} if sampling is not None else {}),
    }
    config = StateModelConfig(
        **recipe["model"], num_genes=len(genes), num_conditions=len(labels),
        cell_set_size=data["cells_per_group"], max_position_embeddings=data["cells_per_group"],
        condition_labels=labels, gene_names=genes, gene_indices=list(range(len(genes))),
        gene_vocabulary_sha256=contract["genes_sha256"],
        action_vocabulary_sha256=contract["action_vocabulary_sha256"], parse10m_contract=contract,
    )
    return recipe, options, train, validation, config, training, destination, run_name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_RECIPE)
    parser.add_argument("--protocol", choices=("drug_cell_type", "drug", "cell_type"))
    parser.add_argument("--fold", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume-from-checkpoint", metavar="PATH|latest")
    parser.add_argument("--swanlab-mode", choices=("cloud", "local", "offline"))
    parser.add_argument("--check", action="store_true")
    cli = parser.parse_args()
    recipe, options, train, validation, config, training, output_dir, run_name = compile_experiment(
        cli.config, protocol=cli.protocol, fold=cli.fold, output_dir=cli.output_dir,
    )
    sampling = visit_sampling_config(recipe["data"], training)
    # Meta construction counts parameters without allocating the full model or a GPU.
    with torch.device("meta"):
        parameter_count = sum(p.numel() for p in StateTransitionModel(config).parameters())
    if cli.check:
        print(json.dumps({
            "model": "STATE", "protocol": options["protocol"], "fold": options["fold"],
            "train_groups": len(train), "validation_groups": len(validation),
            "genes": config.num_genes, "conditions": config.num_conditions,
            "cells_per_group": config.cell_set_size, "parameters": parameter_count,
            "hidden_size": config.hidden_size, "layers": config.num_hidden_layers,
            "sampling_mode": "visit_decay" if sampling is not None else "epoch",
            **({"max_steps": training["max_steps"]}
               if training.get("max_steps", -1) > 0
               else {"epochs": training["num_train_epochs"]}),
            "precision": "fp32",
            "optimizer": "Adam", "output_dir": str(output_dir),
            "load_best_model_at_end": training.get("load_best_model_at_end", False),
            "metric_for_best_model": training.get("metric_for_best_model"),
            "swanlab": recipe["swanlab"],
        }, indent=2))
        return
    args = TrainingArguments(
        **{**training, **({"num_train_epochs": 0} if sampling is not None else {})},
        output_dir=str(output_dir), run_name=run_name,
        remove_unused_columns=False, ignore_data_skip=sampling is not None, report_to=[],
    )
    if sampling is None and (args.world_size != 1 or (args.device.type == "cuda" and torch.cuda.device_count() != 1)):
        raise ValueError("STATE baseline requires a single visible GPU; set CUDA_VISIBLE_DEVICES")
    checkpoint = None
    if cli.resume_from_checkpoint == "latest":
        find_checkpoint = last_visit_checkpoint if sampling is not None else get_last_checkpoint
        checkpoint = find_checkpoint(str(output_dir)) if output_dir.is_dir() else None
        if checkpoint is None:
            raise FileNotFoundError(f"No Trainer checkpoint under {output_dir}")
    elif cli.resume_from_checkpoint:
        checkpoint = str(Path(cli.resume_from_checkpoint).resolve())
    elif output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"{output_dir} is nonempty; use --resume-from-checkpoint or --output-dir")
    if checkpoint:
        saved = json.loads((Path(checkpoint) / "config.json").read_text())
        if saved.get("parse10m_contract") != config.parse10m_contract:
            raise ValueError("Checkpoint and requested STATE data/model/training contracts differ")
        if saved.get("parse10m_world_size") != args.world_size:
            raise ValueError("Checkpoint and requested world sizes differ")
        state = json.loads((Path(checkpoint) / "trainer_state.json").read_text())
        total_steps = args.max_steps if args.max_steps > 0 else math.ceil(
            math.ceil(len(train) / args.train_batch_size) / args.gradient_accumulation_steps
        ) * args.num_train_epochs
        if sampling is None and state["global_step"] >= math.ceil(total_steps):
            raise ValueError("This training run is already complete")
    config.parse10m_world_size = args.world_size
    callbacks = []
    swan = recipe["swanlab"]
    if swan.get("enabled", True):
        from swanlab.integration.transformers import SwanLabCallback

        callbacks.append(SwanLabCallback(
            project=swan["project"], workspace=swan.get("workspace"), experiment_name=run_name,
            logdir=str(output_dir / "swanlab"), mode=cli.swanlab_mode or swan["mode"],
        ))
    set_seed(args.seed)
    model = StateTransitionModel(config)
    trainer_type = StateVisitTrainer if sampling is not None else StateTrainer
    trainer = trainer_type(
        model=model, args=args, train_dataset=train, eval_dataset=validation,
        data_collator=SparseBatchCollator({name: i for i, name in enumerate(config.condition_labels)}),
        callbacks=callbacks,
        **({"visit_sampling": sampling} if sampling is not None else {}),
        retrieval=({**recipe["retrieval"], "dataset_options": options}
                   if recipe.get("retrieval", {}).get("enabled", False) else None),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "experiment.json", {
        "model": "STATE", "parameters": parameter_count, "trainable_parameters": parameter_count,
        "precision": "fp32", "optimizer": "Adam", "recipe": recipe,
        "contract": config.parse10m_contract, "run_name": run_name,
    })
    print(f"STATE: {parameter_count:,} trainable parameters, single device, FP32", flush=True)
    try:
        result = trainer.train(resume_from_checkpoint=checkpoint)
        finish_training(trainer, result, output_dir)
    finally:
        train.close()
    if swan.get("enabled", True):
        import swanlab

        swanlab.finish()


if __name__ == "__main__":
    main()
