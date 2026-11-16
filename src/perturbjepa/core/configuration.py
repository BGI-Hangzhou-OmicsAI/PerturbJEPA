"""Serializable configuration for the perturbation world model."""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Sequence

from transformers import PretrainedConfig

VALID_DISTRIBUTION_LOSSES = (
    "energy",
    "gaussian",
    "laplacian",
    "sinkhorn",
)
VALID_GEOMLOSS_BACKENDS = ("auto", "multiscale", "online", "tensorized")
VALID_SIGREG_SCOPES = ("both", "control", "target")
VALID_SIGREG_SAMPLE_MODES = ("batch", "batch_cells", "cells", "batch_per_cell")
VALID_GENE_POOLING_NORMALIZATIONS = ("mean", "l2", "none")
VALID_PREDICTOR_TYPES = ("transformer", "mlp", "token_transformer", "token_adaln")

# These are deliberately narrower than ``PerturbWorldModelConfig.__init__``.
# Action vocabulary fields are derived from the selected dataset, while generic
# Hugging Face config kwargs are only accepted when restoring a checkpoint.
MODEL_EXPERIMENT_FIELDS = frozenset(
    {
        "action_embedding_size",
        "action_encoder_type",
        "attention_dropout",
        "encoder_attention_heads",
        "encoder_intermediate_size",
        "encoder_layers",
        "hidden_dropout",
        "hidden_size",
        "initializer_range",
        "gene_pooling_normalization",
        "encoder_layer_scale_init",
        "normalize_action_embedding",
        "adaln_weight_init_std",
        "adaln_gate_init",
        "action_input_scale",
        "num_genes",
        "predictor_layer_scale_init",
        "predictor_type",
        "predictor_attention_heads",
        "predictor_intermediate_size",
        "predictor_layers",
        "rms_norm_eps",
        "bulk_effect_intermediate_size",
        "bulk_effect_input_scale",
    }
)
LOSS_EXPERIMENT_FIELDS = frozenset(
    {
        "distribution_loss",
        "distribution_loss_weight",
        "distribution_normalize_by_sqrt_dim",
        "geomloss_backend",
        "geomloss_blur",
        "geomloss_debias",
        "geomloss_p",
        "geomloss_reach",
        "geomloss_scaling",
        "sigreg_n_points",
        "sigreg_normalize_by_num_samples",
        "sigreg_num_slices",
        "sigreg_sample_mode",
        "sigreg_scope",
        "sigreg_t_max",
        "sigreg_weight",
        "bulk_effect_weight",
        "bulk_effect_sigreg_weight",
        "bulk_effect_scale",
        "action_rank_weight",
        "action_rank_num_negatives",
        "action_rank_temperature",
        "action_rank_cells",
        "dose_rank_weight",
    }
)


def validate_experiment_sections(model: Mapping[str, Any], loss: Mapping[str, Any]) -> None:
    """Reject misspelled or misplaced YAML fields before model construction."""

    unknown_model = sorted(set(model) - MODEL_EXPERIMENT_FIELDS)
    if unknown_model:
        raise ValueError("Unknown model configuration fields: {}".format(", ".join(unknown_model)))
    unknown_loss = sorted(set(loss) - LOSS_EXPERIMENT_FIELDS)
    if unknown_loss:
        raise ValueError("Unknown loss configuration fields: {}".format(", ".join(unknown_loss)))


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("{} must be a positive integer".format(name))
    return int(value)


def _nonnegative_float(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("{} must be a nonnegative number".format(name))
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError("{} must be a finite nonnegative number".format(name))
    return result


def _positive_float(name: str, value: Any) -> float:
    result = _nonnegative_float(name, value)
    if result == 0:
        raise ValueError("{} must be positive".format(name))
    return result


def _dropout(name: str, value: Any) -> float:
    result = _nonnegative_float(name, value)
    if result >= 1:
        raise ValueError("{} must be in [0, 1)".format(name))
    return result


def _mapping(
    name: str,
    value: Optional[Sequence[int]],
    *,
    length: int,
    upper_bound: int,
) -> list[int]:
    if value is None:
        if upper_bound == 1:
            return [0] * length
        if upper_bound == length:
            return list(range(length))
        raise ValueError(
            "{} is required when its vocabulary size differs from num_conditions".format(name)
        )
    if isinstance(value, (str, bytes)) or len(value) != length:
        raise ValueError("{} must contain one entry per condition".format(name))
    result = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError("{} must contain integers".format(name))
        item = int(item)
        if item < 0 or item >= upper_bound:
            raise ValueError("{} contains an out-of-range index".format(name))
        result.append(item)
    return result


def _labels(name: str, value: Optional[Sequence[Any]], *, length: int) -> Optional[list[str]]:
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or len(value) != length:
        raise ValueError("{} must contain {} entries".format(name, length))
    result = [str(item) for item in value]
    if len(set(result)) != len(result):
        raise ValueError("{} must not contain duplicates".format(name))
    return result


def _gene_vocabulary(
    *,
    num_genes: int,
    gene_indices: Optional[Sequence[int]],
    gene_names: Optional[Sequence[Any]],
    gene_vocabulary_sha256: Optional[str],
) -> tuple[Optional[list[int]], Optional[list[str]], Optional[str]]:
    provided = (
        gene_indices is not None,
        gene_names is not None,
        gene_vocabulary_sha256 is not None,
    )
    if not any(provided):
        return None, None, None
    if not all(provided):
        raise ValueError(
            "gene_indices, gene_names, and gene_vocabulary_sha256 must be configured together"
        )
    if isinstance(gene_indices, (str, bytes)) or len(gene_indices) != num_genes:
        raise ValueError("gene_indices must contain one entry per model gene")
    indices = []
    for value in gene_indices:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("gene_indices must contain nonnegative integers")
        indices.append(int(value))
    if len(set(indices)) != len(indices):
        raise ValueError("gene_indices must not contain duplicates")

    names = _labels("gene_names", gene_names, length=num_genes)
    if names is None or any(not name for name in names):
        raise ValueError("gene_names must contain non-empty strings")
    vocabulary_hash = str(gene_vocabulary_sha256).strip().lower()
    if not vocabulary_hash:
        raise ValueError("gene_vocabulary_sha256 must be non-empty")
    return indices, names, vocabulary_hash


class PerturbWorldModelConfig(PretrainedConfig):
    """All architecture and objective choices needed to restore a checkpoint."""

    model_type = "perturb_world_model"

    def __init__(
        self,
        num_genes: int = 62_710,
        gene_indices: Optional[Sequence[int]] = None,
        gene_names: Optional[Sequence[Any]] = None,
        gene_vocabulary_sha256: Optional[str] = None,
        hidden_size: int = 512,
        encoder_layers: int = 8,
        encoder_attention_heads: int = 8,
        encoder_intermediate_size: int = 2_048,
        predictor_type: str = "transformer",
        predictor_layers: int = 4,
        predictor_attention_heads: int = 8,
        predictor_intermediate_size: int = 2_048,
        predictor_layer_scale_init: float = 0.1,
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        rms_norm_eps: float = 1e-6,
        initializer_range: float = 0.02,
        gene_pooling_normalization: str = "mean",
        encoder_layer_scale_init: float = 1.0,
        num_conditions: int = 1,
        num_drugs: int = 1,
        num_doses: int = 1,
        condition_to_drug_id: Optional[Sequence[int]] = None,
        condition_to_dose_id: Optional[Sequence[int]] = None,
        condition_labels: Optional[Sequence[str]] = None,
        drug_labels: Optional[Sequence[str]] = None,
        dose_values: Optional[Sequence[float]] = None,
        action_vocabulary_sha256: Optional[str] = None,
        action_embedding_size: Optional[int] = None,
        action_encoder_type: str = "drug_dose",
        normalize_action_embedding: bool = False,
        adaln_weight_init_std: float = 0.0,
        adaln_gate_init: float = 0.0,
        action_input_scale: float = 0.0,
        distribution_loss: str = "energy",
        distribution_loss_weight: float = 1.0,
        distribution_normalize_by_sqrt_dim: bool = True,
        geomloss_p: int = 2,
        geomloss_blur: float = 0.05,
        geomloss_reach: Optional[float] = None,
        geomloss_scaling: float = 0.9,
        geomloss_debias: bool = True,
        geomloss_backend: str = "auto",
        sigreg_weight: float = 1.0,
        sigreg_num_slices: int = 256,
        sigreg_n_points: int = 17,
        sigreg_t_max: float = 3.0,
        sigreg_scope: str = "both",
        sigreg_sample_mode: str = "batch",
        sigreg_normalize_by_num_samples: bool = False,
        bulk_effect_intermediate_size: int = 128,
        bulk_effect_input_scale: float = 1.0,
        bulk_effect_weight: float = 0.0,
        bulk_effect_sigreg_weight: float = 0.0,
        bulk_effect_scale: float = 0.2,
        action_rank_weight: float = 0.0,
        action_rank_num_negatives: int = 8,
        action_rank_temperature: float = 0.01,
        action_rank_cells: int = 32,
        dose_rank_weight: float = 0.0,
        **kwargs: Any,
    ) -> None:
        unsupported = sorted(
            key for key in kwargs if key.startswith(("precomputed_", "freeze_precomputed_"))
        )
        if unsupported:
            raise ValueError(f"Unsupported configuration fields: {unsupported}")
        kwargs.setdefault("tie_word_embeddings", False)
        super().__init__(**kwargs)

        self.num_genes = _positive_int("num_genes", num_genes)
        (
            self.gene_indices,
            self.gene_names,
            self.gene_vocabulary_sha256,
        ) = _gene_vocabulary(
            num_genes=self.num_genes,
            gene_indices=gene_indices,
            gene_names=gene_names,
            gene_vocabulary_sha256=gene_vocabulary_sha256,
        )
        self.hidden_size = _positive_int("hidden_size", hidden_size)
        self.encoder_layers = _positive_int("encoder_layers", encoder_layers)
        self.encoder_attention_heads = _positive_int(
            "encoder_attention_heads", encoder_attention_heads
        )
        self.encoder_intermediate_size = _positive_int(
            "encoder_intermediate_size", encoder_intermediate_size
        )
        self.predictor_type = str(predictor_type).lower()
        if self.predictor_type not in VALID_PREDICTOR_TYPES:
            raise ValueError("predictor_type must be one of {}".format(VALID_PREDICTOR_TYPES))
        self.predictor_layers = _positive_int("predictor_layers", predictor_layers)
        self.predictor_attention_heads = _positive_int(
            "predictor_attention_heads", predictor_attention_heads
        )
        self.predictor_intermediate_size = _positive_int(
            "predictor_intermediate_size", predictor_intermediate_size
        )
        self.predictor_layer_scale_init = _nonnegative_float(
            "predictor_layer_scale_init", predictor_layer_scale_init
        )
        if self.hidden_size % self.encoder_attention_heads:
            raise ValueError("hidden_size must be divisible by encoder_attention_heads")
        if (
            self.predictor_type in {"transformer", "token_transformer", "token_adaln"}
            and self.hidden_size % self.predictor_attention_heads
        ):
            raise ValueError("hidden_size must be divisible by predictor_attention_heads")

        self.attention_dropout = _dropout("attention_dropout", attention_dropout)
        self.hidden_dropout = _dropout("hidden_dropout", hidden_dropout)
        self.rms_norm_eps = _positive_float("rms_norm_eps", rms_norm_eps)
        self.initializer_range = _positive_float("initializer_range", initializer_range)
        self.gene_pooling_normalization = str(gene_pooling_normalization).lower()
        if self.gene_pooling_normalization not in VALID_GENE_POOLING_NORMALIZATIONS:
            raise ValueError(
                "gene_pooling_normalization must be one of {}".format(
                    VALID_GENE_POOLING_NORMALIZATIONS
                )
            )
        if not isinstance(normalize_action_embedding, bool):
            raise ValueError("normalize_action_embedding must be boolean")
        self.normalize_action_embedding = normalize_action_embedding
        self.encoder_layer_scale_init = _nonnegative_float(
            "encoder_layer_scale_init", encoder_layer_scale_init
        )
        self.adaln_weight_init_std = _nonnegative_float(
            "adaln_weight_init_std", adaln_weight_init_std
        )
        self.adaln_gate_init = _nonnegative_float("adaln_gate_init", adaln_gate_init)
        self.action_input_scale = _nonnegative_float("action_input_scale", action_input_scale)

        self.num_conditions = _positive_int("num_conditions", num_conditions)
        self.action_encoder_type = str(action_encoder_type).lower()
        if self.action_encoder_type not in {"drug_dose", "condition_id"}:
            raise ValueError("action_encoder_type must be drug_dose or condition_id")
        self.condition_labels = _labels(
            "condition_labels", condition_labels, length=self.num_conditions
        )
        if self.action_encoder_type == "condition_id":
            if self.predictor_type not in {"token_transformer", "token_adaln"}:
                raise ValueError("Condition IDs require a token predictor")
            if (
                dose_values is not None
                or condition_to_dose_id is not None
                or condition_to_drug_id is not None
                or drug_labels is not None
                or num_drugs != 1
                or num_doses != 1
                or dose_rank_weight != 0
            ):
                raise ValueError("Condition-ID encoding has no dose settings")
        else:
            self.num_drugs = _positive_int("num_drugs", num_drugs)
            self.num_doses = _positive_int("num_doses", num_doses)
            self.condition_to_drug_id = _mapping(
                "condition_to_drug_id",
                condition_to_drug_id,
                length=self.num_conditions,
                upper_bound=self.num_drugs,
            )
            self.condition_to_dose_id = _mapping(
                "condition_to_dose_id",
                condition_to_dose_id,
                length=self.num_conditions,
                upper_bound=self.num_doses,
            )
            self.drug_labels = _labels("drug_labels", drug_labels, length=self.num_drugs)
            if dose_values is None:
                self.dose_values = None
            else:
                if isinstance(dose_values, (str, bytes)) or len(dose_values) != self.num_doses:
                    raise ValueError("dose_values must contain one value per dose")
                self.dose_values = [float(item) for item in dose_values]
                if not all(math.isfinite(item) for item in self.dose_values):
                    raise ValueError("dose_values must be finite")
            if self.predictor_type in {"token_transformer", "token_adaln"}:
                if self.dose_values is None:
                    if self.num_doses == 1:
                        self.dose_values = [0.0]
                    else:
                        raise ValueError(
                            "dose_values are required for {}".format(self.predictor_type)
                        )
                if any(item < 0 for item in self.dose_values):
                    raise ValueError(
                        "dose_values must be nonnegative for {}".format(self.predictor_type)
                    )
        self.action_vocabulary_sha256 = (
            None if action_vocabulary_sha256 is None else str(action_vocabulary_sha256)
        )
        self.action_embedding_size = (
            self.hidden_size
            if action_embedding_size is None
            else _positive_int("action_embedding_size", action_embedding_size)
        )
        self.distribution_loss = str(distribution_loss).lower()
        if self.distribution_loss not in VALID_DISTRIBUTION_LOSSES:
            raise ValueError(
                "distribution_loss must be one of {}".format(VALID_DISTRIBUTION_LOSSES)
            )
        self.distribution_loss_weight = _nonnegative_float(
            "distribution_loss_weight", distribution_loss_weight
        )
        if not isinstance(distribution_normalize_by_sqrt_dim, bool):
            raise ValueError("distribution_normalize_by_sqrt_dim must be boolean")
        self.distribution_normalize_by_sqrt_dim = distribution_normalize_by_sqrt_dim
        self.geomloss_p = _positive_int("geomloss_p", geomloss_p)
        if self.geomloss_p not in (1, 2):
            raise ValueError("geomloss_p must be 1 or 2")
        self.geomloss_blur = _positive_float("geomloss_blur", geomloss_blur)
        self.geomloss_reach = (
            None if geomloss_reach is None else _positive_float("geomloss_reach", geomloss_reach)
        )
        self.geomloss_scaling = _positive_float("geomloss_scaling", geomloss_scaling)
        if self.geomloss_scaling > 1:
            raise ValueError("geomloss_scaling must be in (0, 1]")
        if not isinstance(geomloss_debias, bool):
            raise ValueError("geomloss_debias must be boolean")
        self.geomloss_debias = geomloss_debias
        self.geomloss_backend = str(geomloss_backend).lower()
        if self.geomloss_backend not in VALID_GEOMLOSS_BACKENDS:
            raise ValueError("geomloss_backend must be one of {}".format(VALID_GEOMLOSS_BACKENDS))

        self.sigreg_weight = _nonnegative_float("sigreg_weight", sigreg_weight)
        self.sigreg_num_slices = _positive_int("sigreg_num_slices", sigreg_num_slices)
        self.sigreg_n_points = _positive_int("sigreg_n_points", sigreg_n_points)
        if self.sigreg_n_points < 3 or self.sigreg_n_points % 2 != 1:
            raise ValueError("sigreg_n_points must be an odd integer of at least 3")
        self.sigreg_t_max = _positive_float("sigreg_t_max", sigreg_t_max)
        self.sigreg_scope = str(sigreg_scope).lower()
        if self.sigreg_scope not in VALID_SIGREG_SCOPES:
            raise ValueError("sigreg_scope must be one of {}".format(VALID_SIGREG_SCOPES))
        self.sigreg_sample_mode = str(sigreg_sample_mode).lower()
        if self.sigreg_sample_mode not in VALID_SIGREG_SAMPLE_MODES:
            raise ValueError(
                "sigreg_sample_mode must be one of {}".format(VALID_SIGREG_SAMPLE_MODES)
            )
        if not isinstance(sigreg_normalize_by_num_samples, bool):
            raise ValueError("sigreg_normalize_by_num_samples must be boolean")
        self.sigreg_normalize_by_num_samples = sigreg_normalize_by_num_samples
        self.bulk_effect_intermediate_size = _positive_int(
            "bulk_effect_intermediate_size", bulk_effect_intermediate_size
        )
        self.bulk_effect_input_scale = _positive_float(
            "bulk_effect_input_scale", bulk_effect_input_scale
        )
        self.bulk_effect_weight = _nonnegative_float("bulk_effect_weight", bulk_effect_weight)
        self.bulk_effect_sigreg_weight = _nonnegative_float(
            "bulk_effect_sigreg_weight", bulk_effect_sigreg_weight
        )
        self.bulk_effect_scale = _positive_float("bulk_effect_scale", bulk_effect_scale)
        self.action_rank_weight = _nonnegative_float(
            "action_rank_weight", action_rank_weight
        )
        self.action_rank_num_negatives = _positive_int(
            "action_rank_num_negatives", action_rank_num_negatives
        )
        self.action_rank_temperature = _positive_float(
            "action_rank_temperature", action_rank_temperature
        )
        self.action_rank_cells = _positive_int("action_rank_cells", action_rank_cells)
        if self.action_encoder_type == "drug_dose":
            self.dose_rank_weight = _nonnegative_float("dose_rank_weight", dose_rank_weight)


__all__ = [
    "LOSS_EXPERIMENT_FIELDS",
    "MODEL_EXPERIMENT_FIELDS",
    "VALID_DISTRIBUTION_LOSSES",
    "VALID_GENE_POOLING_NORMALIZATIONS",
    "VALID_GEOMLOSS_BACKENDS",
    "VALID_PREDICTOR_TYPES",
    "VALID_SIGREG_SAMPLE_MODES",
    "VALID_SIGREG_SCOPES",
    "PerturbWorldModelConfig",
    "validate_experiment_sections",
]
