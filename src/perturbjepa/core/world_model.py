"""Hugging Face model for action-conditioned virtual-cell prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from .action import ConditionIDTokenEncoder, DrugDoseActionEncoder, DrugDoseTokenEncoder
from .bulk_effect import RawBulkEffectEncoder, build_bulk_difference
from .configuration import PerturbWorldModelConfig
from .gene_bag import ExpressionWeightedGeneBag
from .losses import EmbeddingSIGRegLoss, PopulationDistributionLoss, SIGRegLoss
from .predictor import ActionConditionedMLPPredictor
from .set_transformer import (
    ActionConditionedPredictor,
    ActionTokenAdaLNZeroPredictor,
    ActionTokenSetTransformerPredictor,
    RMSNorm,
    SetTransformer,
)


@dataclass
class PerturbWorldModelOutput(ModelOutput):
    """Losses, optional latent populations, and detached training diagnostics."""

    loss: Optional[Tensor] = None
    distribution_loss: Optional[Tensor] = None
    sigreg_loss: Optional[Tensor] = None
    predicted_perturbed: Optional[Tensor] = None
    encoded_target: Optional[Tensor] = None
    encoded_control: Optional[Tensor] = None
    diagnostics: Optional[Dict[str, Tensor]] = None
    sigreg_raw_loss: Optional[Tensor] = None
    sigreg_normalized_loss: Optional[Tensor] = None
    bulk_effect_loss: Optional[Tensor] = None
    bulk_effect_sigreg_loss: Optional[Tensor] = None
    bulk_effect_sigreg_raw_loss: Optional[Tensor] = None
    bulk_effect_sigreg_normalized_loss: Optional[Tensor] = None
    bulk_effect_embedding: Optional[Tensor] = None
    action_rank_loss: Optional[Tensor] = None
    dose_rank_loss: Optional[Tensor] = None


@dataclass(frozen=True)
class _ActionRankingResult:
    action_loss: Optional[Tensor]
    dose_loss: Optional[Tensor]
    metrics: Dict[str, Tensor]


class PerturbWorldModel(PreTrainedModel):
    """Encode control/target cell sets and predict the drug-perturbed distribution."""

    config_class = PerturbWorldModelConfig
    base_model_prefix = "perturb_world_model"
    main_input_name = "gene_ids"

    def __init__(self, config: PerturbWorldModelConfig) -> None:
        super().__init__(config)
        self.gene_tokenizer = ExpressionWeightedGeneBag(
            embedding_dim=config.hidden_size,
            num_embeddings=config.num_genes,
            pooling_normalization=config.gene_pooling_normalization,
        )
        self.encoder = SetTransformer(
            hidden_size=config.hidden_size,
            depth=config.encoder_layers,
            num_heads=config.encoder_attention_heads,
            intermediate_size=config.encoder_intermediate_size,
            attention_dropout=config.attention_dropout,
            hidden_dropout=config.hidden_dropout,
            rms_norm_eps=config.rms_norm_eps,
            layer_scale_init=config.encoder_layer_scale_init,
        )
        if config.action_encoder_type == "condition_id":
            self.action_encoder = ConditionIDTokenEncoder(
                num_conditions=config.num_conditions, hidden_size=config.hidden_size,
                embedding_size=config.action_embedding_size,
                normalize_output=config.normalize_action_embedding,
                rms_norm_eps=config.rms_norm_eps,
            )
        else:
            action_encoder_arguments = {
                "num_conditions": config.num_conditions,
                "num_drugs": config.num_drugs,
                "num_doses": config.num_doses,
                "condition_to_drug_id": config.condition_to_drug_id,
                "condition_to_dose_id": config.condition_to_dose_id,
                "hidden_size": config.hidden_size,
                "embedding_size": config.action_embedding_size,
                "normalize_output": config.normalize_action_embedding,
                "rms_norm_eps": config.rms_norm_eps,
            }
            if config.predictor_type in {"token_transformer", "token_adaln"}:
                self.action_encoder = DrugDoseTokenEncoder(
                    **action_encoder_arguments,
                    dose_values=config.dose_values,
                )
            else:
                self.action_encoder = DrugDoseActionEncoder(**action_encoder_arguments)

        if config.predictor_type == "mlp":
            self.predictor = ActionConditionedMLPPredictor(
                hidden_size=config.hidden_size,
                depth=config.predictor_layers,
                intermediate_size=config.predictor_intermediate_size,
                hidden_dropout=config.hidden_dropout,
                rms_norm_eps=config.rms_norm_eps,
                layer_scale_init=config.predictor_layer_scale_init,
            )
        elif config.predictor_type == "transformer":
            self.predictor = ActionConditionedPredictor(
                hidden_size=config.hidden_size,
                depth=config.predictor_layers,
                num_heads=config.predictor_attention_heads,
                intermediate_size=config.predictor_intermediate_size,
                attention_dropout=config.attention_dropout,
                hidden_dropout=config.hidden_dropout,
                rms_norm_eps=config.rms_norm_eps,
                modulation_weight_std=config.adaln_weight_init_std,
                gate_init=config.adaln_gate_init,
                action_input_scale=config.action_input_scale,
            )
        elif config.predictor_type == "token_transformer":
            # Predictor constructors consume different amounts of randomness. Forking here
            # lets Hugging Face post_init start from the same state for paired token ablations.
            with torch.random.fork_rng(devices=[]):
                self.predictor = ActionTokenSetTransformerPredictor(
                    hidden_size=config.hidden_size,
                    depth=config.predictor_layers,
                    num_heads=config.predictor_attention_heads,
                    intermediate_size=config.predictor_intermediate_size,
                    num_action_tokens=self.action_encoder.num_action_tokens,
                    attention_dropout=config.attention_dropout,
                    hidden_dropout=config.hidden_dropout,
                    rms_norm_eps=config.rms_norm_eps,
                    layer_scale_init=config.predictor_layer_scale_init,
                )
        elif config.predictor_type == "token_adaln":
            with torch.random.fork_rng(devices=[]):
                self.predictor = ActionTokenAdaLNZeroPredictor(
                    hidden_size=config.hidden_size,
                    depth=config.predictor_layers,
                    num_heads=config.predictor_attention_heads,
                    intermediate_size=config.predictor_intermediate_size,
                    num_action_tokens=self.action_encoder.num_action_tokens,
                    attention_dropout=config.attention_dropout,
                    hidden_dropout=config.hidden_dropout,
                    rms_norm_eps=config.rms_norm_eps,
                    modulation_weight_std=config.adaln_weight_init_std,
                    gate_init=config.adaln_gate_init,
                    action_input_scale=config.action_input_scale,
                )
        else:  # PerturbWorldModelConfig rejects this before model construction.
            raise AssertionError("unsupported predictor_type")
        self.distribution_criterion: Optional[PopulationDistributionLoss]
        if (
            config.distribution_loss_weight > 0
            or config.action_rank_weight > 0
            or getattr(config, "dose_rank_weight", 0.0) > 0
        ):
            self.distribution_criterion = PopulationDistributionLoss(
                config.distribution_loss,
                p=config.geomloss_p,
                blur=config.geomloss_blur,
                reach=config.geomloss_reach,
                scaling=config.geomloss_scaling,
                debias=config.geomloss_debias,
                backend=config.geomloss_backend,
                normalize_by_sqrt_dim=(config.distribution_normalize_by_sqrt_dim),
            )
        else:
            self.distribution_criterion = None
        self.sigreg_regularizer: Optional[SIGRegLoss]
        if config.sigreg_weight > 0:
            self.sigreg_regularizer = SIGRegLoss(
                num_slices=config.sigreg_num_slices,
                n_points=config.sigreg_n_points,
                t_max=config.sigreg_t_max,
                scope=config.sigreg_scope,
                sample_mode=config.sigreg_sample_mode,
                normalize_by_num_samples=config.sigreg_normalize_by_num_samples,
            )
        else:
            self.sigreg_regularizer = None

        self.bulk_effect_encoder: Optional[RawBulkEffectEncoder] = None
        self.bulk_effect_regularizer: Optional[EmbeddingSIGRegLoss] = None
        if config.bulk_effect_weight > 0 or config.bulk_effect_sigreg_weight > 0:
            self.bulk_effect_encoder = RawBulkEffectEncoder(
                num_genes=config.num_genes,
                hidden_size=config.hidden_size,
                intermediate_size=config.bulk_effect_intermediate_size,
                input_scale=config.bulk_effect_input_scale,
            )
        if config.bulk_effect_sigreg_weight > 0:
            self.bulk_effect_regularizer = EmbeddingSIGRegLoss(
                num_slices=config.sigreg_num_slices,
                n_points=config.sigreg_n_points,
                t_max=config.sigreg_t_max,
                normalize_by_num_samples=True,
            )

        self.post_init()
        if isinstance(self.predictor, ActionConditionedPredictor):
            # Hugging Face initialization touches Linear layers, so restore the
            # configured AdaLN initialization.
            self.predictor.reset_adaln_parameters()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, ExpressionWeightedGeneBag):
            # This nonpersistent buffer is absent from checkpoints. HF can
            # materialize it with empty_like when loading a model on meta.
            nn.init.ones_(module._normalization_embedding)
        elif isinstance(module, nn.Linear):
            std = getattr(module, "_bulk_effect_init_std", self.config.initializer_range)
            nn.init.normal_(module.weight, mean=0.0, std=float(std))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, RMSNorm) and module.weight is not None:
            nn.init.ones_(module.weight)
        elif isinstance(module, (nn.Embedding, nn.EmbeddingBag)):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=float(self.config.initializer_range),
            )
            padding_idx = getattr(module, "padding_idx", None)
            if padding_idx is not None:
                with torch.no_grad():
                    module.weight[int(padding_idx)].zero_()

    def get_input_embeddings(self) -> nn.Module:
        return self.gene_tokenizer.embedding_bag

    def set_input_embeddings(self, value: nn.Module) -> None:
        if not isinstance(value, nn.EmbeddingBag):
            raise TypeError("input embeddings must be torch.nn.EmbeddingBag")
        self.gene_tokenizer.embedding_bag = value

    @staticmethod
    def _pool_action_embedding(action_embedding: Tensor) -> Tensor:
        if action_embedding.ndim == 2:
            return action_embedding
        if action_embedding.ndim == 3:
            return action_embedding.mean(dim=1)
        raise ValueError("action embedding must have shape [batch, dim] or [batch, tokens, dim]")

    def _hard_action_negatives(self, condition_ids, positive_action_embedding):
        return self._hard_same_dose_negatives(condition_ids, positive_action_embedding)

    def _hard_same_dose_negatives(
        self,
        condition_ids: Tensor,
        positive_action_embedding: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Select current-embedding nearest actions at the same dose but a different drug."""

        negative_count = int(self.config.action_rank_num_negatives)
        all_condition_ids = torch.arange(
            int(self.config.num_conditions), device=condition_ids.device, dtype=torch.long
        )
        with torch.no_grad():
            all_action_embedding = self.action_encoder(all_condition_ids)
            queries = F.normalize(
                self._pool_action_embedding(positive_action_embedding.detach()).float(), dim=-1
            )
            candidates = F.normalize(
                self._pool_action_embedding(all_action_embedding).float(), dim=-1
            )
            similarities = queries @ candidates.transpose(0, 1)

            condition_to_dose = self.action_encoder.condition_to_dose_id
            condition_to_drug = self.action_encoder.condition_to_drug_id
            positive_doses = condition_to_dose[condition_ids]
            positive_drugs = condition_to_drug[condition_ids]
            valid = condition_to_dose.unsqueeze(0) == positive_doses.unsqueeze(1)
            valid = valid & (
                condition_to_drug.unsqueeze(0) != positive_drugs.unsqueeze(1)
            )
            valid_counts = valid.sum(dim=1)
            if bool((valid_counts < negative_count).any()):
                minimum = int(valid_counts.min().item())
                raise ValueError(
                    "action ranking needs {} same-dose different-drug negatives per "
                    "condition, but this batch has as few as {}".format(
                        negative_count, minimum
                    )
                )
            selected_similarity, selected_ids = similarities.masked_fill(
                ~valid, -torch.inf
            ).topk(negative_count, dim=1)
        return selected_ids, selected_similarity

    def _same_drug_dose_negatives(self, condition_ids: Tensor) -> Tensor:
        """Return the two other dose conditions for each treated drug action."""

        with torch.no_grad():
            condition_to_dose = self.action_encoder.condition_to_dose_id
            condition_to_drug = self.action_encoder.condition_to_drug_id
            positive_doses = condition_to_dose[condition_ids]
            positive_drugs = condition_to_drug[condition_ids]
            valid = condition_to_drug.unsqueeze(0) == positive_drugs.unsqueeze(1)
            valid = valid & (condition_to_dose.unsqueeze(0) != positive_doses.unsqueeze(1))
            valid_counts = valid.sum(dim=1)
            if bool((valid_counts != 2).any()):
                minimum = int(valid_counts.min().item())
                maximum = int(valid_counts.max().item())
                raise ValueError(
                    "dose ranking requires exactly 2 same-drug different-dose negatives "
                    "per condition, but this batch has between {} and {}".format(
                        minimum, maximum
                    )
                )

            all_condition_ids = torch.arange(
                int(self.config.num_conditions), device=condition_ids.device, dtype=torch.long
            )
            candidates = all_condition_ids.unsqueeze(0).expand(condition_ids.shape[0], -1)
            selected = candidates.masked_fill(~valid, int(self.config.num_conditions)).topk(
                2, dim=1, largest=False, sorted=True
            ).values
            selected_doses = condition_to_dose[selected]
            if bool((selected_doses[:, 0] == selected_doses[:, 1]).any()):
                raise ValueError(
                    "dose ranking requires the 2 same-drug negatives to have distinct dose IDs"
                )
        return selected

    def _candidate_action_distances(
        self,
        encoded_control: Tensor,
        encoded_target: Tensor,
        candidate_ids: Tensor,
    ) -> Tensor:
        if self.distribution_criterion is None:
            raise RuntimeError("action ranking requires a population distribution criterion")
        batch_size, candidate_count = candidate_ids.shape
        rank_cells = min(int(self.config.action_rank_cells), int(encoded_control.shape[1]))
        controls = encoded_control[:, :rank_cells]
        targets = encoded_target[:, :rank_cells]
        feature_size = int(controls.shape[-1])

        candidate_actions = self.action_encoder(candidate_ids.reshape(-1))
        repeated_controls = controls.unsqueeze(1).expand(
            -1, candidate_count, -1, -1
        ).reshape(batch_size * candidate_count, rank_cells, feature_size)
        candidate_predictions = self.predictor(repeated_controls, candidate_actions)
        repeated_targets = targets.unsqueeze(1).expand(
            -1, candidate_count, -1, -1
        ).reshape(batch_size * candidate_count, rank_cells, feature_size)
        return self.distribution_criterion.per_population(
            candidate_predictions, repeated_targets
        ).reshape(batch_size, candidate_count)

    def _ranking_objective(
        self,
        distances: Tensor,
        *,
        prefix: str,
        selected_similarity: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        batch_size = int(distances.shape[0])
        logits = -distances / float(self.config.action_rank_temperature)
        labels = torch.zeros(batch_size, device=distances.device, dtype=torch.long)
        loss = F.cross_entropy(logits, labels)
        with torch.no_grad():
            hardest_negative = distances[:, 1:].min(dim=1).values
            metrics = {
                "{}_accuracy".format(prefix): (distances.argmin(dim=1) == 0).float().mean(),
                "{}_positive_distance".format(prefix): distances[:, 0].mean(),
                "{}_hardest_negative_distance".format(prefix): hardest_negative.mean(),
                "{}_margin".format(prefix): (hardest_negative - distances[:, 0]).mean(),
            }
            if selected_similarity is not None:
                metrics["{}_selected_cosine".format(prefix)] = selected_similarity.mean()
        return loss, metrics

    def _action_ranking_losses(
        self,
        encoded_control: Tensor,
        encoded_target: Tensor,
        condition_ids: Tensor,
        positive_action_embedding: Tensor,
    ) -> _ActionRankingResult:
        candidate_blocks = [condition_ids.unsqueeze(1)]
        action_negative_count = 0
        action_selected_similarity = None
        if self.config.action_rank_weight > 0:
            action_negative_ids, action_selected_similarity = self._hard_action_negatives(
                condition_ids, positive_action_embedding
            )
            action_negative_count = int(action_negative_ids.shape[1])
            candidate_blocks.append(action_negative_ids)

        dose_negative_count = 0
        if getattr(self.config, "dose_rank_weight", 0.0) > 0:
            dose_negative_ids = self._same_drug_dose_negatives(condition_ids)
            dose_negative_count = int(dose_negative_ids.shape[1])
            candidate_blocks.append(dose_negative_ids)

        distances = self._candidate_action_distances(
            encoded_control,
            encoded_target,
            torch.cat(candidate_blocks, dim=1),
        )
        positive_distance = distances[:, :1]
        cursor = 1
        action_loss = None
        dose_loss = None
        metrics: Dict[str, Tensor] = {}
        if action_negative_count:
            action_distances = torch.cat(
                (positive_distance, distances[:, cursor : cursor + action_negative_count]), dim=1
            )
            cursor += action_negative_count
            action_loss, action_metrics = self._ranking_objective(
                action_distances,
                prefix="action_rank",
                selected_similarity=action_selected_similarity,
            )
            metrics.update(action_metrics)
        if dose_negative_count:
            dose_distances = torch.cat(
                (positive_distance, distances[:, cursor : cursor + dose_negative_count]), dim=1
            )
            dose_loss, dose_metrics = self._ranking_objective(
                dose_distances,
                prefix="dose_rank",
            )
            metrics.update(dose_metrics)
        return _ActionRankingResult(
            action_loss=action_loss,
            dose_loss=dose_loss,
            metrics=metrics,
        )

    def _action_ranking_loss(
        self,
        encoded_control: Tensor,
        encoded_target: Tensor,
        condition_ids: Tensor,
        positive_action_embedding: Tensor,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Compatibility wrapper for the original drug-ranking objective."""

        result = self._action_ranking_losses(
            encoded_control,
            encoded_target,
            condition_ids,
            positive_action_embedding,
        )
        if result.action_loss is None:
            raise RuntimeError("action ranking is disabled")
        return result.action_loss, {
            name: value for name, value in result.metrics.items() if name.startswith("action_rank_")
        }

    @staticmethod
    def _population_axis_metrics(prefix: str, values: Tensor) -> Dict[str, Tensor]:
        values = values.float()
        centers = values.mean(dim=1)
        within = (values - centers.unsqueeze(1)).square().mean().sqrt()
        if values.shape[0] > 1:
            between = (centers - centers.mean(dim=0)).square().mean().sqrt()
        else:
            between = centers.new_zeros(())
        return {
            "{}_within_n_rms".format(prefix): within.detach(),
            "{}_between_b_rms".format(prefix): between.detach(),
            "{}_within_to_between".format(prefix): (within / between.clamp_min(1e-12)).detach(),
            "{}_n_feature_std".format(prefix): values.std(dim=1, unbiased=False).mean().detach(),
        }

    @staticmethod
    def _diagnostics(
        encoded_control: Tensor,
        encoded_target: Optional[Tensor],
        prediction: Tensor,
        action_embedding: Tensor,
    ) -> Dict[str, Tensor]:
        with torch.no_grad():
            states = (
                encoded_control
                if encoded_target is None
                else torch.cat((encoded_control, encoded_target), dim=0)
            )
            states = states.float()
            prediction_float = prediction.float()
            action_float = action_embedding.float()
            metrics = {
                "latent_mean": states.mean().detach(),
                "latent_std": states.std(unbiased=False).detach(),
                "latent_rms": states.square().mean().sqrt().detach(),
                "latent_feature_std": states.flatten(0, 1)
                .std(dim=0, unbiased=False)
                .mean()
                .detach(),
                "prediction_mean": prediction_float.mean().detach(),
                "prediction_std": prediction_float.std(unbiased=False).detach(),
                "prediction_rms": prediction_float.square().mean().sqrt().detach(),
                "action_rms": action_float.square().mean().sqrt().detach(),
            }
            if action_float.ndim == 3 and action_float.shape[1] == 1:
                token = action_float[:, 0]
                metrics["condition_token_rms"] = token.square().mean().sqrt().detach()
                metrics["condition_token_between_b_rms"] = (
                    (token - token.mean(dim=0)).square().mean().sqrt().detach()
                )
            elif action_float.ndim == 3 and action_float.shape[1] == 2:
                for token_name, token in zip(("drug_token", "dose_token"), action_float.unbind(1)):
                    metrics["{}_rms".format(token_name)] = token.square().mean().sqrt().detach()
                    metrics["{}_between_b_rms".format(token_name)] = (
                        (token - token.mean(dim=0)).square().mean().sqrt().detach()
                    )
            metrics.update(PerturbWorldModel._population_axis_metrics("control", encoded_control))
            metrics.update(PerturbWorldModel._population_axis_metrics("prediction", prediction))
            if encoded_target is not None:
                metrics.update(PerturbWorldModel._population_axis_metrics("target", encoded_target))
                control_center = encoded_control.float().mean(dim=1)
                target_center = encoded_target.float().mean(dim=1)
                metrics["control_target_center_rms_gap"] = (
                    (control_center - target_center).square().mean().sqrt().detach()
                )
                metrics["control_target_center_cosine_similarity"] = (
                    torch.cosine_similarity(control_center, target_center, dim=-1).mean().detach()
                )
            return metrics

    def _encode_states(
        self,
        tokenizer_inputs: Dict[str, Tensor],
        *,
        has_target: bool,
        require_effect_target: bool,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        """Encode prediction targets and a same-coordinate PEA target.

        The third result must share the control encoder's coordinate system. It
        may differ from the prediction target in models that use a momentum
        encoder for the main objective.
        """

        target_tokens, control_tokens = self.gene_tokenizer(tokenizer_inputs)
        batch_size = int(control_tokens.shape[0])
        if has_target:
            populations = torch.cat((control_tokens, target_tokens), dim=0)
            encoded = self.encoder(populations)
            encoded_control, encoded_target = encoded.split(batch_size, dim=0)
            effect_target = encoded_target if require_effect_target else None
            return encoded_control, encoded_target, effect_target
        return self.encoder(control_tokens), None, None

    def forward(
        self,
        gene_ids: Tensor,
        gene_values: Tensor,
        cell_offsets: Tensor,
        cell_value_sums: Tensor,
        target_cell_map: Optional[Tensor],
        control_cell_map: Tensor,
        condition_ids: Tensor,
        return_latents: bool = False,
        return_dict: Optional[bool] = None,
    ) -> Union[PerturbWorldModelOutput, Tuple[Tensor, ...]]:
        """Run training/evaluation from a flat sparse batch.

        Supplying ``target_cell_map`` computes the configured training objective. Omitting it
        performs control-only prediction and returns no loss. Set
        ``return_latents=True`` for inference or detailed evaluation; training keeps
        these large tensors out of the returned ``ModelOutput`` by default.
        """

        return_dict = self.config.use_return_dict if return_dict is None else return_dict
        if not isinstance(control_cell_map, Tensor):
            raise TypeError("control_cell_map must be a torch.Tensor")
        if control_cell_map.ndim != 2:
            raise ValueError("control_cell_map must have shape [batch, cells]")
        has_target = target_cell_map is not None
        if has_target:
            if not isinstance(target_cell_map, Tensor):
                raise TypeError("target_cell_map must be a torch.Tensor")
            if target_cell_map.shape != control_cell_map.shape:
                raise ValueError("target and control cell maps must have identical shapes")
        tokenizer_target_map = control_cell_map if target_cell_map is None else target_cell_map
        tokenizer_inputs = {
            "gene_ids": gene_ids,
            "gene_values": gene_values,
            "cell_offsets": cell_offsets,
            "cell_value_sums": cell_value_sums,
            "target_cell_map": tokenizer_target_map,
            "control_cell_map": control_cell_map,
        }
        require_effect_target = has_target and self.bulk_effect_encoder is not None
        encoded_control, encoded_target, encoded_effect_target = self._encode_states(
            tokenizer_inputs,
            has_target=has_target,
            require_effect_target=require_effect_target,
        )
        batch_size = int(encoded_control.shape[0])
        if not isinstance(condition_ids, Tensor):
            raise TypeError("condition_ids must be a torch.Tensor")
        if condition_ids.shape != (batch_size,):
            raise ValueError("condition_ids must have one entry per population")
        if condition_ids.device != encoded_control.device:
            raise ValueError("condition_ids and cell tokens must be on the same device")

        action_embedding = self.action_encoder(condition_ids)
        prediction = self.predictor(encoded_control, action_embedding)

        distribution_loss = None
        sigreg_loss = None
        sigreg_raw_loss = None
        sigreg_normalized_loss = None
        bulk_effect_loss = None
        bulk_effect_sigreg_loss = None
        bulk_effect_sigreg_raw_loss = None
        bulk_effect_sigreg_normalized_loss = None
        bulk_effect_embedding = None
        action_rank_loss = None
        dose_rank_loss = None
        ranking_metrics: Dict[str, Tensor] = {}
        bulk_difference = None
        center_difference = None
        total_loss = None
        if encoded_target is not None:
            differentiable_zero = prediction.float().sum() * 0.0
            distribution_loss = (
                differentiable_zero
                if self.distribution_criterion is None
                else self.distribution_criterion(prediction, encoded_target)
            )
            if self.sigreg_regularizer is None:
                sigreg_loss = differentiable_zero
            else:
                sigreg_result = self.sigreg_regularizer.compute(encoded_control, encoded_target)
                sigreg_loss = sigreg_result.loss
                sigreg_raw_loss = sigreg_result.raw_loss
                sigreg_normalized_loss = sigreg_result.normalized_loss
            if self.bulk_effect_encoder is not None:
                if encoded_effect_target is None:
                    raise RuntimeError(
                        "PEA requires a target encoded in the control coordinate system"
                    )
                bulk_difference = build_bulk_difference(
                    gene_ids, gene_values, cell_offsets,
                    target_cell_map, control_cell_map, num_genes=self.config.num_genes,
                )
                bulk_effect_embedding = self.bulk_effect_encoder(bulk_difference)
                center_difference = (
                    encoded_effect_target.float().mean(dim=1)
                    - encoded_control.float().mean(dim=1)
                )
                # A fixed target scale removes the rescaling escape of an
                # unconstrained learned predictor of two centers. Both paths learn.
                bulk_effect_loss = (
                    center_difference / float(self.config.bulk_effect_scale)
                    - bulk_effect_embedding.float()
                ).square().mean()
                if self.bulk_effect_regularizer is not None:
                    effect_result = self.bulk_effect_regularizer.compute(bulk_effect_embedding)
                    bulk_effect_sigreg_loss = effect_result.loss
                    bulk_effect_sigreg_raw_loss = effect_result.raw_loss
                    bulk_effect_sigreg_normalized_loss = effect_result.normalized_loss
            total_loss = (
                float(self.config.distribution_loss_weight) * distribution_loss
                + float(self.config.sigreg_weight) * sigreg_loss
            )
            if self.config.action_rank_weight > 0 or getattr(self.config, "dose_rank_weight", 0.0) > 0:
                ranking = self._action_ranking_losses(
                    encoded_control,
                    encoded_target,
                    condition_ids,
                    action_embedding,
                )
                action_rank_loss = ranking.action_loss
                dose_rank_loss = ranking.dose_loss
                ranking_metrics = ranking.metrics
            if action_rank_loss is not None:
                total_loss = total_loss + (
                    float(self.config.action_rank_weight) * action_rank_loss
                )
            if dose_rank_loss is not None:
                total_loss = total_loss + float(getattr(self.config, "dose_rank_weight", 0.0)) * dose_rank_loss
            if bulk_effect_loss is not None:
                total_loss = total_loss + float(self.config.bulk_effect_weight) * bulk_effect_loss
            if bulk_effect_sigreg_loss is not None:
                total_loss = total_loss + (
                    float(self.config.bulk_effect_sigreg_weight) * bulk_effect_sigreg_loss
                )

        diagnostics = self._diagnostics(
            encoded_control, encoded_target, prediction, action_embedding
        )
        diagnostics.update(
            {name: value.detach() for name, value in ranking_metrics.items()}
        )
        with torch.no_grad():
            for name, value, weight in (
                ("distribution", distribution_loss, self.config.distribution_loss_weight),
                ("state_sigreg", sigreg_loss, self.config.sigreg_weight),
                ("bulk_effect", bulk_effect_loss, self.config.bulk_effect_weight),
                ("bulk_effect_sigreg", bulk_effect_sigreg_loss, self.config.bulk_effect_sigreg_weight),
                ("action_rank", action_rank_loss, self.config.action_rank_weight),
                ("dose_rank", dose_rank_loss, getattr(self.config, "dose_rank_weight", 0.0)),
            ):
                if value is not None:
                    diagnostics["weighted_{}_loss".format(name)] = value.detach() * float(weight)
            if bulk_effect_embedding is not None:
                effect = bulk_effect_embedding.detach().float()
                diagnostics.update({
                    "raw_bulk_difference_rms": bulk_difference.square().mean().sqrt().detach(),
                    "bulk_effect_rms": effect.square().mean().sqrt(),
                    "bulk_effect_between_b_rms": (effect - effect.mean(dim=0)).square().mean().sqrt(),
                    "bulk_effect_target_rms": (
                        effect * float(self.config.bulk_effect_scale)
                    ).square().mean().sqrt(),
                    "bulk_effect_alignment_cosine": torch.cosine_similarity(
                        center_difference.detach(), effect, dim=-1
                    ).mean(),
                })
        if not return_dict:
            values = (
                total_loss,
                distribution_loss,
                sigreg_loss,
                prediction if return_latents else None,
                encoded_target if return_latents else None,
                encoded_control if return_latents else None,
                diagnostics,
                sigreg_raw_loss,
                sigreg_normalized_loss,
                bulk_effect_loss,
                bulk_effect_sigreg_loss,
                bulk_effect_sigreg_raw_loss,
                bulk_effect_sigreg_normalized_loss,
                bulk_effect_embedding if return_latents else None,
                action_rank_loss,
                dose_rank_loss,
            )
            return tuple(value for value in values if value is not None)
        return PerturbWorldModelOutput(
            loss=total_loss,
            distribution_loss=distribution_loss,
            sigreg_loss=sigreg_loss,
            sigreg_raw_loss=sigreg_raw_loss,
            sigreg_normalized_loss=sigreg_normalized_loss,
            predicted_perturbed=prediction if return_latents else None,
            encoded_target=encoded_target if return_latents else None,
            encoded_control=encoded_control if return_latents else None,
            diagnostics=diagnostics,
            bulk_effect_loss=bulk_effect_loss,
            bulk_effect_sigreg_loss=bulk_effect_sigreg_loss,
            bulk_effect_sigreg_raw_loss=bulk_effect_sigreg_raw_loss,
            bulk_effect_sigreg_normalized_loss=bulk_effect_sigreg_normalized_loss,
            bulk_effect_embedding=bulk_effect_embedding if return_latents else None,
            action_rank_loss=action_rank_loss,
            dose_rank_loss=dose_rank_loss,
        )


__all__ = [
    "PerturbWorldModel",
    "PerturbWorldModelConfig",
    "PerturbWorldModelOutput",
]
