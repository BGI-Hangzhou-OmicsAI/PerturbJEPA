"""Use the Perturb world model with split-aware action-ranking candidates."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from .core.world_model import PerturbWorldModel


class Parse10MWorldModel(PerturbWorldModel):
    """Train the reference objective with negatives from training conditions only."""

    def __init__(self, config) -> None:
        configured_ids = getattr(config, "training_condition_ids", None)
        if configured_ids is None:
            if config.action_rank_weight > 0:
                raise ValueError("Action ranking requires training_condition_ids in the model config")
            configured_ids = range(config.num_conditions)
        condition_ids = list(configured_ids)
        if (
            not condition_ids
            or any(type(index) is not int or not 0 <= index < config.num_conditions
                   for index in condition_ids)
            or len(set(condition_ids)) != len(condition_ids)
        ):
            raise ValueError("training_condition_ids must be unique, in-range condition IDs")

        super().__init__(config)
        # Rebuild candidate IDs from config even when Hugging Face loads on meta.
        self._training_condition_ids = tuple(sorted(condition_ids))

    def _hard_action_negatives(
        self,
        condition_ids: Tensor,
        positive_action_embedding: Tensor,
    ) -> tuple[Tensor, Tensor]:
        negative_count = int(self.config.action_rank_num_negatives)
        candidate_ids = condition_ids.new_tensor(self._training_condition_ids, dtype=torch.long)
        with torch.no_grad():
            candidate_embeddings = self.action_encoder(candidate_ids)
            queries = F.normalize(
                self._pool_action_embedding(positive_action_embedding.detach()).float(), dim=-1
            )
            candidates = F.normalize(
                self._pool_action_embedding(candidate_embeddings).float(), dim=-1
            )
            similarities = queries @ candidates.transpose(0, 1)

            if self.config.action_encoder_type == "condition_id":
                valid = candidate_ids.unsqueeze(0) != condition_ids.unsqueeze(1)
            else:
                dose_ids = self.action_encoder.condition_to_dose_id
                drug_ids = self.action_encoder.condition_to_drug_id
                valid = dose_ids[candidate_ids].unsqueeze(0) == dose_ids[condition_ids].unsqueeze(1)
                valid = valid & (
                    drug_ids[candidate_ids].unsqueeze(0) != drug_ids[condition_ids].unsqueeze(1)
                )
            valid_counts = valid.sum(dim=1)
            if bool((valid_counts < negative_count).any()):
                minimum = int(valid_counts.min().item())
                raise ValueError(
                    "Action ranking needs {} different training-factor negatives per "
                    "condition, but this batch has as few as {}".format(negative_count, minimum)
                )
            selected_similarity, selected_columns = similarities.masked_fill(
                ~valid, -torch.inf
            ).topk(negative_count, dim=1)
        return candidate_ids[selected_columns], selected_similarity

    def _hard_same_dose_negatives(self, condition_ids, positive_action_embedding):
        """Compatibility entry point for existing checkpoint analysis scripts."""
        return self._hard_action_negatives(condition_ids, positive_action_embedding)
