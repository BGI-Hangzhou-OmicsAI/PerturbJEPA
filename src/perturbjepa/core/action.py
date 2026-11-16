"""Encode perturbation conditions as action vectors or tokens."""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
from torch import Tensor, nn

from .set_transformer import RMSNorm


def _action_id_maps(
    *,
    num_conditions: int,
    num_drugs: int,
    num_doses: int,
    condition_to_drug_id: Sequence[int],
    condition_to_dose_id: Sequence[int],
) -> tuple[Tensor, Tensor]:
    drug_values = list(condition_to_drug_id)
    dose_values = list(condition_to_dose_id)
    if len(drug_values) != num_conditions or len(dose_values) != num_conditions:
        raise ValueError("action mappings must contain one entry per condition")
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0 or item >= num_drugs
        for item in drug_values
    ) or any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0 or item >= num_doses
        for item in dose_values
    ):
        raise ValueError("action mappings contain an out-of-range index")
    return (
        torch.tensor(drug_values, dtype=torch.long),
        torch.tensor(dose_values, dtype=torch.long),
    )


def _lookup_action_ids(
    condition_ids: Tensor,
    *,
    num_conditions: int,
    condition_to_drug_id: Tensor,
    condition_to_dose_id: Tensor,
) -> tuple[Tensor, Tensor]:
    if not isinstance(condition_ids, Tensor):
        raise TypeError("condition_ids must be a torch.Tensor")
    if condition_ids.ndim != 1:
        raise ValueError("condition_ids must have shape [batch]")
    if condition_ids.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        raise TypeError("condition_ids must contain integers")
    condition_ids = condition_ids.to(dtype=torch.long)
    if condition_ids.numel() and (
        bool((condition_ids < 0).any()) or bool((condition_ids >= num_conditions).any())
    ):
        raise IndexError("condition_ids contains an out-of-range condition")
    return condition_to_drug_id[condition_ids], condition_to_dose_id[condition_ids]


class ConditionIDTokenEncoder(nn.Module):
    """One trainable identity token per factor; no external features or dose."""

    num_action_tokens = 1

    def __init__(self, *, num_conditions, hidden_size, embedding_size=None,
                 normalize_output=False, rms_norm_eps=1e-6):
        super().__init__()
        self.num_conditions = num_conditions
        width = hidden_size if embedding_size is None else embedding_size
        self.condition_embedding = nn.Embedding(num_conditions, width)
        self.projection = nn.Identity() if width == hidden_size else nn.Linear(width, hidden_size)
        self.output_norm = RMSNorm(hidden_size, eps=rms_norm_eps) if normalize_output else nn.Identity()

    def forward(self, condition_ids):
        if not isinstance(condition_ids, Tensor) or condition_ids.ndim != 1:
            raise ValueError("condition_ids must be a one-dimensional tensor")
        if condition_ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise TypeError("condition_ids must contain integers")
        if condition_ids.numel() and (
            bool((condition_ids < 0).any()) or bool((condition_ids >= self.num_conditions).any())
        ):
            raise IndexError("condition_ids contains an out-of-range factor ID")
        values = self.projection(self.condition_embedding(condition_ids.long()))
        return self.output_norm(values).unsqueeze(1)


class DrugDoseActionEncoder(nn.Module):
    """Map a condition ID to learned drug and categorical-dose representations."""

    def __init__(
        self,
        *,
        num_conditions: int,
        num_drugs: int,
        num_doses: int,
        condition_to_drug_id: Sequence[int],
        condition_to_dose_id: Sequence[int],
        hidden_size: int,
        embedding_size: Optional[int] = None,
        normalize_output: bool = False,
        rms_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.num_conditions = int(num_conditions)
        self.hidden_size = int(hidden_size)
        embedding_size = self.hidden_size if embedding_size is None else int(embedding_size)
        drug_map, dose_map = _action_id_maps(
            num_conditions=self.num_conditions,
            num_drugs=int(num_drugs),
            num_doses=int(num_doses),
            condition_to_drug_id=condition_to_drug_id,
            condition_to_dose_id=condition_to_dose_id,
        )
        self.register_buffer("condition_to_drug_id", drug_map, persistent=True)
        self.register_buffer("condition_to_dose_id", dose_map, persistent=True)

        self.drug_embedding = nn.Embedding(int(num_drugs), embedding_size)
        self.dose_embedding = nn.Embedding(int(num_doses), embedding_size)
        self.output = nn.Sequential(
            nn.Linear(2 * embedding_size, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.output_norm: nn.Module = (
            RMSNorm(self.hidden_size, eps=rms_norm_eps) if normalize_output else nn.Identity()
        )

    def forward(self, condition_ids: Tensor) -> Tensor:
        drug_ids, dose_ids = _lookup_action_ids(
            condition_ids,
            num_conditions=self.num_conditions,
            condition_to_drug_id=self.condition_to_drug_id,
            condition_to_dose_id=self.condition_to_dose_id,
        )
        drug = self.drug_embedding(drug_ids)
        dose = self.dose_embedding(dose_ids)
        return self.output_norm(self.output(torch.cat((drug, dose), dim=-1)))


class DrugDoseTokenEncoder(nn.Module):
    """Encode each condition as separate drug and continuous-dose tokens."""

    num_action_tokens = 2

    def __init__(
        self,
        *,
        num_conditions: int,
        num_drugs: int,
        num_doses: int,
        condition_to_drug_id: Sequence[int],
        condition_to_dose_id: Sequence[int],
        dose_values: Sequence[float],
        hidden_size: int,
        embedding_size: Optional[int] = None,
        normalize_output: bool = False,
        rms_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.num_conditions = int(num_conditions)
        self.hidden_size = int(hidden_size)
        embedding_size = self.hidden_size if embedding_size is None else int(embedding_size)
        if embedding_size <= 0:
            raise ValueError("embedding_size must be positive")

        drug_map, dose_map = _action_id_maps(
            num_conditions=self.num_conditions,
            num_drugs=int(num_drugs),
            num_doses=int(num_doses),
            condition_to_drug_id=condition_to_drug_id,
            condition_to_dose_id=condition_to_dose_id,
        )
        self.register_buffer("condition_to_drug_id", drug_map, persistent=True)
        self.register_buffer("condition_to_dose_id", dose_map, persistent=True)

        numeric_doses = [float(value) for value in dose_values]
        if len(numeric_doses) != int(num_doses):
            raise ValueError("dose_values must contain one value per dose")
        if not all(math.isfinite(value) and value >= 0 for value in numeric_doses):
            raise ValueError("dose_values must be finite and nonnegative")
        positive_doses = [value for value in numeric_doses if value > 0]
        dose_reference = min(positive_doses) if positive_doses else 1.0
        self.register_buffer(
            "dose_values",
            torch.tensor(numeric_doses, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "dose_reference",
            torch.tensor(dose_reference, dtype=torch.float32),
            persistent=True,
        )

        self.drug_embedding = nn.Embedding(int(num_drugs), embedding_size)
        self.drug_token_projection = (
            nn.Identity()
            if embedding_size == self.hidden_size
            else nn.Linear(embedding_size, self.hidden_size)
        )
        self.dose_encoder = nn.Sequential(
            nn.Linear(1, embedding_size),
            nn.SiLU(),
            nn.Linear(embedding_size, self.hidden_size),
        )
        self.token_type_embedding = nn.Embedding(self.num_action_tokens, self.hidden_size)
        self.output_norm: nn.Module = (
            RMSNorm(self.hidden_size, eps=rms_norm_eps) if normalize_output else nn.Identity()
        )

    def forward(self, condition_ids: Tensor) -> Tensor:
        drug_ids, dose_ids = _lookup_action_ids(
            condition_ids,
            num_conditions=self.num_conditions,
            condition_to_drug_id=self.condition_to_drug_id,
            condition_to_dose_id=self.condition_to_dose_id,
        )
        drug = self.drug_token_projection(self.drug_embedding(drug_ids))
        dose = self.dose_values[dose_ids]
        continuous_dose = torch.log1p(dose / self.dose_reference).unsqueeze(-1)
        dose = self.dose_encoder(continuous_dose)
        token_types = self.token_type_embedding.weight
        tokens = torch.stack((drug + token_types[0], dose + token_types[1]), dim=1)
        return self.output_norm(tokens)


__all__ = ["ConditionIDTokenEncoder", "DrugDoseActionEncoder", "DrugDoseTokenEncoder"]
