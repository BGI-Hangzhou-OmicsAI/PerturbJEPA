"""Population distribution matching and collapse-prevention losses."""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, ContextManager, Optional

import torch
from torch import Tensor, nn

from .configuration import (
    VALID_DISTRIBUTION_LOSSES,
    VALID_SIGREG_SAMPLE_MODES,
    VALID_SIGREG_SCOPES,
)
from .sigreg import SlicedSIGReg


def _autocast_disabled(tensor: Tensor) -> ContextManager[Any]:
    try:
        return torch.autocast(device_type=tensor.device.type, enabled=False)
    except (RuntimeError, ValueError):  # pragma: no cover - uncommon device backends
        return nullcontext()


def _distributed_world_size() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_world_size())
    return 1


class PopulationDistributionLoss(nn.Module):
    """Batched GeomLoss matching between predicted and observed cell sets."""

    def __init__(
        self,
        loss: str = "energy",
        *,
        p: int = 2,
        blur: float = 0.05,
        reach: Optional[float] = None,
        scaling: float = 0.9,
        debias: bool = True,
        backend: str = "auto",
        normalize_by_sqrt_dim: bool = True,
    ) -> None:
        super().__init__()
        loss = str(loss).lower()
        if loss not in VALID_DISTRIBUTION_LOSSES:
            raise ValueError("loss must be one of {}".format(VALID_DISTRIBUTION_LOSSES))
        try:
            from geomloss import SamplesLoss
        except ImportError as exc:  # pragma: no cover - depends on runtime extras
            raise ImportError(
                "Population distribution matching requires the 'geomloss' package"
            ) from exc
        kwargs = {
            "loss": loss,
            "p": int(p),
            "blur": float(blur),
            "scaling": float(scaling),
            "debias": bool(debias),
            "backend": str(backend),
        }
        if reach is not None:
            kwargs["reach"] = float(reach)
        self.samples_loss = SamplesLoss(**kwargs)
        if not isinstance(normalize_by_sqrt_dim, bool):
            raise ValueError("normalize_by_sqrt_dim must be boolean")
        self.normalize_by_sqrt_dim = normalize_by_sqrt_dim

    def per_population(self, prediction: Tensor, target: Tensor) -> Tensor:
        if prediction.ndim != 3 or target.ndim != 3:
            raise ValueError("prediction and target must have shape [batch, cells, dim]")
        if prediction.shape != target.shape:
            raise ValueError("prediction and target must have identical shapes")
        if prediction.device != target.device:
            raise ValueError("prediction and target must be on the same device")
        with _autocast_disabled(prediction):
            prediction_float = prediction.float()
            target_float = target.float()
            if self.normalize_by_sqrt_dim:
                scale = 1.0 / math.sqrt(float(prediction.shape[-1]))
                prediction_float = prediction_float * scale
                target_float = target_float * scale
            values = self.samples_loss(prediction_float, target_float)
            values = values.float()
            if values.ndim == 0:
                values = values.expand(prediction.shape[0])
            return values.reshape(prediction.shape[0], -1).mean(dim=-1)

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        return self.per_population(prediction, target).mean()


@dataclass(frozen=True)
class SIGRegLossResult:
    """Raw SIGReg statistic and the value selected for optimization."""

    loss: Tensor
    raw_loss: Tensor
    normalized_loss: Tensor
    global_sample_count: int


def _build_sigreg_regularizer(*, num_slices: int, n_points: int, t_max: float) -> nn.Module:
    device = torch.empty(0).device
    # Initialize the fixed quadrature buffers on CPU during HF meta loading.
    with torch.device("cpu" if device.type == "meta" else device):
        return SlicedSIGReg(num_slices=int(num_slices), n_points=int(n_points), t_max=float(t_max))


def _compute_sigreg(
    regularizer: nn.Module,
    samples: Tensor,
    *,
    sample_mode: str,
    normalize_by_num_samples: bool,
) -> SIGRegLossResult:
    sample_count = int(samples.shape[-2])
    if sample_count < 2:
        raise ValueError(
            "SIGReg requires at least two samples in {!r} mode".format(sample_mode)
        )
    with _autocast_disabled(samples):
        raw_loss = regularizer(samples.float()).float()
    # The statistic averages leading axes and all-reduces sufficient statistics over
    # DDP. Only its penultimate sample axis contributes to the raw N multiplier.
    global_sample_count = sample_count * _distributed_world_size()
    normalized_loss = raw_loss / float(global_sample_count)
    return SIGRegLossResult(
        loss=normalized_loss if normalize_by_num_samples else raw_loss,
        raw_loss=raw_loss,
        normalized_loss=normalized_loss,
        global_sample_count=global_sample_count,
    )


class SIGRegLoss(nn.Module):
    """SIGReg over selected sample axes and latent states."""

    def __init__(
        self,
        *,
        num_slices: int = 256,
        n_points: int = 17,
        t_max: float = 3.0,
        scope: str = "both",
        sample_mode: str = "batch",
        normalize_by_num_samples: bool = False,
    ) -> None:
        super().__init__()
        scope = str(scope).lower()
        if scope not in VALID_SIGREG_SCOPES:
            raise ValueError("scope must be one of {}".format(VALID_SIGREG_SCOPES))
        sample_mode = str(sample_mode).lower()
        if sample_mode not in VALID_SIGREG_SAMPLE_MODES:
            raise ValueError("sample_mode must be one of {}".format(VALID_SIGREG_SAMPLE_MODES))
        if not isinstance(normalize_by_num_samples, bool):
            raise ValueError("normalize_by_num_samples must be boolean")
        self.scope = scope
        self.sample_mode = sample_mode
        self.normalize_by_num_samples = normalize_by_num_samples
        self.regularizer = _build_sigreg_regularizer(
            num_slices=num_slices, n_points=n_points, t_max=t_max,
        )

    def compute(self, encoded_control: Tensor, encoded_target: Tensor) -> SIGRegLossResult:
        if encoded_control.ndim != 3 or encoded_target.ndim != 3:
            raise ValueError(
                "encoded_control and encoded_target must have shape [batch, cells, dim]"
            )
        if encoded_control.shape != encoded_target.shape:
            raise ValueError("encoded control and target populations must align")
        if encoded_control.device != encoded_target.device:
            raise ValueError("encoded control and target must be on the same device")
        if self.sample_mode == "batch":
            control_centers = encoded_control.mean(dim=1, dtype=torch.float32)
            target_centers = encoded_target.mean(dim=1, dtype=torch.float32)
            if self.scope == "both":
                # Treat control and target population centers as one [2B, D]
                # sample set. The characteristic function is averaged over DDP.
                samples = torch.cat((control_centers, target_centers), dim=0)
            elif self.scope == "control":
                samples = control_centers
            else:
                samples = target_centers
        elif self.sample_mode == "batch_per_cell":
            control_samples = encoded_control.transpose(0, 1)
            target_samples = encoded_target.transpose(0, 1)
            if self.scope == "both":
                # [state=2, cell_slot=N, samples=B, dim=D]: states and cell
                # slots are separate tests, each measuring variation over B.
                samples = torch.stack((control_samples, target_samples), dim=0)
            elif self.scope == "control":
                samples = control_samples
            else:
                samples = target_samples
        elif self.sample_mode == "batch_cells":
            control_samples = encoded_control.reshape(-1, encoded_control.shape[-1])
            target_samples = encoded_target.reshape(-1, encoded_target.shape[-1])
            if self.scope == "both":
                # Keep control/context and target as separate [B * N, D] samples.
                samples = torch.stack((control_samples, target_samples), dim=0)
            elif self.scope == "control":
                samples = control_samples
            else:
                samples = target_samples
        else:
            if self.scope == "both":
                # [state=2, population=B, samples=N, dim=D]. The regularizer
                # averages the 2B independent statistics rather than allowing
                # between-population variation to hide within-population collapse.
                samples = torch.stack((encoded_control, encoded_target), dim=0)
            elif self.scope == "control":
                samples = encoded_control
            else:
                samples = encoded_target
        return _compute_sigreg(
            self.regularizer,
            samples,
            sample_mode=self.sample_mode,
            normalize_by_num_samples=self.normalize_by_num_samples,
        )

    def forward(self, encoded_control: Tensor, encoded_target: Tensor) -> Tensor:
        return self.compute(encoded_control, encoded_target).loss


class EmbeddingSIGRegLoss(nn.Module):
    """B-axis SIGReg for one effect embedding per population, shaped [B, D]."""

    def __init__(
        self,
        *,
        num_slices: int = 256,
        n_points: int = 17,
        t_max: float = 3.0,
        normalize_by_num_samples: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(normalize_by_num_samples, bool):
            raise ValueError("normalize_by_num_samples must be boolean")
        self.normalize_by_num_samples = normalize_by_num_samples
        self.regularizer = _build_sigreg_regularizer(
            num_slices=num_slices, n_points=n_points, t_max=t_max,
        )

    def compute(self, embeddings: Tensor) -> SIGRegLossResult:
        if embeddings.ndim != 2:
            raise ValueError("embeddings must have shape [batch, dim]")
        return _compute_sigreg(
            self.regularizer,
            embeddings,
            sample_mode="embedding_batch",
            normalize_by_num_samples=self.normalize_by_num_samples,
        )

    def forward(self, embeddings: Tensor) -> Tensor:
        return self.compute(embeddings).loss


__all__ = [
    "EmbeddingSIGRegLoss",
    "PopulationDistributionLoss",
    "SIGRegLoss",
    "SIGRegLossResult",
]
