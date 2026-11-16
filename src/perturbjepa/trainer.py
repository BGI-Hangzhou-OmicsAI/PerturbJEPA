"""Hugging Face Trainer for epoch-resampled Parse10M JEPA batches."""

from collections.abc import Mapping
from numbers import Real

import torch
from transformers import Trainer, TrainerCallback

from data import EpochSampler


_LOSSES = (
    "distribution_loss",
    "action_rank_loss",
    "sigreg_loss",
    "sigreg_raw_loss",
    "sigreg_normalized_loss",
    "bulk_effect_loss",
    "bulk_effect_sigreg_loss",
    "bulk_effect_sigreg_raw_loss",
    "bulk_effect_sigreg_normalized_loss",
)


class _ScalarMetrics:
    def __init__(self):
        self.totals = {}

    def add(self, metrics, weight, device):
        names, values = [], []
        for name, value in metrics.items():
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                value = value.detach().to(device=device, dtype=torch.float32).reshape(())
            elif isinstance(value, Real) and not isinstance(value, bool):
                value = torch.tensor(value, dtype=torch.float32, device=device)
            else:
                continue
            names.append(name)
            values.append(value)
        if values:
            values = torch.stack(values)
            finite = torch.isfinite(values)
            sums = torch.where(finite, values, 0.0) * weight
            counts = finite.to(dtype=values.dtype) * weight
            totals = torch.stack((sums, counts), dim=1)
            schema = tuple(names)
            if schema in self.totals:
                self.totals[schema].add_(totals)
            else:
                self.totals[schema] = totals

    def flush(self, device):
        totals_by_name = {}
        for names, totals in self.totals.items():
            for name, total in zip(names, totals.unbind()):
                if name in totals_by_name:
                    totals_by_name[name] = totals_by_name[name] + total
                else:
                    totals_by_name[name] = total
        self.totals.clear()

        names = sorted(totals_by_name)
        distributed = torch.distributed.is_initialized()
        if distributed:
            rank_names = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(rank_names, names)
            names = sorted({name for local in rank_names for name in local})
        if not names:
            return {}

        zero = torch.zeros(2, dtype=torch.float32, device=device)
        totals = torch.stack([totals_by_name.get(name, zero) for name in names])
        if distributed:
            torch.distributed.all_reduce(totals)
        return {
            name: total / count
            for name, (total, count) in zip(names, totals.cpu().tolist())
            if count > 0
        }


class _SetEpochBeforeResumeSkip(TrainerCallback):
    def on_epoch_begin(self, args, state, control, train_dataloader=None, **kwargs):
        if hasattr(train_dataloader, "set_epoch"):
            # Mid-epoch DDP resume adds a skip wrapper around the sharded sampler.
            # Set the epoch on the original prepared loader before that wrapping.
            train_dataloader.set_epoch(int(state.epoch or 0))


class Parse10MTrainer(Trainer):
    """Keep Trainer batching, scheduling and checkpoints with Parse10M sampling."""

    def __init__(self, *args, retrieval=None, **kwargs):
        self._train_scalars = _ScalarMetrics()
        self._eval_scalars = _ScalarMetrics()
        self.retrieval = retrieval
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False
        self.label_names = []
        self.can_return_loss = True
        self.add_callback(_SetEpochBeforeResumeSkip())
        if self.is_deepspeed_enabled or self.is_fsdp_enabled:
            raise ValueError("Parse10M JEPA uses Trainer's standard AdamW/DDP optimizer path")

    def _get_train_sampler(self, train_dataset=None):
        dataset = self.train_dataset if train_dataset is None else train_dataset
        return EpochSampler(dataset)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        scalars = {
            "loss/" + name.removesuffix("_loss"): getattr(outputs, name, None)
            for name in _LOSSES
        }
        diagnostics = getattr(outputs, "diagnostics", None)
        if isinstance(diagnostics, Mapping):
            scalars.update({"diagnostics/" + name: value for name, value in diagnostics.items()})
        metrics = self._train_scalars if model.training else self._eval_scalars
        metrics.add(scalars, weight=inputs["target_cell_map"].shape[0], device=self.args.device)
        return (outputs.loss, outputs) if return_outputs else outputs.loss

    def evaluation_loop(self, dataloader, description, prediction_loss_only=None,
                        ignore_keys=None, metric_key_prefix="eval"):
        self._eval_scalars.totals.clear()
        output = super().evaluation_loop(
            dataloader, description, prediction_loss_only=True,
            ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix,
        )
        metrics = self._eval_scalars.flush(self.args.device)
        output.metrics.update({metric_key_prefix + "_" + name: value for name, value in metrics.items()})
        if self.retrieval:
            retrieval_metrics = self.evaluate_retrieval_split("val", bootstrap_replicates=0)
            output.metrics.update(retrieval_metrics)
            # Trainer requires an eval_ metric to track, preserve and reload the
            # validation optimum. Keep existing logging keys for prior runs.
            output.metrics[metric_key_prefix + '_retrieval_mrr'] = retrieval_metrics['retrieval/val/macro/mrr']
        return output

    def evaluate_retrieval_split(self, split, *, bootstrap_replicates=None, checkpoint=None):
        """Rank zero scores; every rank receives either metrics or the same error."""
        from pathlib import Path
        from data import Parse10MDataset
        from .retrieval import evaluate_retrieval, log_metrics, validate_checkpoint_data

        payload = [None]
        if self.is_world_process_zero():
            try:
                settings = self.retrieval
                options = settings["dataset_options"]
                store = options.get("store", getattr(self.train_dataset, "store", None))
                if store is not None:
                    options = {**options, "store": store}
                model = self.accelerator.unwrap_model(self.model_wrapped)
                validate_checkpoint_data(model.config, options["root"], options["protocol"], options["fold"])
                dataset = Parse10MDataset(
                    **{**options, "seed": settings.get("seed", 20260930),
                       "cells_per_group": settings.get("cells_per_group", 256)}, split=split,
                )
                checkpoint_identity = checkpoint if checkpoint is not None else {
                    "training_output": self.args.output_dir, "global_step": self.state.global_step,
                }
                evaluation_step = checkpoint_identity["global_step"]
                # A fresh directory also permits repeat evaluate() calls at the same step.
                directory = Path(self.args.output_dir) / "retrieval" / f"step-{evaluation_step}" / split
                suffix = 1
                while directory.exists():
                    directory = directory.with_name(f"{split}-{suffix}")
                    suffix += 1
                report = evaluate_retrieval(
                    model, dataset, directory, candidate_scope=settings.get("candidate_scope", "all_actions"),
                    candidate_batch_size=settings.get("candidate_batch_size", 32),
                    precision="bf16" if self.args.bf16 else "fp32",
                    bootstrap_replicates=(settings.get("bootstrap_replicates", 4000)
                                          if bootstrap_replicates is None else bootstrap_replicates),
                    checkpoint=checkpoint_identity,
                )
                payload[0] = {"metrics": log_metrics(report, split)}
            except Exception as error:
                payload[0] = {"error": f"{type(error).__name__}: {error}"}
        if torch.distributed.is_initialized():
            torch.distributed.broadcast_object_list(payload, src=0)
        if "error" in payload[0]:
            raise RuntimeError(f"Retrieval failed: {payload[0]['error']}")
        return payload[0]["metrics"]

    def log(self, logs, start_time=None):
        if "loss" in logs or "train_loss" in logs:
            logs = {**logs, **self._train_scalars.flush(self.args.device)}
        super().log(logs, start_time=start_time)
