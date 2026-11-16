"""Trainer integration for the acknowledged, already rank-sharded visit stream."""

import json
import os
from pathlib import Path

import torch
from transformers.trainer_callback import TrainerState
from transformers.trainer_utils import TrainOutput

from data.visit import PLAN_ID, Parse10MVisitLoader, VisitSamplingConfig
from .retrieval import sha256, write_json
from .trainer import Parse10MTrainer, _SetEpochBeforeResumeSkip


COMPLETE = "visit_checkpoint_complete.json"


def last_visit_checkpoint(output_dir):
    candidates = []
    for directory in Path(output_dir).glob("checkpoint-*"):
        if directory.name.removeprefix("checkpoint-").isdigit() and (directory / COMPLETE).is_file():
            candidates.append(directory)
    return str(max(candidates, key=lambda path: int(path.name.removeprefix("checkpoint-")))) if candidates else None


class VisitTrainerMixin:
    """Add step-only loading and exact per-rank sampler checkpoint commits."""

    def __init__(self, *args, visit_sampling=None, **kwargs):
        self.visit_sampling = visit_sampling or VisitSamplingConfig()
        self.visit_loader = None
        self._visit_committing = False
        super().__init__(*args, **kwargs)
        self.remove_callback(_SetEpochBeforeResumeSkip)
        self.args.ignore_data_skip = True
        if self.args.max_steps <= 0:
            raise ValueError("Visit training requires positive max_steps")
        if self.args.auto_find_batch_size or self.args.save_only_model:
            raise ValueError("Visit training requires a fixed batch size and complete checkpoints")
        if self.args.world_size == 1 and self.args.n_gpu > 1:
            raise ValueError("Use one visible GPU per process for visit training")

    def get_train_dataloader(self):
        if self.visit_loader is None:
            self.visit_loader = Parse10MVisitLoader(
                self.train_dataset, collator=self.data_collator, sampling=self.visit_sampling,
                batch_size=self.args.per_device_train_batch_size,
                num_workers=self.args.dataloader_num_workers,
                prefetch_factor=self.args.dataloader_prefetch_factor or 2,
                pin_memory=self.args.dataloader_pin_memory,
                consumer_rank=self.args.process_index, consumer_count=self.args.world_size,
            )
        # The loader already owns complete groups on this rank. It must not be
        # passed to Accelerator.prepare(), which would shard the stream again.
        return self.visit_loader

    def training_step(self, model, inputs, num_items_in_batch=None):
        plan_id = inputs.pop(PLAN_ID)
        loss = super().training_step(model, inputs, num_items_in_batch)
        self.visit_loader.mark_consumed(plan_id)
        return loss

    def log(self, logs, start_time=None):
        logs = {key: value for key, value in logs.items() if key != "epoch"}
        if self.visit_loader is not None:
            logs.update({"data/" + key: value for key, value in self.visit_loader.describe().items()})
        previous = self.state.epoch
        self.state.epoch = None
        try:
            super().log(logs, start_time=start_time)
        finally:
            self.state.epoch = previous

    def _sampler_path(self, directory):
        return Path(directory) / f"visit_sampler_rank_{self.args.process_index}.pt"

    def _save_checkpoint(self, model, trial):
        directory = Path(self._get_output_dir(trial=trial)) / f"checkpoint-{self.state.global_step}"
        marker = directory / COMPLETE
        if self.is_world_process_zero():
            marker.unlink(missing_ok=True)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        self._visit_committing = True
        try:
            super()._save_checkpoint(model, trial)
        finally:
            self._visit_committing = False
        error = None
        try:
            loader = self.get_train_dataloader()
            state = loader.state_dict()
            if state["consumed"] != self.state.global_step * self.args.gradient_accumulation_steps:
                raise ValueError("Consumed microbatches disagree with optimizer steps")
            payload = {"global_step": self.state.global_step, "max_steps": self.args.max_steps,
                       "gradient_accumulation_steps": self.args.gradient_accumulation_steps,
                       "loader": state, "sampling_metrics": loader.describe(include_visit_quantiles=True)}
            path = self._sampler_path(directory)
            temporary = path.with_suffix(".pt.tmp")
            torch.save(payload, temporary)
            os.replace(temporary, path)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        errors = [error]
        if torch.distributed.is_initialized():
            errors = [None] * self.args.world_size
            torch.distributed.all_gather_object(errors, error)
        if any(errors):
            raise RuntimeError(f"Visit checkpoint was not committed: {errors}")
        error = None
        if self.is_world_process_zero():
            try:
                hashes = {f"visit_sampler_rank_{rank}.pt": sha256(directory / f"visit_sampler_rank_{rank}.pt")
                          for rank in range(self.args.world_size)}
                temporary = directory / (COMPLETE + ".tmp")
                write_json(temporary, {"version": 1, "global_step": self.state.global_step,
                                       "world_size": self.args.world_size, "sampler_sha256": hashes})
                os.replace(temporary, marker)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        if torch.distributed.is_initialized():
            payload = [error]
            torch.distributed.broadcast_object_list(payload, src=0)
            error = payload[0]
        if error:
            raise RuntimeError(f"Visit checkpoint completion marker failed: {error}")
        if self.is_world_process_zero():
            self._rotate_checkpoints(use_mtime=True, output_dir=str(directory.parent))
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    def _rotate_checkpoints(self, use_mtime=False, output_dir=None):
        if not self._visit_committing:
            super()._rotate_checkpoints(use_mtime=use_mtime, output_dir=output_dir)

    def _load_optimizer_and_scheduler(self, checkpoint):
        # Accelerate assigns indexed CPU devices in multi-process runs, while
        # torch.load expects the unindexed "cpu" storage tag.
        if checkpoint and self.args.device.type == "cpu" and self.args.world_size > 1:
            self.optimizer.load_state_dict(torch.load(Path(checkpoint) / "optimizer.pt",
                                                       map_location="cpu", weights_only=True))
            self.lr_scheduler.load_state_dict(torch.load(Path(checkpoint) / "scheduler.pt",
                                                          map_location="cpu", weights_only=True))
        else:
            super()._load_optimizer_and_scheduler(checkpoint)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        directory = Path(resume_from_checkpoint)
        marker = directory / COMPLETE
        if not marker.is_file():
            raise ValueError("Missing complete visit checkpoint marker; epoch checkpoints cannot resume this stream")
        complete = json.loads(marker.read_text())
        if complete.get("version") != 1 or complete.get("world_size") != self.args.world_size:
            raise ValueError("Visit checkpoint rank/world size differs")
        path = self._sampler_path(directory)
        if complete.get("sampler_sha256", {}).get(path.name) != sha256(path):
            raise ValueError("Visit sampler checkpoint hash differs")
        for name in ("config.json", "trainer_state.json", "optimizer.pt", "scheduler.pt"):
            if not (directory / name).is_file():
                raise ValueError(f"Incomplete visit checkpoint: {name}")
        if not any((directory / name).is_file() for name in ("model.safetensors", "pytorch_model.bin")):
            raise ValueError("Incomplete visit checkpoint: model weights")
        rng = "rng_state.pth" if self.args.world_size == 1 else f"rng_state_{self.args.process_index}.pth"
        if not (directory / rng).is_file():
            raise ValueError("Incomplete visit checkpoint: RNG state")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        trainer_state = json.loads((directory / "trainer_state.json").read_text())
        if payload.get("global_step") != complete.get("global_step") or payload["global_step"] != trainer_state["global_step"]:
            raise ValueError("Visit checkpoint step identities differ")
        if payload.get("max_steps") != self.args.max_steps or payload.get("gradient_accumulation_steps") != self.args.gradient_accumulation_steps:
            raise ValueError("Visit checkpoint total steps or gradient accumulation differs")
        if payload["loader"].get("consumed") != payload["global_step"] * self.args.gradient_accumulation_steps:
            raise ValueError("Visit checkpoint consumed position differs")
        self.get_train_dataloader().load_state_dict(payload["loader"])
        super()._load_from_checkpoint(resume_from_checkpoint, model=model)

    def train(self, resume_from_checkpoint=None, *args, **kwargs):
        if resume_from_checkpoint is True:
            resume_from_checkpoint = last_visit_checkpoint(self.args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError("No complete visit checkpoint exists")
        try:
            if resume_from_checkpoint:
                saved = TrainerState.load_from_json(str(Path(resume_from_checkpoint) / "trainer_state.json"))
                if saved.global_step >= self.args.max_steps:
                    self._load_from_checkpoint(resume_from_checkpoint)
                    self.state = saved
                    if self.args.load_best_model_at_end and self.state.best_model_checkpoint:
                        self._load_best_model()
                    return TrainOutput(saved.global_step, 0.0, {"train_loss": 0.0})
            return super().train(resume_from_checkpoint=resume_from_checkpoint, *args, **kwargs)
        finally:
            if self.visit_loader is not None:
                self.visit_loader.close()


class Parse10MVisitTrainer(VisitTrainerMixin, Parse10MTrainer):
    pass
