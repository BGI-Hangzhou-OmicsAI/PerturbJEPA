"""Train the Tiny PerturbJEPA experiments with epoch or visit sampling."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from data import Parse10MDataset, condition_labels
from data.visit import visit_sampling_config

from .core.configuration import PerturbWorldModelConfig, validate_experiment_sections
from .selection import finish_training


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECIPE = PROJECT_ROOT / "configs/parse10m_tiny_epoch_35400.yaml"


def _absolute(path, base):
    path = Path(path).expanduser()
    return (base / path).resolve()


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model_code_sha256():
    package = Path(__file__).resolve().parent
    sources = sorted((package / "core").glob("*.py"))
    sources.append(package / "model.py")
    digest = hashlib.sha256()
    for source in sources:
        digest.update(str(source.relative_to(package)).encode())
        digest.update(source.read_bytes())
    return digest.hexdigest()


def _run_name(protocol, fold):
    if protocol == "cell_type":
        return f"cell_type_fold_{fold:02d}"
    return protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_RECIPE)
    parser.add_argument("--protocol", choices=("drug_cell_type", "drug", "cell_type"))
    parser.add_argument("--fold", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume-from-checkpoint", metavar="PATH|latest")
    parser.add_argument("--swanlab-mode", choices=("offline", "local", "cloud"))
    parser.add_argument("--check", action="store_true", help="Validate data and configuration before training.")
    cli = parser.parse_args()

    recipe_path = cli.config.resolve()
    recipe = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
    data = recipe["data"]
    model_settings = dict(recipe["model"])
    model_settings.setdefault("action_encoder_type", "condition_id")
    if model_settings["action_encoder_type"] != "condition_id":
        raise ValueError("Parse10M training requires condition_id action encoding")
    if "action_features" in data:
        raise ValueError("Condition-ID training does not read external action features")
    sampling = visit_sampling_config(data, recipe["training"])
    protocol = cli.protocol or data["protocol"]
    fold = cli.fold if cli.fold is not None else data.get("fold")
    if protocol == "cell_type":
        if fold not in range(1, 8):
            parser.error("cell_type requires --fold 1 through 7")
    elif fold is not None:
        parser.error("--fold is only valid with --protocol cell_type")

    dataset_root = _absolute(data["dataset_root"], recipe_path.parent)
    dataset_options = {
        "root": dataset_root,
        "protocol": protocol,
        "fold": fold,
        "cells_per_group": data["cells_per_group"],
        "seed": data["seed"],
    }
    train = Parse10MDataset(**dataset_options, split="train")
    dataset_options["store"] = train.store
    validation = Parse10MDataset(**dataset_options, split="val")

    source = dataset_root / "manifest.json"
    manifest = json.loads(source.read_text(encoding="utf-8"))
    conditions = condition_labels(manifest)
    condition_ids = {name: index for index, name in enumerate(conditions)}
    train_condition_ids = sorted(condition_ids[name] for name in train.conditions)
    vocabulary_sha256 = hashlib.sha256(
        json.dumps(conditions, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    action_contract = {"action_vocabulary_sha256": vocabulary_sha256}
    genes_file = dataset_root / "genes.npy"
    genes = np.load(genes_file, allow_pickle=False).tolist()
    if len(genes) != manifest["stats"]["retained_genes"] or len(set(genes)) != len(genes):
        raise ValueError("Parse10M gene vocabulary disagrees with the source manifest")

    loss_settings = recipe["loss"]
    validate_experiment_sections(model_settings, loss_settings)
    if "num_genes" in model_settings:
        raise ValueError("Gene count is derived from data")
    if loss_settings.get("dose_rank_weight", 0):
        raise ValueError("This recipe expects one condition per cytokine.")
    if loss_settings.get("action_rank_weight", 0) and len(train_condition_ids) <= loss_settings.get(
        "action_rank_num_negatives", 8
    ):
        raise ValueError("The training vocabulary is too small for action ranking")

    training = dict(recipe["training"])
    output_base = _absolute(training.pop("output_dir"), PROJECT_ROOT)
    split_name = _run_name(protocol, fold)
    output_dir = cli.output_dir.resolve() if cli.output_dir else output_base / split_name
    run_name = training.pop("run_name", "parse10m") + "_" + split_name
    for key in ("report_to", "remove_unused_columns", "ignore_data_skip"):
        if key in training:
            raise ValueError(f"The training entry point manages {key}")
    if training.get("load_best_model_at_end", False):
        if not recipe.get("retrieval", {}).get("enabled", False):
            raise ValueError("Validation MRR selection requires retrieval.enabled=true")
        if training.get("metric_for_best_model") != "eval_retrieval_mrr":
            raise ValueError("Select the validation macro MRR with metric_for_best_model=eval_retrieval_mrr")
        if training.get("greater_is_better") is not True:
            raise ValueError("Validation MRR selection requires greater_is_better=true")
    if loss_settings.get("sigreg_weight", 0) or loss_settings.get("bulk_effect_sigreg_weight", 0):
        for dataset, key in (
            (train, "per_device_train_batch_size"),
            (validation, "per_device_eval_batch_size"),
        ):
            batch_size = training.get(key, 8)
            if batch_size < 2 or len(dataset) < 2:
                raise ValueError(f"{key} and split size must both be at least 2 for SIGReg")
            if not training.get("dataloader_drop_last", False) and len(dataset) % batch_size == 1:
                raise ValueError(f"{key}={batch_size} leaves a singleton SIGReg batch in {dataset.split}")

    provenance = {
        "model_code_sha256": _model_code_sha256(),
        "source_manifest_sha256": _sha256(source),
        "train_split_sha256": _sha256(train.split_path),
        "validation_split_sha256": _sha256(validation.split_path),
        **action_contract,
        "genes_sha256": _sha256(genes_file),
        "protocol": protocol,
        "fold": fold,
        "data_seed": data["seed"],
        "cells_per_group": data["cells_per_group"],
        "model": model_settings,
        "loss": loss_settings,
        "training": training,
        **({"train_sampling": data["train_sampling"]} if sampling is not None else {}),
    }
    config = PerturbWorldModelConfig(
        **model_settings,
        **loss_settings,
        num_genes=len(genes),
        gene_indices=list(range(len(genes))),
        gene_names=genes,
        gene_vocabulary_sha256=provenance["genes_sha256"],
        num_conditions=len(conditions),
        condition_labels=conditions,
        action_vocabulary_sha256=vocabulary_sha256,
        training_condition_ids=train_condition_ids,
        parse10m_contract=provenance,
    )
    if cli.check:
        print(json.dumps({
            "protocol": protocol,
            "fold": fold,
            "train_groups": len(train),
            "validation_groups": len(validation),
            "genes": config.num_genes,
            "conditions": config.num_conditions,
            "ranking_training_conditions": len(train_condition_ids),
            "action_encoder_type": config.action_encoder_type,
            "action_tokens": 1,
            "action_embedding_size": config.action_embedding_size,
            "action_rank_cells": config.action_rank_cells,
            "load_best_model_at_end": training.get("load_best_model_at_end", False),
            "metric_for_best_model": training.get("metric_for_best_model"),
            "output_dir": str(output_dir),
            "sampling_mode": "visit_decay" if sampling is not None else "epoch",
            "max_steps": training.get("max_steps"),
        }, indent=2))
        return

    from transformers import TrainingArguments, set_seed
    from transformers.trainer_utils import get_last_checkpoint

    from .batch import SparseBatchCollator
    from .model import Parse10MWorldModel
    from .trainer import Parse10MTrainer
    from .visit_trainer import Parse10MVisitTrainer, last_visit_checkpoint

    args = TrainingArguments(
        **{**training, **({"num_train_epochs": 0} if sampling is not None else {})},
        output_dir=str(output_dir),
        run_name=run_name,
        remove_unused_columns=False,
        ignore_data_skip=sampling is not None,
        report_to=[],
    )
    checkpoint = None
    if cli.resume_from_checkpoint == "latest":
        find_checkpoint = last_visit_checkpoint if sampling is not None else get_last_checkpoint
        checkpoint = find_checkpoint(str(output_dir)) if output_dir.is_dir() else None
        if checkpoint is None:
            raise FileNotFoundError(f"No Trainer checkpoint under {output_dir}")
    elif cli.resume_from_checkpoint:
        checkpoint = str(Path(cli.resume_from_checkpoint).resolve())
    elif args.process_index == 0 and output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"{output_dir} is nonempty. Pass --resume-from-checkpoint or --output-dir.")
    if checkpoint:
        checkpoint_config = json.loads((Path(checkpoint) / "config.json").read_text(encoding="utf-8"))
        if checkpoint_config.get("parse10m_contract") != provenance:
            raise ValueError("Checkpoint and requested Parse10M training contracts differ")
        checkpoint_state = json.loads((Path(checkpoint) / "trainer_state.json").read_text(encoding="utf-8"))
        if checkpoint_state.get("train_batch_size") != args.train_batch_size:
            raise ValueError("Checkpoint and requested train batch sizes differ")
        saved_world_size = checkpoint_config.get("parse10m_world_size")
        if saved_world_size != args.world_size:
            raise ValueError("Checkpoint and requested distributed world sizes differ")
    config.parse10m_world_size = args.world_size

    swan = recipe["swanlab"]
    callbacks = []
    if swan.get("enabled", True):
        from swanlab.integration.transformers import SwanLabCallback

        callbacks.append(SwanLabCallback(
            project=swan["project"],
            workspace=swan.get("workspace"),
            experiment_name=run_name,
            logdir=str(output_dir / "swanlab"),
            mode=cli.swanlab_mode or swan["mode"],
        ))

    set_seed(args.seed)
    trainer_type = Parse10MVisitTrainer if sampling is not None else Parse10MTrainer
    trainer = trainer_type(
        model=Parse10MWorldModel(config),
        args=args,
        train_dataset=train,
        eval_dataset=validation,
        data_collator=SparseBatchCollator(condition_ids),
        callbacks=callbacks,
        **({"visit_sampling": sampling} if sampling is not None else {}),
        retrieval=({**recipe["retrieval"], "dataset_options": dataset_options}
                   if recipe.get("retrieval", {}).get("enabled", False) else None),
    )
    try:
        result = trainer.train(resume_from_checkpoint=checkpoint)
        finish_training(trainer, result, output_dir)
    finally:
        train.close()


if __name__ == "__main__":
    main()
