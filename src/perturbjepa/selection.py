"""Save and test the weights selected by the training run's validation policy."""

from pathlib import Path

from .retrieval import sha256, write_json


def finish_training(trainer, result, output_dir):
    """Trainer reloads the validation optimum before returning from train()."""
    output_dir = Path(output_dir)
    selected_step = trainer.state.global_step
    checkpoint = None
    if trainer.args.load_best_model_at_end:
        checkpoint = trainer.state.best_model_checkpoint
        if checkpoint is None or trainer.state.best_metric is None:
            raise RuntimeError("Training ended without a validation-selected checkpoint")
        if not Path(checkpoint).is_dir():
            raise FileNotFoundError(f"Validation-selected checkpoint was removed: {checkpoint}")
        selected_step = getattr(trainer.state, "best_global_step", None)
        if selected_step is None:
            selected_step = int(Path(checkpoint).name.removeprefix("checkpoint-"))

    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    trainer.save_state()
    trainer.log_metrics("train", result.metrics)
    trainer.save_metrics("train", result.metrics)
    identity = {
        "selection": "best_validation" if checkpoint else "last_training_step",
        "path": str(final_dir.resolve()),
        "training_output": str(output_dir.resolve()),
        "global_step": int(selected_step),
        "training_global_step": trainer.state.global_step,
        "best_model_checkpoint": checkpoint,
        "metric_for_best_model": trainer.args.metric_for_best_model if checkpoint else None,
        "metric_value": trainer.state.best_metric if checkpoint else None,
    }
    if trainer.is_world_process_zero():
        identity["artifact_sha256"] = {
            path.name: sha256(path) for path in sorted(final_dir.iterdir())
            if path.name == "config.json" or path.suffix in {".safetensors", ".bin"}
        }
        write_json(output_dir / "model_selection.json", identity)
    if trainer.retrieval:
        metrics = trainer.evaluate_retrieval_split("test", checkpoint=identity)
        trainer.log(metrics)
        trainer.save_metrics("test_retrieval", metrics)
    return identity
