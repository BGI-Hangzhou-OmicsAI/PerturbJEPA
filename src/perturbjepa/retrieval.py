"""Frozen-population Parse10M retrieval using Perturb's energy-MMD protocol."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np
import torch

from data import Parse10MDataset, Parse10MStore


METRIC_NAMES = (
    "recall_at_1", "recall_at_5", "recall_at_10", "recall_at_20", "recall_at_50",
    "mrr", "ndcg_at_10", "normalized_rank",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


@contextmanager
def _strict_distance_matmul(tensor):
    if tensor.device.type != 'cuda':
        yield
        return
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def energy_mmd(predictions, target):
    """GeomLoss energy V-statistic; match Perturb's 1e-8 squared-distance floor.

    No feature rescaling or population-mean reduction is applied. The scoring
    runner disables autocast for this float32 calculation, as in Perturb.
    """
    if predictions.ndim != 3 or target.ndim != 2 or predictions.shape[-1] != target.shape[-1]:
        raise ValueError("Expected predictions [candidates, cells, features] and target [cells, features]")
    if min(predictions.shape) < 1 or min(target.shape) < 1:
        raise ValueError("Retrieval populations must be nonempty")
    with torch.autocast(device_type=predictions.device.type, enabled=False), _strict_distance_matmul(predictions):
        predicted, observed = predictions.float(), target.float()
        if not bool(torch.isfinite(predicted).all() and torch.isfinite(observed).all()):
            raise ValueError("Retrieval populations contain nonfinite values")

        def distances(left, right):
            squared = (
                left.square().sum(-1).unsqueeze(-1)
                + right.square().sum(-1).unsqueeze(-2)
                - 2 * (left @ right.transpose(-1, -2))
            )
            return squared.clamp_min(1e-8).sqrt()

        cross = distances(predicted, observed.unsqueeze(0)).mean((1, 2))
        within_prediction = distances(predicted, predicted).mean((1, 2))
        within_target = distances(observed, observed).mean()
        return (cross - 0.5 * within_prediction - 0.5 * within_target).clamp_min(0)


def rank_metrics(scores, candidate_ids, true_id):
    scores = np.asarray(scores, dtype=np.float64)
    candidate_ids = np.asarray(candidate_ids, dtype=np.int64)
    if scores.shape != candidate_ids.shape or scores.ndim != 1 or not len(scores):
        raise ValueError("Every query must have one score per candidate")
    if not np.isfinite(scores).all() or len(np.unique(candidate_ids)) != len(candidate_ids):
        raise ValueError("Candidate scores must be finite and candidate IDs unique")
    # Match Perturb: ascending distance, then canonical action identity for ties.
    order = np.lexsort((candidate_ids, scores))
    locations = np.flatnonzero(candidate_ids[order] == true_id)
    if locations.size != 1:
        raise ValueError("The correct action must occur exactly once in the candidate catalog")
    rank = int(locations[0]) + 1
    metrics = {f"recall_at_{k}": float(rank <= k) for k in (1, 5, 10, 20, 50)}
    metrics.update(
        mrr=1.0 / rank,
        ndcg_at_10=1.0 / np.log2(rank + 1) if rank <= 10 else 0.0,
        normalized_rank=(rank - 1) / (len(scores) - 1) if len(scores) > 1 else 0.0,
    )
    return rank, order, metrics


def summarize(rows, *, bootstrap_replicates=0, seed=20260930):
    """Average donors within (cell type, factor), factors within type, then types.

    This is Perturb's drug/background macro average with donor replicates in
    place of repeated queries and no dose axis. Request-equal scores are also
    retained. Bootstrap clusters are factors, shared across all cell types.
    """
    if not rows:
        raise ValueError("Cannot evaluate an empty retrieval split")
    donor_groups = defaultdict(list)
    for row in rows:
        donor_groups[(row["cell_type"], row["condition"], row["donor"])].append(
            [row[name] for name in METRIC_NAMES]
        )
    groups = defaultdict(list)
    for (cell_type, condition, _), values in donor_groups.items():
        groups[(cell_type, condition)].append(np.mean(values, axis=0))
    factors = sorted({condition for _, condition in groups})
    types = sorted({cell_type for cell_type, _ in groups})
    matrix = np.zeros((len(types), len(factors), len(METRIC_NAMES)), dtype=np.float64)
    support = np.zeros((len(types), len(factors)), dtype=bool)
    for (cell_type, condition), values in groups.items():
        i, j = types.index(cell_type), factors.index(condition)
        matrix[i, j] = np.mean(values, axis=0)
        support[i, j] = True

    def macro(weights):
        denominators = support @ weights
        available = denominators > 0
        by_type = np.einsum("tfm,f->tm", matrix, weights)[available]
        return (by_type / denominators[available, None]).mean(axis=0)

    result = {
        "macro": dict(zip(METRIC_NAMES, macro(np.ones(len(factors))).tolist())),
        "request_equal": {
            name: float(np.mean([row[name] for row in rows])) for name in METRIC_NAMES
        },
        "hit_counts": {
            f"recall_at_{k}": int(sum(row[f"recall_at_{k}"] for row in rows))
            for k in (1, 5, 10, 20, 50)
        },
        "coverage": {"total_queries": len(rows), "evaluated_queries": len(rows), "failed_queries": 0, "rate": 1.0},
        "by_cell_type": {},
        "by_donor": {},
    }
    for field, key in (("cell_type", "by_cell_type"), ("donor", "by_donor")):
        for value in sorted({row[field] for row in rows}):
            selected = [row for row in rows if row[field] == value]
            result[key][value] = {
                "queries": len(selected),
                **{name: float(np.mean([row[name] for row in selected])) for name in METRIC_NAMES},
            }
    if bootstrap_replicates:
        rng = np.random.default_rng(seed)
        estimates = [
            macro(rng.multinomial(len(factors), np.full(len(factors), 1.0 / len(factors))))
            for _ in range(bootstrap_replicates)
        ]
        intervals = np.quantile(estimates, [0.025, 0.975], axis=0)
        result["macro_ci95"] = {
            name: {"low": float(intervals[0, index]), "high": float(intervals[1, index])}
            for index, name in enumerate(METRIC_NAMES)
        }
    return result


def prepare_queries(dataset, condition_labels, candidate_scope):
    """Freeze query membership and physical rows without loading expression."""
    if candidate_scope not in {"all_actions", "target_split"}:
        raise ValueError("candidate_scope must be all_actions or target_split")
    ids = {label: index for index, label in enumerate(condition_labels)}
    if len(ids) != len(condition_labels):
        raise ValueError("Checkpoint condition labels must be unique")
    candidates_by_type = defaultdict(set)
    for pair in dataset._pairs:
        if pair["condition"] not in ids:
            raise ValueError("Split contains an action absent from the checkpoint")
        candidates_by_type[pair["cell_type"]].add(ids[pair["condition"]])
    queries, truth, control_rows, target_rows = [], [], [], []
    # Reuse predictions for each fixed PBS pool; target labels never enter the scorer.
    pairs = sorted(dataset._pairs, key=lambda pair: (pair["control_id"], pair["treated_id"]))
    for pair in pairs:
        control_id, treated_id = pair["control_id"], pair["treated_id"]
        candidate_ids = (
            list(range(len(condition_labels))) if candidate_scope == "all_actions"
            else sorted(candidates_by_type[pair["cell_type"]])
        )
        query_id = f"{dataset.split}-{treated_id}"
        queries.append({
            "query_id": query_id, "donor": pair["donor"], "cell_type": pair["cell_type"],
            "control_id": control_id, "treated_id": treated_id, "candidate_ids": candidate_ids,
        })
        truth.append({"query_id": query_id, "condition": pair["condition"], "condition_id": ids[pair["condition"]]})
        for block_id, arm, selected_rows in ((control_id, 0, control_rows), (treated_id, 1, target_rows)):
            rng = np.random.default_rng([dataset.seed, 0, block_id, arm])
            selected_rows.append(np.sort(rng.choice(
                dataset._blocks[block_id]["n_cells"], dataset.cells_per_group, replace=False,
            )))
    return queries, truth, np.asarray(control_rows), np.asarray(target_rows)


class PopulationScorer:
    """Only expression populations and candidate actions are visible to the model."""

    def __init__(self, model, *, candidate_batch_size=32, precision="bf16"):
        if candidate_batch_size < 1 or precision not in {"bf16", "fp32"}:
            raise ValueError("Invalid candidate batch size or precision")
        self.model = model
        self.device = next(model.parameters()).device
        self.candidate_batch_size = candidate_batch_size
        self.precision = precision

    def autocast(self):
        return (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if self.device.type == "cuda" and self.precision == "bf16" else nullcontext()
        )

    def encode(self, population):
        # One physical bank; both logical maps refer to this anonymous population.
        nonempty = np.flatnonzero(np.diff(population.indptr))
        sums = np.zeros(population.shape[0], dtype=np.float32)
        if nonempty.size:
            sums[nonempty] = np.add.reduceat(population.values, population.indptr[nonempty], dtype=np.float32)
        slots = torch.arange(population.shape[0], device=self.device).unsqueeze(0)
        inputs = {
            "gene_ids": torch.from_numpy(population.indices).to(self.device),
            "gene_values": torch.from_numpy(population.values).to(self.device),
            "cell_offsets": torch.from_numpy(population.indptr).to(self.device),
            "cell_value_sums": torch.from_numpy(sums).to(self.device),
            "target_cell_map": slots, "control_cell_map": slots,
        }
        with self.autocast():
            tokens, _ = self.model.gene_tokenizer(inputs)
            return self.model.encoder(tokens)

    def predict(self, control, candidate_ids):
        populations = []
        with self.autocast():
            for start in range(0, len(candidate_ids), self.candidate_batch_size):
                ids = torch.tensor(candidate_ids[start:start + self.candidate_batch_size], device=self.device)
                actions = self.model.action_encoder(ids)
                populations.append(self.model.predictor(control.expand(len(ids), -1, -1), actions))
        return torch.cat(populations)

    def distances(self, predictions, target):
        return torch.cat([
            energy_mmd(chunk, target.squeeze(0))
            for chunk in predictions.split(self.candidate_batch_size)
        ]).cpu().numpy()


@torch.inference_mode()
def evaluate_retrieval(
    model, dataset, output_dir, *, candidate_scope="all_actions", candidate_batch_size=32,
    precision="bf16", bootstrap_replicates=4000, checkpoint=None, progress_every=50,
):
    """Score every query or raise; a partial run never produces a success report."""
    if dataset.split not in {"val", "test"}:
        raise ValueError("Retrieval requires a held-out validation or test split")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Retrieval output directory is nonempty: {output_dir}")
    labels = list(model.config.condition_labels)
    queries, truth, control_rows, target_rows = prepare_queries(dataset, labels, candidate_scope)
    if not queries:
        raise ValueError("Retrieval split is empty")
    metadata = {
        "format": "parse10m-retrieval-v1", "split": dataset.split,
        "candidate_scope": candidate_scope, "condition_labels": labels,
        "cells_per_group": dataset.cells_per_group, "seed": dataset.seed,
        "distance": "energy-mmd-v1", "distance_normalize_by_sqrt_dim": False,
        "distance_squared_floor": 1e-8, "precision": precision,
        "candidate_batch_size": candidate_batch_size,
        "aggregation": "mean donors per (cell_type, factor), mean factors per cell_type, mean cell_types",
        "bootstrap_replicates": bootstrap_replicates, "bootstrap_cluster": "factor",
        "split_sha256": sha256(dataset.split_path),
        "source_manifest_sha256": sha256(dataset.root / "manifest.json"),
        "genes_sha256": sha256(dataset.root / "genes.npy"),
        "evaluation_code_sha256": sha256(Path(__file__)),
        "checkpoint": checkpoint,
        "query_count": len(queries),
        "candidate_count_min": min(len(q["candidate_ids"]) for q in queries),
        "candidate_count_max": max(len(q["candidate_ids"]) for q in queries),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("queries", queries), ("truth", truth)):
        (output_dir / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    np.savez_compressed(output_dir / "cells.npz", control_rows=control_rows, target_rows=target_rows)
    metadata["artifact_sha256"] = {
        name: sha256(output_dir / name) for name in ("queries.jsonl", "truth.jsonl", "cells.npz")
    }
    write_json(output_dir / "manifest.json", metadata)
    scorer = PopulationScorer(model, candidate_batch_size=candidate_batch_size, precision=precision)
    previous_training = model.training
    model.eval()
    scores, candidate_arrays, ranks, previous_key = [], [], [], None
    try:
        with (output_dir / "per_query.jsonl").open("w") as handle:
            for index, query in enumerate(queries):
                control_id, treated_id = query["control_id"], query["treated_id"]
                key = (control_id, tuple(query["candidate_ids"]))
                if key != previous_key:
                    population = dataset.read_rows(control_id, control_rows[index])
                    encoded_control = scorer.encode(population)
                    predictions = scorer.predict(encoded_control, query["candidate_ids"])
                    previous_key = key
                population = dataset.read_rows(treated_id, target_rows[index])
                encoded_target = scorer.encode(population)
                distances = scorer.distances(predictions, encoded_target)
                target_truth = truth[index]  # Evaluator-only label, after model scoring.
                rank, order, metrics = rank_metrics(distances, query["candidate_ids"], target_truth["condition_id"])
                row = {
                    **query, "condition": target_truth["condition"], "rank": rank,
                    "candidate_count": len(distances), "top_condition": labels[query["candidate_ids"][order[0]]],
                    "true_distance": float(distances[query["candidate_ids"].index(target_truth["condition_id"])]),
                    **metrics,
                }
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                ranks.append(row)
                scores.extend(distances.tolist())
                candidate_arrays.extend(query["candidate_ids"])
                if progress_every and ((index + 1) % progress_every == 0 or index + 1 == len(queries)):
                    print(f"retrieval/{dataset.split}: {index + 1}/{len(queries)} queries", flush=True)
    finally:
        model.train(previous_training)
    offsets = np.concatenate(([0], np.cumsum([len(q["candidate_ids"]) for q in queries])))
    np.savez_compressed(
        output_dir / "scores.npz", query_ids=np.asarray([q["query_id"] for q in queries]),
        offsets=offsets, candidate_ids=np.asarray(candidate_arrays, dtype=np.int64),
        energy_mmd=np.asarray(scores, dtype=np.float32),
    )
    report = {
        "metadata": metadata,
        **summarize(ranks, bootstrap_replicates=bootstrap_replicates, seed=dataset.seed),
        "random_macro": {
            f"recall_at_{k}": float(np.mean([
                np.mean([min(k / r["candidate_count"], 1.0) for r in ranks if r["cell_type"] == cell_type])
                for cell_type in sorted({r["cell_type"] for r in ranks})
            ])) for k in (1, 5, 10, 20, 50)
        },
    }
    report["result_artifact_sha256"] = {
        name: sha256(output_dir / name) for name in ("scores.npz", "per_query.jsonl")
    }
    # Written last: its presence denotes complete coverage and finite scores.
    write_json(output_dir / "report.json", report)
    return report


def log_metrics(report, split):
    return {
        f"retrieval/{split}/{aggregation}/{name}": value
        for aggregation in ("macro", "request_equal", "coverage", "hit_counts")
        for name, value in report[aggregation].items()
    }


def validate_checkpoint_data(config, root, protocol, fold):
    from data.parse10m import split_manifest_path

    root = Path(root)
    contract = config.parse10m_contract
    if (contract["protocol"], contract["fold"]) != (protocol, fold):
        raise ValueError("Requested protocol/fold differs from the checkpoint training contract")
    for key, path in (
        ("source_manifest_sha256", root / "manifest.json"),
        ("genes_sha256", root / "genes.npy"),
        ("train_split_sha256", split_manifest_path(root, protocol=protocol, fold=fold, split="train")),
        ("validation_split_sha256", split_manifest_path(root, protocol=protocol, fold=fold, split="val")),
    ):
        if contract[key] != sha256(path):
            raise ValueError(f"Checkpoint data identity differs: {key}")


def main():
    import argparse
    import time
    import yaml
    from .model import Parse10MWorldModel
    from .train import DEFAULT_RECIPE

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_RECIPE)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--splits", choices=("val", "test"), nargs="+", default=["val", "test"])
    parser.add_argument("--candidate-scope", choices=("all_actions", "target_split"))
    parser.add_argument("--candidate-batch-size", type=int)
    parser.add_argument("--bootstrap-replicates", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--swanlab-mode", choices=("cloud", "offline", "local", "disabled"))
    args = parser.parse_args()
    recipe = yaml.safe_load(args.config.read_text())
    settings = recipe.get("retrieval", {})
    checkpoint = args.checkpoint.resolve()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("--output-dir must be new or empty")
    if len(set(args.splits)) != len(args.splits):
        parser.error("--splits must not contain duplicates")
    bootstrap_replicates = args.bootstrap_replicates if args.bootstrap_replicates is not None else settings.get("bootstrap_replicates", 4000)
    batch_size = (args.candidate_batch_size if args.candidate_batch_size is not None
                  else settings.get("candidate_batch_size", 32))
    if bootstrap_replicates < 0 or batch_size < 1:
        parser.error("Invalid bootstrap count or candidate batch size")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; request a GPU worker or explicitly use --device cpu")
    model, loading = Parse10MWorldModel.from_pretrained(checkpoint, output_loading_info=True)
    if any(loading.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"Checkpoint did not load exactly: {loading}")
    model.to(args.device).eval()
    data = recipe["data"]
    root = (args.config.resolve().parent / data["dataset_root"]).resolve()
    contract = model.config.parse10m_contract
    protocol, fold = contract["protocol"], contract["fold"]
    validate_checkpoint_data(model.config, root, protocol, fold)
    identity = {
        "path": str(checkpoint), "config_sha256": sha256(checkpoint / "config.json"),
        "weights_sha256": sha256(checkpoint / "model.safetensors"),
    }
    scope = args.candidate_scope or settings.get("candidate_scope", "all_actions")
    swan = recipe.get("swanlab", {})
    mode = args.swanlab_mode or (swan.get("mode", "cloud") if swan.get("enabled", True) else "disabled")
    run = None
    if mode != "disabled":
        import swanlab
        run = swanlab.init(
            project=swan.get("project", "perturbjepa-parse10m"), workspace=swan.get("workspace"),
            experiment_name=f"retrieval_{protocol}_{checkpoint.name}_{args.output_dir.name}",
            job_type="retrieval", mode=mode, logdir=str(args.output_dir / "swanlab"),
            config={"checkpoint": identity, "protocol": protocol, "fold": fold,
                    "candidate_scope": scope, "cells_per_group": settings.get("cells_per_group", 256),
                    "seed": settings.get("seed", 20260930), "precision": args.precision,
                    "bootstrap_replicates": bootstrap_replicates},
        )
    store = Parse10MStore(root)
    reports = {}
    try:
        for split in args.splits:
            started = time.monotonic()
            dataset = Parse10MDataset(
                root, protocol=protocol, fold=fold, split=split,
                cells_per_group=settings.get("cells_per_group", 256), seed=settings.get("seed", 20260930),
                store=store,
            )
            report = evaluate_retrieval(
                model, dataset, args.output_dir / split, candidate_scope=scope,
                candidate_batch_size=batch_size, precision=args.precision,
                bootstrap_replicates=bootstrap_replicates, checkpoint=identity,
            )
            reports[split] = report
            metrics = {**log_metrics(report, split), f"retrieval/{split}/runtime_seconds": time.monotonic() - started}
            print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
            if run:
                swanlab.log(metrics, step=0)
        write_json(args.output_dir / "summary.json", reports)
    finally:
        store.close()
        if run:
            swanlab.finish()


if __name__ == "__main__":
    main()
