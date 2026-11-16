"""Train-only additive Ridge responses with explicit point-mass predictions.

This is a project-defined baseline, not a reproduction of an upstream model
named ``Linear``. Plate pseudobulks are averaged equally within each cell/action
pair. The objective is sum squared delta error plus alpha times the squared
cell/action coefficients; the intercept is not penalized. Validation chooses
one alpha by pair-equal delta MSE, without fitting on validation observations.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_LINEAR_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
LINEAR_SCHEMA_VERSION = "linear-additive/v1"


def _json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(",", ":")
    )


def _labels(values: Sequence[Any], name: str, *, unique: bool = False) -> tuple:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of labels")
    result = tuple("" if value is None else str(value) for value in values)
    if not result or not all(result) or (unique and len(set(result)) != len(result)):
        raise ValueError("{} must contain non-empty{} labels".format(
            name, " unique" if unique else ""
        ))
    return result


def _gene_indices(values: Any, width: int) -> np.ndarray:
    genes = np.asarray(values)
    if (
        genes.shape != (width,) or genes.dtype.kind not in "iu"
        or np.any(genes < 0) or len(np.unique(genes)) != width
    ):
        raise ValueError("gene_indices must contain one distinct nonnegative integer per feature")
    return genes.astype(np.int64)


@dataclass
class LinearProfileBank:
    """An auditable additive predictor over seen backgrounds and exact actions."""

    cell_ids: Sequence[str]
    condition_ids: Sequence[str]
    intercept: np.ndarray
    cell_effects: np.ndarray
    condition_effects: np.ndarray
    gene_indices: np.ndarray
    alpha: float
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        self.cell_ids = _labels(self.cell_ids, "cell_ids", unique=True)
        self.condition_ids = _labels(self.condition_ids, "condition_ids", unique=True)
        self.intercept = np.asarray(self.intercept, dtype=np.float64)
        self.cell_effects = np.asarray(self.cell_effects, dtype=np.float64)
        self.condition_effects = np.asarray(self.condition_effects, dtype=np.float64)
        if self.intercept.ndim != 1 or self.intercept.size == 0:
            raise ValueError("intercept must be a nonempty feature vector")
        width = len(self.intercept)
        if (
            self.cell_effects.shape != (len(self.cell_ids), width)
            or self.condition_effects.shape != (len(self.condition_ids), width)
            or not all(np.isfinite(value).all() for value in (
                self.intercept, self.cell_effects, self.condition_effects
            ))
        ):
            raise ValueError("Linear coefficient shapes or values are invalid")
        self.gene_indices = _gene_indices(self.gene_indices, width)
        self.alpha = float(self.alpha)
        if not np.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("alpha must be finite and strictly positive")
        self.metadata = json.loads(_json(dict(self.metadata)))
        self._cell_index = {value: i for i, value in enumerate(self.cell_ids)}
        self._condition_index = {value: i for i, value in enumerate(self.condition_ids)}

    @property
    def identity(self) -> str:
        digest = hashlib.sha256(_json({
            "schema_version": LINEAR_SCHEMA_VERSION,
            "cell_ids": self.cell_ids, "condition_ids": self.condition_ids,
            "alpha": self.alpha, "metadata": self.metadata,
        }).encode("utf-8"))
        for array, dtype in (
            (self.intercept, "<f8"), (self.cell_effects, "<f8"),
            (self.condition_effects, "<f8"), (self.gene_indices, "<i8"),
        ):
            digest.update(_json(list(array.shape)).encode("utf-8"))
            digest.update(np.ascontiguousarray(array, dtype=dtype).tobytes())
        return "sha256:" + digest.hexdigest()

    def predict(self, control_mean: Any, cell_id: str, condition_ids: Sequence[str]) -> np.ndarray:
        """Return one unclipped mean profile per requested condition, in input order."""
        control = np.asarray(control_mean, dtype=np.float64)
        if control.shape != self.intercept.shape or not np.isfinite(control).all():
            raise ValueError("control_mean must be a finite vector in the fitted gene order")
        try:
            cell_index = self._cell_index[str(cell_id)]
        except KeyError as exc:
            raise ValueError("cell_id is absent from the Linear training vocabulary") from exc
        conditions = _labels(condition_ids, "condition_ids")
        try:
            action_indices = [self._condition_index[value] for value in conditions]
        except KeyError as exc:
            raise ValueError("condition_id is absent from the Linear training vocabulary") from exc
        return (
            control[None, :] + self.intercept[None, :]
            + self.cell_effects[cell_index][None, :] + self.condition_effects[action_indices]
        )

    def write(self, path: Any) -> None:
        path = Path(path)
        if path.suffix != ".npz":
            raise ValueError("Linear cache must end in .npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        document = _json({
            "schema_version": LINEAR_SCHEMA_VERSION, "identity": self.identity,
            "alpha": self.alpha, "fit": self.metadata,
        })
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=path.parent, prefix=".linear-", delete=False
            ) as handle:
                temporary = Path(handle.name)
                np.savez_compressed(
                    handle, cell_ids=np.asarray(self.cell_ids),
                    condition_ids=np.asarray(self.condition_ids), intercept=self.intercept,
                    cell_effects=self.cell_effects, condition_effects=self.condition_effects,
                    gene_indices=self.gene_indices, metadata=np.asarray(document),
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @classmethod
    def read(cls, path: Any) -> LinearProfileBank:
        with np.load(path, allow_pickle=False) as payload:
            if set(payload.files) != {
                "cell_ids", "condition_ids", "intercept", "cell_effects",
                "condition_effects", "gene_indices", "metadata",
            }:
                raise ValueError("Linear cache has unexpected arrays")
            document = json.loads(str(payload["metadata"].item()))
            if document.get("schema_version") != LINEAR_SCHEMA_VERSION:
                raise ValueError("Unsupported Linear cache schema")
            bank = cls(
                cell_ids=tuple(payload["cell_ids"].tolist()),
                condition_ids=tuple(payload["condition_ids"].tolist()),
                intercept=payload["intercept"], cell_effects=payload["cell_effects"],
                condition_effects=payload["condition_effects"],
                gene_indices=payload["gene_indices"], alpha=document["alpha"],
                metadata=document["fit"],
            )
        if bank.identity != document.get("identity"):
            raise ValueError("Linear cache content hash mismatch")
        return bank


def _collapse_pairs(
    rows: Sequence[int], cells: tuple, conditions: tuple, plates: tuple,
    treated: np.ndarray, control: np.ndarray,
) -> tuple:
    """Average plate deltas, without consulting rows outside the requested split."""
    grouped = {}
    seen = set()
    for row in rows:
        key = (cells[row], conditions[row])
        physical_group = key + (plates[row],)
        if physical_group in seen:
            raise ValueError("duplicate cell/plate/condition pseudobulk group")
        seen.add(physical_group)
        grouped.setdefault(key, []).append(row)
    pairs = tuple(sorted(grouped))
    deltas = np.empty((len(pairs), treated.shape[1]), dtype=np.float64)
    for position, key in enumerate(pairs):
        selected = grouped[key]
        left = np.asarray(treated[selected], dtype=np.float64)
        right = np.asarray(control[selected], dtype=np.float64)
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            raise ValueError("train/validation pseudobulks must be finite")
        deltas[position] = (left - right).mean(axis=0)
    return pairs, deltas


def fit_linear_additive(
    *, cell_ids: Sequence[str], condition_ids: Sequence[str], plate_ids: Sequence[str],
    treated_means: Any, control_means: Any, splits: Sequence[str], gene_indices: Any,
    alphas: Sequence[float] = DEFAULT_LINEAR_ALPHAS, metadata: Mapping[str, Any] | None = None,
) -> LinearProfileBank:
    """Fit train-only Ridge; choose alpha using held-out validation delta MSE.

    Inputs are one row per cell/plate/action, not individual cells. Test/control
    values are not read beyond checking the overall array shapes. Counts are
    deliberately not weights: plates within a pair, then pairs, are equal.
    No validation rows are used to refit coefficients after selecting alpha.
    """
    cells = _labels(cell_ids, "cell_ids")
    conditions = _labels(condition_ids, "condition_ids")
    plates = _labels(plate_ids, "plate_ids")
    split_values = _labels(splits, "splits")
    if not len(cells) == len(conditions) == len(plates) == len(split_values):
        raise ValueError("pseudobulk metadata arrays must have equal lengths")
    if set(split_values) - {"train", "validation", "test", "control"}:
        raise ValueError("splits must be train, validation, test, or control")
    treated = np.asarray(treated_means)
    control = np.asarray(control_means)
    if (
        treated.ndim != 2 or treated.shape[0] != len(cells) or treated.shape[1] == 0
        or control.shape != treated.shape
    ):
        raise ValueError("treated/control means must be aligned [groups, genes] matrices")
    genes = _gene_indices(gene_indices, treated.shape[1])
    alpha_values = tuple(sorted({float(value) for value in alphas}))
    if not alpha_values or any(not np.isfinite(value) or value <= 0 for value in alpha_values):
        raise ValueError("alphas must be finite and strictly positive")
    train_rows = [i for i, split in enumerate(split_values) if split == "train"]
    val_rows = [i for i, split in enumerate(split_values) if split == "validation"]
    if not train_rows or not val_rows:
        raise ValueError("Linear fitting requires both train and validation groups")
    train_pairs, targets = _collapse_pairs(
        train_rows, cells, conditions, plates, treated, control
    )
    val_pairs, val_targets = _collapse_pairs(
        val_rows, cells, conditions, plates, treated, control
    )
    if set(train_pairs) & set(val_pairs):
        raise ValueError("train and validation must hold out complete cell/action pairs")
    cell_vocab = tuple(sorted({pair[0] for pair in train_pairs}))
    condition_vocab = tuple(sorted({pair[1] for pair in train_pairs}))
    cell_index = {value: i for i, value in enumerate(cell_vocab)}
    action_index = {value: i for i, value in enumerate(condition_vocab)}
    if any(c not in cell_index or a not in action_index for c, a in val_pairs):
        raise ValueError("validation cell/action is absent from the training vocabulary")
    n_cells = len(cell_vocab)
    width = n_cells + len(condition_vocab)
    ci = np.asarray([cell_index[c] for c, _ in train_pairs], dtype=np.int64)
    ai = np.asarray([n_cells + action_index[a] for _, a in train_pairs], dtype=np.int64)
    feature_counts = np.bincount(np.concatenate((ci, ai)), minlength=width).astype(np.float64)
    # Build sufficient statistics without a dense [50k pairs, 1k one-hot columns] design.
    gram = np.diag(feature_counts)
    np.add.at(gram, (ci, ai), 1.0)
    np.add.at(gram, (ai, ci), 1.0)
    rhs = np.zeros((width, targets.shape[1]), dtype=np.float64)
    np.add.at(rhs, ci, targets)
    np.add.at(rhs, ai, targets)
    target_mean = targets.mean(axis=0)
    feature_mean = feature_counts / len(train_pairs)
    gram -= np.outer(feature_counts, feature_mean)
    rhs -= feature_counts[:, None] * target_mean[None, :]
    # Centering eliminates the unpenalized intercept. One eigendecomposition is
    # reused for all positive penalties; tiny negative round-off eigenvalues are zero.
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    if float(eigenvalues.min()) < -1.0e-8 * max(1.0, float(feature_counts.max())):
        raise RuntimeError("centered Linear Gram matrix is not positive semidefinite")
    eigenvalues = np.maximum(eigenvalues, 0.0)
    projected_rhs = eigenvectors.T @ rhs
    vci = np.asarray([cell_index[c] for c, _ in val_pairs], dtype=np.int64)
    vai = np.asarray([n_cells + action_index[a] for _, a in val_pairs], dtype=np.int64)
    best = None
    trace = []
    for alpha in alpha_values:
        effects = eigenvectors @ (projected_rhs / (eigenvalues[:, None] + alpha))
        intercept = target_mean - feature_mean @ effects
        predicted = intercept[None, :] + effects[vci] + effects[vai]
        loss = float(np.square(predicted - val_targets).mean())
        if not np.isfinite(loss):
            raise ValueError("validation MSE is not finite")
        trace.append({"alpha": alpha, "validation_delta_mse": loss})
        if best is None or loss < best[0]:
            best = (loss, alpha, intercept.copy(), effects.copy())
    fit_metadata = dict(metadata or {})
    fit_metadata.update({
        "implementation": LINEAR_SCHEMA_VERSION, "fit_split": "train",
        "selection_split": "validation", "selection_metric": "pair_equal_delta_mse",
        "refit_on_validation": False, "aggregation": "plate_equal_then_cell_action_equal",
        "objective": "sum_squared_delta_error_plus_alpha_effect_l2",
        "intercept_penalized": False, "prediction_distribution": "point_mass",
        "prediction_clipping": "none", "n_train_groups": len(train_rows),
        "n_train_pairs": len(train_pairs), "n_validation_groups": len(val_rows),
        "n_validation_pairs": len(val_pairs), "validation_scores": trace,
        "selected_alpha": best[1], "selected_validation_delta_mse": best[0],
    })
    return LinearProfileBank(
        cell_ids=cell_vocab, condition_ids=condition_vocab, intercept=best[2],
        cell_effects=best[3][:n_cells], condition_effects=best[3][n_cells:],
        gene_indices=genes, alpha=best[1], metadata=fit_metadata,
    )




import argparse
import shutil
import time
from collections import Counter, defaultdict
import torch
import yaml
from data import Parse10MDataset, condition_labels
from perturbjepa.retrieval import energy_mmd, log_metrics, rank_metrics, sha256, summarize, write_json
from perturbjepa.train import DEFAULT_RECIPE, PROJECT_ROOT


class FullPopulationMeans:
    """Read independent per-well summaries of the unchanged source CSR pools."""

    def __init__(self, root):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        self.blocks = {block["id"]: block for block in self.manifest["blocks"]}
        self.counts = np.load(self.root / "_plan/well_counts_retained.npy", allow_pickle=False)
        self.well_conditions = np.load(self.root / "_plan/well_condition.npy", allow_pickle=False)
        self.donor = None
        self.statistics = None
        self.paths = set()
        self.controls = {}
        if (self.root / "dashboard/_SUCCESS").read_text().strip() != self.manifest["config_id"]:
            raise ValueError("Dashboard summaries do not match the source export")

    def close(self):
        if self.statistics is not None:
            self.statistics._mmap.close()
            self.statistics = None
            self.donor = None

    def mean(self, block_id):
        block = self.blocks[block_id]
        if block_id in self.controls:
            return self.controls[block_id]
        categories = self.manifest["categories"]
        donor = categories["donor"].index(block["donor"])
        cell_type = self.manifest["config"]["cell_types"].index(block["cell_type"])
        source_type = categories["cell_type"].index(block["cell_type"])
        condition = categories["cytokine"].index(block["condition"])
        if self.donor != donor:
            self.close()
            directory = self.root / "dashboard" / self.manifest["directory_names"]["donor"][donor]
            if (directory / "_SUCCESS").read_text().strip() != self.manifest["config_id"]:
                raise ValueError("A donor's mean cache is incomplete or stale")
            path = directory / "gene_stats.npy"
            self.paths.add(path)
            self.statistics = np.load(path, mmap_mode="r", allow_pickle=False)
            self.donor = donor
        wells = np.flatnonzero(self.well_conditions[donor] == condition)
        weights = self.counts[donor, wells, source_type].astype(np.float64)
        if int(weights.sum()) != block["n_cells"]:
            raise ValueError(f"Mean-cache population size differs from block {block_id}")
        expected = (block["shape"][1], len(self.manifest["config"]["cell_types"]),
                    len(categories["bc1_well"]), 2)
        if self.statistics.shape != expected:
            raise ValueError("Mean-cache axes differ from the source gene/type/well vocabulary")
        values = np.asarray(self.statistics[:, cell_type, :, 0][:, wells], dtype=np.float64)
        mean = values @ (weights / weights.sum())
        if not np.isfinite(mean).all():
            raise ValueError("Mean cache contains nonfinite values")
        if block["condition"] == self.manifest["config"]["control"]:
            self.controls[block_id] = mean
        return mean


def csr_mean(population):
    return np.bincount(population.indices, weights=population.values,
                       minlength=population.shape[1]) / population.shape[0]


def check_cached_means(provider, train, validation):
    """Spot-check full-cell means directly against CSR, including a PBS pool."""
    train_pairs = sorted(train._pairs, key=lambda pair: pair["n_cells"])
    validation_pair = min(validation._pairs, key=lambda pair: pair["n_cells"])
    controls = sorted({pair["control_id"] for pair in train._pairs},
                      key=lambda index: provider.blocks[index]["n_cells"])
    selected = [train_pairs[0]["treated_id"], train_pairs[len(train_pairs) // 2]["treated_id"],
                validation_pair["treated_id"], controls[0]]
    results = []
    for block_id in dict.fromkeys(selected):
        block = provider.blocks[block_id]
        directory = provider.root / block["path"]
        indices = np.load(directory / "indices.npy", mmap_mode="r", allow_pickle=False)
        values = np.load(directory / "values.npy", mmap_mode="r", allow_pickle=False)
        exact = np.bincount(indices, weights=values, minlength=block["shape"][1]) / block["n_cells"]
        cached = provider.mean(block_id)
        np.testing.assert_allclose(cached, exact, rtol=1e-6, atol=3e-7)
        results.append({"block_id": block_id, "cells": block["n_cells"],
                        "maximum_absolute_error": float(np.abs(cached - exact).max())})
        indices._mmap.close()
        values._mmap.close()
    return results


def fit_parse10m(train, validation, output_dir, *, alphas=DEFAULT_LINEAR_ALPHAS):
    if train.root != validation.root:
        raise ValueError("Train and validation must use the same source export")
    provider = FullPopulationMeans(train.root)
    grouped = defaultdict(list)
    totals = {}
    donor_counts = Counter()
    split_labels = []
    try:
        checks = check_cached_means(provider, train, validation)
        for dataset in (train, validation):
            for pair in sorted(dataset._pairs, key=lambda pair: (pair["donor"], pair["treated_id"])):
                key = (dataset.split, pair["cell_type"], pair["condition"])
                donor_key = key + (pair['donor'],)
                delta = provider.mean(pair["treated_id"]) - provider.mean(pair["control_id"])
                if donor_key not in totals:
                    totals[donor_key] = np.zeros_like(delta)
                totals[donor_key] += delta
                donor_counts[donor_key] += 1
                grouped[key].append(pair["treated_id"])
            print(f"linear/profiles: {dataset.split}, {len(dataset)} full-cell blocks complete", flush=True)
        by_pair = defaultdict(list)
        for donor_key, total in totals.items():
            by_pair[donor_key[:3]].append(total / donor_counts[donor_key])
        keys = sorted(by_pair)
        deltas = np.stack([np.mean(by_pair[key], axis=0) for key in keys])
        split_labels = ["validation" if key[0] == "val" else key[0] for key in keys]
        metadata = {
            "dataset": "Parse10M", "genes": f"all retained {deltas.shape[1]} genes; existing CP10K+log1p",
            "source_manifest_sha256": sha256(train.root / "manifest.json"),
            "genes_sha256": sha256(train.root / "genes.npy"),
            "train_split_sha256": sha256(train.split_path),
            "validation_split_sha256": sha256(validation.split_path),
            "mean_source": "independent full-cell per-well float32 dashboard cache",
            "control_pooling": "cell-count weighted across all matched PBS wells",
            "donor_aggregation": "equal donors within each factor/cell-type pair before fitting",
            "original_train_blocks": len(train), "original_validation_blocks": len(validation),
            "test_blocks_used_for_fit_or_selection": 0,
            "csr_mean_checks": checks,
            "mean_cache_sha256": {str(path): sha256(path) for path in sorted(provider.paths)},
            "counts_sha256": sha256(train.root / "_plan/well_counts_retained.npy"),
            "well_conditions_sha256": sha256(train.root / "_plan/well_condition.npy"),
            "linear_code_sha256": sha256(Path(__file__)),
            "runner_code_sha256": sha256(Path(__file__)),
        }
        # Equal-donor collapsing is identical to reference plate-equal collapsing.
        bank = fit_linear_additive(
            cell_ids=[key[1] for key in keys], condition_ids=[key[2] for key in keys],
            plate_ids=["donor-equal-aggregate"] * len(keys), splits=split_labels,
            treated_means=deltas, control_means=np.zeros_like(deltas),
            gene_indices=np.arange(deltas.shape[1]), alphas=alphas, metadata=metadata,
        )
        bank.write(Path(output_dir) / "model.npz")
        write_json(Path(output_dir) / "fit.json", bank.metadata)
        return bank
    finally:
        provider.close()


@torch.inference_mode()
def evaluate_linear(bank, dataset, query_dir, output_dir, *, device="cuda", bootstrap_replicates=4000):
    """Use exactly the JEPA query cells and full candidate catalogs, in expression space."""
    query_dir, output_dir = Path(query_dir), Path(output_dir)
    metadata = json.loads((query_dir / "manifest.json").read_text())
    if metadata["split"] != dataset.split or metadata["candidate_scope"] != "all_actions":
        raise ValueError("Linear requires a matching frozen all-actions query manifest")
    for name, expected in metadata["artifact_sha256"].items():
        if sha256(query_dir / name) != expected:
            raise ValueError(f"Frozen query artifact changed: {name}")
    if metadata["source_manifest_sha256"] != sha256(dataset.root / "manifest.json"):
        raise ValueError("Query/source identity mismatch")
    if metadata["split_sha256"] != sha256(dataset.split_path):
        raise ValueError("Frozen queries use a different split")
    if metadata["genes_sha256"] != bank.metadata["genes_sha256"]:
        raise ValueError("Linear and JEPA query gene vocabularies differ")
    labels = metadata["condition_labels"]
    if set(labels) != set(bank.condition_ids):
        raise ValueError("Linear training does not cover every candidate factor")
    if len(bank.gene_indices) != len(np.load(dataset.root / "genes.npy", allow_pickle=False)):
        raise ValueError("Linear scoring must use the complete source gene vocabulary")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    for name in ("queries.jsonl", "truth.jsonl", "cells.npz"):
        shutil.copyfile(query_dir / name, output_dir / name)
    queries = [json.loads(line) for line in (query_dir / "queries.jsonl").read_text().splitlines()]
    truth = [json.loads(line) for line in (query_dir / "truth.jsonl").read_text().splitlines()]
    if len(queries) != len(dataset) or len(truth) != len(queries):
        raise ValueError("Frozen query coverage differs from the original split")
    cells = np.load(query_dir / "cells.npz", allow_pickle=False)
    metadata = {
        **metadata, "format": "parse10m-retrieval-v1",
        "model": "Linear-additive Ridge", "model_identity": bank.identity,
        "alpha": bank.alpha, "representation": "CP10K+log1p expression, all retained genes",
        "prediction_distribution": "point_mass", "precision": "fp32",
        "reference_query_manifest_sha256": sha256(query_dir / "manifest.json"),
        "evaluation_code_sha256": sha256(Path(__file__)), "checkpoint": None,
        "bootstrap_replicates": bootstrap_replicates,
    }
    write_json(output_dir / "manifest.json", metadata)
    scores, rows, candidate_ids, previous_control = [], [], [], None
    try:
        with (output_dir / "per_query.jsonl").open("w") as handle:
            for index, query in enumerate(queries):
                if query["query_id"] != truth[index]["query_id"]:
                    raise ValueError("Truth/query order differs")
                key = (query["control_id"], tuple(query["candidate_ids"]))
                if key != previous_control:
                    population = dataset.read_rows(query["control_id"], cells["control_rows"][index])
                    profiles = bank.predict(csr_mean(population), query["cell_type"],
                                            [labels[i] for i in query["candidate_ids"]])
                    prediction = torch.as_tensor(profiles[:, None, :], dtype=torch.float32, device=device)
                    previous_control = key
                population = dataset.read_rows(query["treated_id"], cells["target_rows"][index])
                # Materialize only the fixed anonymous target population, including zeros.
                dense = np.zeros(population.shape, dtype=np.float32)
                row_ids = np.repeat(np.arange(population.shape[0]), np.diff(population.indptr))
                np.add.at(dense, (row_ids, population.indices), population.values)
                target = torch.as_tensor(dense, device=device)
                distances = energy_mmd(prediction, target).cpu().numpy()
                rank, order, metrics = rank_metrics(distances, query["candidate_ids"], truth[index]["condition_id"])
                row = {
                    **query, "condition": truth[index]["condition"], "rank": rank,
                    "candidate_count": len(distances), "top_condition": labels[query["candidate_ids"][order[0]]],
                    "true_distance": float(distances[query["candidate_ids"].index(truth[index]["condition_id"])]),
                    **metrics,
                }
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                rows.append(row)
                scores.extend(distances.tolist())
                candidate_ids.extend(query["candidate_ids"])
                if (index + 1) % 50 == 0 or index + 1 == len(queries):
                    print(f"linear/{dataset.split}: {index + 1}/{len(queries)} queries", flush=True)
    finally:
        cells.close()
    np.savez_compressed(
        output_dir / "scores.npz", query_ids=np.array([q["query_id"] for q in queries]),
        offsets=np.concatenate(([0], np.cumsum([len(q["candidate_ids"]) for q in queries]))),
        candidate_ids=np.asarray(candidate_ids, dtype=np.int64), energy_mmd=np.asarray(scores, dtype=np.float32),
    )
    report = {
        "metadata": metadata,
        **summarize(rows, bootstrap_replicates=bootstrap_replicates, seed=dataset.seed),
        "result_artifact_sha256": {
            name: sha256(output_dir / name) for name in ("scores.npz", "per_query.jsonl")
        },
    }
    write_json(output_dir / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_RECIPE)
    parser.add_argument("--query-dir", type=Path, required=True, help="Directory containing local val/test frozen retrieval queries.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--check", action="store_true", help="Validate splits and frozen queries without fitting or using CUDA.")
    parser.add_argument("--swanlab-mode", choices=("cloud", "offline", "local", "disabled"))
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("--output-dir must be new or empty")
    if not args.check and args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    # Distance arithmetic matches the reference FP32 scorer without TF32 shortcuts.
    torch.backends.cuda.matmul.allow_tf32 = False
    recipe = yaml.safe_load(args.config.read_text())
    data, swan = recipe["data"], recipe["swanlab"]
    root = (args.config.resolve().parent / data["dataset_root"]).resolve()
    protocol, fold = data["protocol"], data.get("fold")
    if protocol != "drug_cell_type":
        parser.error("This additive baseline requires the existing factor/cell-type combination holdout")
    settings = recipe.get("retrieval", {})
    options = {"root": root, "protocol": protocol, "fold": fold,
               "cells_per_group": settings.get("cells_per_group", 256), "seed": settings.get("seed", 20260930)}
    train = Parse10MDataset(**options, split="train")
    options["store"] = train.store
    validation = Parse10MDataset(**options, split="val")
    if args.check:
        labels = condition_labels(json.loads((root / 'manifest.json').read_text()))
        if set(labels) != set(train.conditions):
            raise ValueError('Every retrieval candidate must appear in training')
        for split in ('val', 'test'):
            dataset = Parse10MDataset(**options, split=split)
            metadata = json.loads((args.query_dir / split / 'manifest.json').read_text())
            if metadata['condition_labels'] != labels or metadata['split_sha256'] != sha256(dataset.split_path):
                raise ValueError('Frozen query vocabulary or split differs')
            if metadata['source_manifest_sha256'] != sha256(root / 'manifest.json'):
                raise ValueError('Frozen query source differs')
            if metadata['genes_sha256'] != sha256(root / 'genes.npy'):
                raise ValueError('Frozen query gene order differs')
            for name, expected in metadata['artifact_sha256'].items():
                if sha256(args.query_dir / split / name) != expected:
                    raise ValueError(f'Frozen query artifact changed: {name}')
        print(json.dumps({'model': 'Linear-additive Ridge', 'train_groups': len(train),
                          'validation_groups': len(validation), 'conditions': len(labels),
                          'output_dir': str(args.output_dir)}, indent=2))
        return
    run, mode = None, args.swanlab_mode or (swan.get("mode", "cloud") if swan.get("enabled", True) else "disabled")
    if mode != "disabled":
        import swanlab
        run = swanlab.init(
            project=swan.get("project", "perturbjepa-parse10m"), workspace=swan.get("workspace"),
            experiment_name=f"linear_{protocol}_{args.output_dir.name}", job_type="linear-retrieval",
            mode=mode, logdir=str(args.output_dir / "swanlab"),
            config={"model": "Linear-additive Ridge", "protocol": protocol,
                    "alphas": list(DEFAULT_LINEAR_ALPHAS), "selection": "validation pair-equal delta MSE",
                    "query_dir": str(args.query_dir.resolve()), "genes": "all retained genes",
                    "candidate_scope": "all_actions", "seed": options["seed"]},
        )
    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        bank = fit_parse10m(train, validation, args.output_dir)
        fit_metrics = {"linear/selected_alpha": bank.alpha,
                       "linear/validation_delta_mse": bank.metadata["selected_validation_delta_mse"],
                       "linear/fit_seconds": time.monotonic() - started}
        print(json.dumps(fit_metrics), flush=True)
        if run:
            swanlab.log(fit_metrics, step=0)
        reports = {}
        for split in ("val", "test"):
            dataset = Parse10MDataset(**options, split=split)
            report = evaluate_linear(
                bank, dataset, args.query_dir / split, args.output_dir / split, device=args.device,
                bootstrap_replicates=settings.get("bootstrap_replicates", 4000),
            )
            reports[split] = report
            print(json.dumps(log_metrics(report, split), indent=2), flush=True)
            if run:
                swanlab.log(log_metrics(report, split), step=1)
        write_json(args.output_dir / "summary.json", reports)
    finally:
        train.close()
        if run:
            swanlab.finish()


if __name__ == "__main__":
    main()
