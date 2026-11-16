"""Read paired Parse10M v1 cell sets from the published split manifests."""

import json
import operator
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np


DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "data/parse10m/v1"


def condition_labels(manifest):
    """Return the sorted Parse10M cytokine catalog without the PBS control."""
    return sorted(set(manifest['categories']['cytokine']) - {manifest['config']['control']})


@dataclass(frozen=True)
class CSRRows:
    values: np.ndarray
    indices: np.ndarray
    indptr: np.ndarray
    shape: Tuple[int, int]
    rows: np.ndarray


@dataclass(frozen=True)
class PerturbationSample:
    donor: str
    cell_type: str
    condition: str
    control_id: int
    treated_id: int
    control: CSRRows
    treated: CSRRows


def _close_arrays(arrays):
    for array in arrays:
        array._mmap.close()


def _open_arrays(directory):
    arrays = []
    try:
        for name in ("indptr", "indices", "values"):
            arrays.append(np.load(directory / (name + ".npy"), mmap_mode="r", allow_pickle=False))
    except BaseException:
        _close_arrays(arrays)
        raise
    return tuple(arrays)


def _read_rows(directory, rows, n_genes):
    """Read standalone CSR rows and release their temporary file mappings."""
    arrays = _open_arrays(directory)
    try:
        return _read_mapped_rows(arrays, rows, n_genes)
    finally:
        _close_arrays(arrays)


def _read_mapped_rows(arrays, rows, n_genes):
    source_indptr, source_indices, source_values = arrays
    starts = source_indptr[rows]
    stops = source_indptr[rows + 1]

    indptr = np.empty(len(rows) + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(stops - starts, out=indptr[1:])
    indices = np.empty(indptr[-1], dtype=source_indices.dtype)
    values = np.empty(indptr[-1], dtype=source_values.dtype)

    # Adjacent sampled rows are contiguous in the source CSR arrays.
    cuts = np.concatenate(([0], np.flatnonzero(np.diff(rows) != 1) + 1, [len(rows)]))
    for left, right in zip(cuts[:-1], cuts[1:]):
        source = slice(starts[left], stops[right - 1])
        target = slice(indptr[left], indptr[right])
        indices[target] = source_indices[source]
        values[target] = source_values[source]

    return CSRRows(values, indices, indptr, (len(rows), n_genes), rows)


def split_manifest_path(root, *, protocol, split, fold=None):
    directory = Path(root) / "splits" / protocol
    if protocol == "cell_type":
        if fold is None or not 1 <= operator.index(fold) <= 7:
            raise ValueError("cell_type requires an integer fold between 1 and 7")
        directory /= "fold_{:02d}".format(fold)
    elif protocol not in {"drug_cell_type", "drug"} or fold is not None:
        raise ValueError("use drug_cell_type or drug without a fold, or cell_type with a fold")
    return directory / (split + ".jsonl")


class Parse10MStore:
    """Own one run's full read-only CSR mappings, shared by all split views."""

    def __init__(self, root=DEFAULT_ROOT):
        self.root = Path(root).resolve()
        self._mapped_blocks = {}
        self._closed = False
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if (self.root / "export.SUCCESS").read_text().strip() != self.manifest["config_id"]:
            raise ValueError("Parse10M v1 export marker and manifest do not match")
        self.blocks = {block["id"]: block for block in self.manifest["blocks"]}
        try:
            for block_id, block in sorted(self.blocks.items()):
                self._mapped_blocks[block_id] = _open_arrays(self.root / block["path"])
        except BaseException:
            self.close()
            raise

    def read_rows(self, block_id, rows):
        if self._closed:
            raise RuntimeError("Parse10MStore is closed")
        return _read_mapped_rows(self._mapped_blocks[block_id], rows, self.blocks[block_id]["shape"][1])

    def close(self):
        """Unmap this process's view; returned samples own their copied arrays."""
        for arrays in self._mapped_blocks.values():
            _close_arrays(arrays)
        self._mapped_blocks.clear()
        self._closed = True

    def __enter__(self):
        if self._closed:
            raise RuntimeError("Parse10MStore is closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __getstate__(self):
        raise TypeError("Parse10MStore requires fork workers to inherit the run's existing mappings")

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class Parse10MDataset:
    """Read one split through the full CSR mappings created at run startup."""

    def __init__(
        self, root=DEFAULT_ROOT, *, protocol="drug_cell_type", split="train",
        fold=None, cells_per_group=256, seed=0, store=None,
    ):
        self._closed = False
        self._owns_store = store is None
        self.root = Path(root).resolve()
        self.split = split
        self.cells_per_group = operator.index(cells_per_group)
        self.seed = operator.index(seed)
        if self.cells_per_group < 1 or self.seed < 0:
            raise ValueError("cells_per_group must be positive and seed must be nonnegative")
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")

        self.split_path = split_manifest_path(self.root, protocol=protocol, split=split, fold=fold)
        if store is not None and (store.root != self.root or store._closed):
            raise ValueError("The shared Parse10MStore must be open and use the same data root")
        self.store = Parse10MStore(self.root) if store is None else store
        manifest = self.store.manifest
        self._blocks = self.store.blocks
        source_pairs = {row["treated_id"]: row for row in manifest["pairs"]}
        with self.split_path.open(encoding="utf-8") as handle:
            self._pairs = [json.loads(line) for line in handle]

        seen = set()
        for pair in self._pairs:
            treated_id = pair["treated_id"]
            if pair != source_pairs.get(treated_id) or treated_id in seen:
                raise ValueError("invalid or duplicate treated_id {} in {}".format(treated_id, self.split_path))
            seen.add(treated_id)
            control = self._blocks[pair["control_id"]]
            if (
                control["condition"] != manifest["config"]["control"]
                or control["donor"] != pair["donor"]
                or control["cell_type"] != pair["cell_type"]
            ):
                raise ValueError("mismatched PBS pool for treated_id {}".format(treated_id))
            if min(pair["n_cells"], pair["n_control"]) < self.cells_per_group:
                raise ValueError("not enough cells for treated_id {}".format(treated_id))

    def read_rows(self, block_id, rows):
        """Read fixed block-local rows through the run's existing mappings."""
        if self._closed:
            raise RuntimeError("Parse10MDataset is closed")
        return self.store.read_rows(block_id, rows)

    def close(self):
        """Close this view, releasing the store only if this dataset owns it."""
        if self._owns_store:
            self.store.close()
        self._closed = True

    def __enter__(self):
        if self._closed:
            raise RuntimeError("Parse10MDataset is closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    @property
    def conditions(self):
        """Return the perturbation labels present in this split."""
        return frozenset(pair["condition"] for pair in self._pairs)

    def __len__(self):
        return len(self._pairs)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            epoch, index = key
            return self.sample(index, epoch=epoch)
        return self.sample(key)

    def sample(self, index, *, epoch=0):
        """Read one pair. Evaluation uses fixed rows for each block."""
        epoch = operator.index(epoch)
        if epoch < 0:
            raise ValueError("epoch must be nonnegative")

        pair = self._pairs[index]
        control_id = pair["control_id"]
        treated_id = pair["treated_id"]
        if self.split == "train":
            control_rng_key = treated_id
        else:
            epoch = 0
            control_rng_key = control_id
        control = self._sample_block(control_id, [self.seed, epoch, control_rng_key, 0])
        treated = self._sample_block(treated_id, [self.seed, epoch, treated_id, 1])
        return PerturbationSample(
            pair["donor"], pair["cell_type"], pair["condition"],
            control_id, treated_id, control, treated,
        )

    def _sample_block(self, block_id, rng_seed):
        block = self._blocks[block_id]
        rng = np.random.default_rng(rng_seed)
        rows = np.sort(rng.choice(block["n_cells"], self.cells_per_group, replace=False))
        return self.read_rows(block_id, rows)


class EpochSampler:
    """Pass epoch through the index so persistent workers can resample."""

    def __init__(self, dataset):
        self.dataset = dataset
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = operator.index(epoch)
        if self.epoch < 0:
            raise ValueError("epoch must be nonnegative")

    def __iter__(self):
        indices = range(len(self.dataset))
        if self.dataset.split == "train":
            rng = np.random.default_rng([self.dataset.seed, self.epoch, 2])
            indices = rng.permutation(len(self.dataset)).tolist()
        return ((self.epoch, index) for index in indices)

    def __len__(self):
        return len(self.dataset)
