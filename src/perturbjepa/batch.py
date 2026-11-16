"""Pack sampled Parse10M CSR rows for the Perturb gene-bag model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch

from data import PerturbationSample


class SparseBatchCollator:
    """Build one flat bank of sparse cell bags and two logical population maps."""

    def __init__(self, condition_to_id: Mapping[str, int]) -> None:
        self.condition_to_id = dict(condition_to_id)

    def __call__(self, samples: Sequence[PerturbationSample]) -> dict[str, torch.Tensor]:
        if not samples:
            raise ValueError("A sparse batch must contain at least one perturbation group")

        cells, genes = samples[0].treated.shape
        groups = len(samples)
        n_cells = 2 * groups * cells
        nnz = sum(sample.control.values.size + sample.treated.values.size for sample in samples)
        gene_ids = np.empty(nnz, dtype=np.int32)
        gene_values = np.empty(nnz, dtype=np.float32)
        cell_offsets = np.empty(n_cells + 1, dtype=np.int64)
        cell_offsets[0] = 0
        cell_value_sums = np.zeros(n_cells, dtype=np.float32)
        control_cell_map = np.empty((groups, cells), dtype=np.int32)
        target_cell_map = np.empty((groups, cells), dtype=np.int32)
        condition_ids = np.empty(groups, dtype=np.int64)

        cell_start = 0
        entry_start = 0
        for group, sample in enumerate(samples):
            condition_ids[group] = self.condition_to_id[sample.condition]
            arms = (
                (sample.control, sample.control_id, control_cell_map),
                (sample.treated, sample.treated_id, target_cell_map),
            )
            for arm, (block, block_id, population_map) in enumerate(arms):
                if block.shape != (cells, genes):
                    raise ValueError("Every control and target must have the same cell and gene dimensions")
                cell_stop = cell_start + cells
                entry_stop = entry_start + block.values.size
                gene_ids[entry_start:entry_stop] = block.indices
                gene_values[entry_start:entry_stop] = block.values
                cell_offsets[cell_start + 1:cell_stop + 1] = block.indptr[1:] + entry_start

                # Skip empty rows so reduceat sees each nonempty span once.
                nonempty = np.flatnonzero(np.diff(block.indptr))
                if nonempty.size:
                    cell_value_sums[cell_start + nonempty] = np.add.reduceat(
                        block.values, block.indptr[nonempty], dtype=np.float32
                    )

                # Keep CSR rows in I/O order and randomize logical slots so that
                # Rank-B's first slots are a deterministic subsample of each arm.
                rng = np.random.default_rng([block_id, arm, *block.rows.tolist()])
                population_map[group] = cell_start + rng.permutation(cells)
                cell_start = cell_stop
                entry_start = entry_stop

        return {
            "gene_ids": torch.from_numpy(gene_ids),
            "gene_values": torch.from_numpy(gene_values),
            "cell_offsets": torch.from_numpy(cell_offsets),
            "cell_value_sums": torch.from_numpy(cell_value_sums),
            "control_cell_map": torch.from_numpy(control_cell_map),
            "target_cell_map": torch.from_numpy(target_cell_map),
            "condition_ids": torch.from_numpy(condition_ids),
        }
