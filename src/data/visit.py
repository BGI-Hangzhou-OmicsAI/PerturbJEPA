"""Continuous visit-decayed sampling over an existing Parse10MStore.

The parent process owns all sampling state. Fork workers receive exact CSR
row requests; an ordered journal retains plans until training acknowledges them.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from collections import Counter, OrderedDict
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np

from ._visit_sampling import (
    InfiniteGroupSampler, UINT32_MAX, UINT64_MAX,
    _nonnegative_integer, _positive_float, _positive_integer,
    _positive_sizes, _probability, _validated_rng_state,
)
from .parse10m import PerturbationSample


PLAN_ID = "_sampling_plan_id"


@dataclass(frozen=True)
class VisitSamplingConfig:
    group_gamma: float = 1.0
    group_uniform_probability: float = 0.02
    group_deadline_multiplier: float | None = 10.0
    cell_gamma: float = 1.0
    cell_uniform_probability: float = 0.01

    def __post_init__(self):
        _positive_float("group_gamma", self.group_gamma)
        _positive_float("cell_gamma", self.cell_gamma)
        _probability("group_uniform_probability", self.group_uniform_probability)
        _probability("cell_uniform_probability", self.cell_uniform_probability)
        if self.group_deadline_multiplier is not None:
            _positive_float("group_deadline_multiplier", self.group_deadline_multiplier)


def visit_sampling_config(data, training):
    """Validate the opt-in recipe before opening the full expression export."""
    settings = data.get("train_sampling")
    if settings is None or settings == {"mode": "epoch"}:
        return None
    if not isinstance(settings, dict) or settings.get("mode") != "visit_decay":
        raise ValueError("data.train_sampling.mode must be visit_decay or epoch")
    unknown = set(settings) - {"mode", *(field.name for field in fields(VisitSamplingConfig))}
    if unknown:
        raise ValueError(f"Unknown visit sampling settings: {sorted(unknown)}")
    config = VisitSamplingConfig(**{key: value for key, value in settings.items() if key != "mode"})
    _positive_integer("max_steps", training.get("max_steps"))
    if "num_train_epochs" in training:
        raise ValueError("visit_decay has no epochs; remove num_train_epochs and set max_steps")
    for prefix in ("logging", "eval", "save"):
        if training.get(prefix + "_strategy") != "steps":
            raise ValueError(f"visit_decay requires {prefix}_strategy=steps")
        _positive_integer(prefix + "_steps", training.get(prefix + "_steps"))
    if training.get("auto_find_batch_size", False) or training.get("save_only_model", False):
        raise ValueError("visit_decay requires a fixed batch size and complete optimizer checkpoints")
    return config


class _CellVisits:
    """Packed exact counts and fixed-weight sampling without replacement."""

    def __init__(self, sizes, *, cells, gamma, epsilon, seed):
        self.sizes = _positive_sizes("pool sizes", sizes)
        self.cells = _positive_integer("cells", cells)
        if np.any(self.sizes < cells):
            raise ValueError("Every pool must contain at least cells_per_group cells")
        self.gamma, self.epsilon = gamma, epsilon
        self.offsets = np.r_[0, self.sizes.cumsum(dtype=np.int64)]
        self.delta = np.zeros(int(self.offsets[-1]), dtype=np.uint32)
        self.base = np.zeros(len(self.sizes), dtype=np.uint64)
        self.rng = np.random.default_rng(seed)
        self.covered = 0

    def _slice(self, pool):
        return slice(int(self.offsets[pool]), int(self.offsets[pool + 1]))

    def take(self, pool):
        # Lazy import keeps Numba/LLVM out of the epoch loader's startup path.
        from ._visit_kernel import draw_without_replacement

        counts = self.delta[self._slice(pool)]
        weights = np.power(1.0 + counts.astype(np.float64), -self.gamma)
        if np.any(~np.isfinite(weights)) or np.any(weights <= 0):
            raise OverflowError("Cell visit weights are no longer representable")
        selected = draw_without_replacement(weights, self.cells, self.epsilon, self.rng)
        if np.any(counts[selected] == UINT32_MAX):
            raise OverflowError("Relative cell visit count reached uint32 capacity")
        if self.base[pool] == 0:
            self.covered += int(np.count_nonzero(counts[selected] == 0))
        counts[selected] += np.uint32(1)
        minimum = int(counts.min())
        if int(self.base[pool]) > UINT64_MAX - minimum:
            raise OverflowError("Cell visit base reached uint64 capacity")
        if minimum:
            counts -= np.uint32(minimum)
            self.base[pool] += np.uint64(minimum)
        return np.sort(selected)

    def state_dict(self):
        return {"delta": self.delta.copy(), "base": self.base.copy(),
                "rng_state": copy.deepcopy(self.rng.bit_generator.state)}

    def load_state_dict(self, state):
        if not isinstance(state, dict):
            raise ValueError("Cell state must be a mapping")
        delta, base = np.asarray(state.get("delta")), np.asarray(state.get("base"))
        for name, value, expected, maximum in (("delta", delta, self.delta, UINT32_MAX),
                                               ("base", base, self.base, UINT64_MAX)):
            if value.shape != expected.shape or value.dtype.kind not in "iu":
                raise ValueError(f"Invalid cell {name}")
            if value.size and (int(value.min()) < 0 or int(value.max()) > maximum):
                raise ValueError(f"Cell {name} is outside its integer range")
        delta, base = delta.astype(np.uint32, copy=True), base.astype(np.uint64, copy=True)
        covered = 0
        for pool in range(len(self.sizes)):
            counts = delta[self._slice(pool)]
            if int(counts.min()) != 0:
                raise ValueError("Cell counts are not canonically rebased")
            if int(base[pool]) > UINT64_MAX - int(counts.max()):
                raise ValueError("Exact cell visits exceed uint64 capacity")
            covered += len(counts) if base[pool] else int(np.count_nonzero(counts))
        rng = _validated_rng_state(state.get("rng_state"))
        self.delta, self.base, self.covered = delta, base, covered
        self.rng.bit_generator.state = rng

    def total(self, pool):
        counts = self.delta[self._slice(pool)]
        return sum(map(int, counts)) + int(self.base[pool]) * len(counts)

    def consumed_coverage(self, pending):
        covered = self.covered
        for pool, rows in pending.items():
            ids, multiplicity = np.unique(np.concatenate(rows), return_counts=True)
            visits = self.delta[self._slice(pool)][ids].astype(np.uint64) + self.base[pool]
            covered -= int(np.count_nonzero(visits == multiplicity))
        return covered / int(self.offsets[-1])

    def quantiles(self, pending):
        visits = self.delta.astype(np.uint64)
        for pool, base in enumerate(self.base):
            if base:
                visits[self._slice(pool)] += base
        for pool, rows in pending.items():
            ids, multiplicity = np.unique(np.concatenate(rows), return_counts=True)
            visits[self.offsets[pool] + ids] -= multiplicity.astype(np.uint64)
        return np.quantile(visits, [.1, .5, .9]).tolist()


@dataclass(frozen=True)
class PopulationRequest:
    sequence_id: int
    treated_id: int
    control_id: int
    target_rows: np.ndarray
    control_rows: np.ndarray
    forced: bool = False


class _RequestDataset:
    def __init__(self, dataset, pairs):
        self.dataset = dataset
        self.pairs = {pair["treated_id"]: pair for pair in pairs}

    def __getitem__(self, request):
        started = time.perf_counter()
        pair = self.pairs[request.treated_id]
        sample = PerturbationSample(
            pair["donor"], pair["cell_type"], pair["condition"], request.control_id, request.treated_id,
            self.dataset.read_rows(request.control_id, request.control_rows),
            self.dataset.read_rows(request.treated_id, request.target_rows),
        )
        return request.sequence_id, sample, time.perf_counter() - started


class _PlanCollator:
    def __init__(self, collator):
        self.collator = collator

    def __call__(self, items):
        plan_id = items[0][0]
        if any(item[0] != plan_id for item in items):
            raise RuntimeError("A batch contains more than one sampling plan")
        return {**self.collator([item[1] for item in items]), PLAN_ID: plan_id,
                "_sampling_materialize_seconds": sum(item[2] for item in items)}


class _PlanSampler:
    def __init__(self, loader):
        self.loader = loader

    def __iter__(self):
        # Replaying a journal never advances any sampling counts or RNG.
        yield from list(self.loader._pending.values())
        while True:
            yield self.loader._plan()


class Parse10MVisitLoader:
    """Unbounded batch iterator with exact acknowledgement and resume semantics."""

    def __init__(self, dataset, *, collator, sampling=None, batch_size=15,
                 num_workers=4, prefetch_factor=2, pin_memory=False,
                 consumer_rank=0, consumer_count=1):
        if dataset.split != "train":
            raise ValueError("Visit sampling only accepts the training split")
        self.dataset, self.collator = dataset, collator
        self.sampling = sampling or VisitSamplingConfig()
        self.batch_size = _positive_integer("batch_size", batch_size)
        self.num_workers = _nonnegative_integer("num_workers", num_workers)
        self.prefetch_factor = _positive_integer("prefetch_factor", prefetch_factor)
        self.pin_memory = bool(pin_memory)
        self.rank = _nonnegative_integer("consumer_rank", consumer_rank)
        self.world_size = _positive_integer("consumer_count", consumer_count)
        if self.rank >= self.world_size:
            raise ValueError("consumer_rank must be smaller than consumer_count")
        all_pairs = sorted(dataset._pairs, key=lambda pair: pair["treated_id"])
        loads = [0] * self.world_size
        owners = {}
        for pair in sorted(all_pairs, key=lambda pair: (-pair["n_cells"], pair["treated_id"])):
            owner = min(range(self.world_size), key=lambda rank: (loads[rank], rank))
            owners[pair["treated_id"]] = owner
            loads[owner] += pair["n_cells"]
        self.pairs = [pair for pair in all_pairs if owners[pair["treated_id"]] == self.rank]
        if len(self.pairs) < self.batch_size:
            raise ValueError("Each rank must own at least batch_size distinct training groups")
        self._pair_index = {pair["treated_id"]: index for index, pair in enumerate(self.pairs)}
        self.control_ids = sorted({pair["control_id"] for pair in self.pairs})
        self._control_index = {key: index for index, key in enumerate(self.control_ids)}
        self._group, self._target, self._control = self._new_samplers()
        identity = {
            "version": 1, "sampling": asdict(self.sampling), "batch_size": self.batch_size,
            "cells_per_group": dataset.cells_per_group, "seed": dataset.seed,
            "rank": self.rank, "world_size": self.world_size,
            "ownership": sorted(owners.items()), "pairs": self.pairs,
            "conditions": sorted(collator.condition_to_id.items()),
            "source_hashes": {name: hashlib.sha256((dataset.root / name).read_bytes()).hexdigest()
                              for name in ("manifest.json", "genes.npy")},
            "split_sha256": hashlib.sha256(dataset.split_path.read_bytes()).hexdigest(),
            "code_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                            for name in ("visit.py", "_visit_sampling.py", "_visit_kernel.py")},
        }
        self.signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        self._pending = OrderedDict()
        self._delivered = set()
        self._next_sequence = self._consumed = 0
        self._last_consumed = np.full(len(self.pairs), -1, dtype=np.int64)
        self._forced_consumed = 0
        self._planning_seconds = self._wait_seconds = self._materialize_seconds = 0.0
        self._iterator = self._dataloader = None
        self._closed = self._failed = False

    def _seed(self, stream):
        return np.random.SeedSequence([self.dataset.seed, self.rank, stream]).generate_state(1)[0]

    def _new_samplers(self):
        config = self.sampling
        group_sizes = [pair["n_cells"] for pair in self.pairs]
        group = InfiniteGroupSampler(
            group_sizes, self.batch_size, cells_per_group=self.dataset.cells_per_group,
            gamma=config.group_gamma, uniform_probability=config.group_uniform_probability,
            deadline_multiplier=config.group_deadline_multiplier, seed=int(self._seed(0)),
        )
        target = _CellVisits(group_sizes, cells=self.dataset.cells_per_group, gamma=config.cell_gamma,
                             epsilon=config.cell_uniform_probability, seed=self._seed(1))
        control = _CellVisits([self.dataset._blocks[key]["n_cells"] for key in self.control_ids],
                              cells=self.dataset.cells_per_group, gamma=config.cell_gamma,
                              epsilon=config.cell_uniform_probability, seed=self._seed(2))
        return group, target, control

    def _plan(self):
        if self._failed or self._closed:
            raise RuntimeError("The visit loader has stopped")
        requests = []
        started = time.perf_counter()
        due = self._group.next_due
        forced = 0 if due is None else min(self.batch_size, int(np.count_nonzero(due <= self._group.take_step)))
        try:
            for slot, raw_index in enumerate(self._group.take()):
                index = int(raw_index)
                pair = self.pairs[index]
                target = self._target.take(index)
                control = self._control.take(self._control_index[pair["control_id"]])
                target.flags.writeable = control.flags.writeable = False
                requests.append(PopulationRequest(self._next_sequence, pair["treated_id"],
                                                  pair["control_id"], target, control, slot < forced))
        except BaseException:
            self._failed = True
            raise
        requests = tuple(requests)
        self._pending[self._next_sequence] = requests
        self._next_sequence += 1
        self._planning_seconds += time.perf_counter() - started
        return requests

    def __iter__(self):
        return self

    def __next__(self):
        if self._closed or self._failed:
            raise RuntimeError("The visit loader has stopped")
        if self._iterator is None:
            import torch
            from torch.utils.data import DataLoader

            options = {"num_workers": self.num_workers, "pin_memory": self.pin_memory,
                       "generator": torch.Generator().manual_seed(int(self._seed(3))), "in_order": True}
            if self.num_workers:
                options.update(multiprocessing_context="fork", persistent_workers=True,
                               prefetch_factor=self.prefetch_factor)
            self._dataloader = DataLoader(_RequestDataset(self.dataset, self.pairs),
                                          batch_sampler=_PlanSampler(self),
                                          collate_fn=_PlanCollator(self.collator), **options)
            self._iterator = iter(self._dataloader)
        try:
            started = time.perf_counter()
            batch = next(self._iterator)
            self._wait_seconds += time.perf_counter() - started
        except BaseException:
            self.close()
            raise
        sequence = batch[PLAN_ID]
        if sequence not in self._pending or sequence in self._delivered:
            raise RuntimeError("The worker returned an invalid or repeated plan")
        self._delivered.add(sequence)
        self._materialize_seconds += batch.pop("_sampling_materialize_seconds")
        return batch

    def mark_consumed(self, plan_id):
        plan_id = _nonnegative_integer("plan_id", plan_id)
        if plan_id != self._consumed or plan_id not in self._delivered:
            raise ValueError("Only the next delivered plan can be marked consumed")
        for request in self._pending.pop(plan_id):
            self._last_consumed[self._pair_index[request.treated_id]] = plan_id
            self._forced_consumed += int(request.forced)
        self._delivered.remove(plan_id)
        self._consumed += 1

    def _stop_workers(self):
        if self._iterator is not None and hasattr(self._iterator, "_shutdown_workers"):
            self._iterator._shutdown_workers()
        self._iterator = self._dataloader = None

    def close(self):
        self._stop_workers()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def state_dict(self):
        if self._failed:
            raise RuntimeError("A failed sampling operation cannot be checkpointed")
        return {"version": 1, "signature": self.signature, "next_sequence": self._next_sequence,
                "consumed": self._consumed, "group": self._group.state_dict(),
                "target": self._target.state_dict(), "control": self._control.state_dict(),
                "last_consumed": self._last_consumed.copy(), "forced_consumed": self._forced_consumed,
                "pending": copy.deepcopy(list(self._pending.items()))}

    def load_state_dict(self, state):
        if not isinstance(state, dict) or state.get("version") != 1 or state.get("signature") != self.signature:
            raise ValueError("Visit checkpoint does not match this loader and data")
        next_sequence = _nonnegative_integer("next_sequence", state.get("next_sequence"))
        consumed = _nonnegative_integer("consumed", state.get("consumed"))
        raw_pending = state.get("pending")
        if not isinstance(raw_pending, list) or consumed > next_sequence or len(raw_pending) != next_sequence - consumed:
            raise ValueError("Invalid pending-plan journal")
        last = np.asarray(state.get("last_consumed"))
        forced = _nonnegative_integer("forced_consumed", state.get("forced_consumed"))
        if (last.shape != (len(self.pairs),) or last.dtype.kind not in "iu"
                or int(last.min()) < -1 or int(last.max()) >= consumed or forced > consumed * self.batch_size):
            raise ValueError("Invalid consumed group history")
        last = last.astype(np.int64, copy=True)
        group, target, control = self._new_samplers()
        group.load_state_dict(state.get("group"))
        target.load_state_dict(state.get("target"))
        control.load_state_dict(state.get("control"))
        draw_counts = group.draw_counts
        if group.take_step != next_sequence or sum(map(int, draw_counts)) != next_sequence * self.batch_size:
            raise ValueError("Group counts disagree with the planned position")
        control_totals = Counter()
        for index, pair in enumerate(self.pairs):
            expected = int(draw_counts[index]) * self.dataset.cells_per_group
            if target.total(index) != expected:
                raise ValueError("Target counts disagree with the group counts")
            control_totals[self._control_index[pair["control_id"]]] += expected
        for index in range(len(self.control_ids)):
            if control.total(index) != control_totals[index]:
                raise ValueError("Shared control counts disagree with the group counts")
        pending = OrderedDict()
        pending_counts = np.zeros(len(self.pairs), dtype=np.int64)
        target_rows, control_rows = {}, {}
        for sequence, item in enumerate(raw_pending, start=consumed):
            if not isinstance(item, (tuple, list)) or len(item) != 2 or item[0] != sequence:
                raise ValueError("Pending plan positions must be ordered and contiguous")
            requests = item[1]
            if not isinstance(requests, (tuple, list)) or len(requests) != self.batch_size:
                raise ValueError("Invalid pending batch size")
            ids, checked = set(), []
            for request in requests:
                if not isinstance(request, PopulationRequest) or request.sequence_id != sequence:
                    raise ValueError("Invalid pending population request")
                if request.treated_id not in self._pair_index or request.treated_id in ids:
                    raise ValueError("Pending groups must be distinct and owned by this rank")
                ids.add(request.treated_id)
                index = self._pair_index[request.treated_id]
                pending_counts[index] += 1
                pair = self.pairs[index]
                if type(request.forced) is not bool:
                    raise ValueError("Invalid forced-group flag")
                if request.control_id != pair["control_id"]:
                    raise ValueError("Pending control does not match the original pairing")
                arrays = []
                for rows, size in ((request.target_rows, pair["n_cells"]), (request.control_rows, pair["n_control"])):
                    rows = np.asarray(rows)
                    if (rows.shape != (self.dataset.cells_per_group,) or rows.dtype.kind not in "iu"
                            or int(rows.min()) < 0 or int(rows.max()) >= size or np.any(rows[1:] <= rows[:-1])):
                        raise ValueError("Pending rows must be sorted, distinct and within the pool")
                    rows = rows.astype(np.int64, copy=True)
                    rows.flags.writeable = False
                    arrays.append(rows)
                target_rows.setdefault(index, []).append(arrays[0])
                control_rows.setdefault(self._control_index[request.control_id], []).append(arrays[1])
                checked.append(PopulationRequest(sequence, request.treated_id, request.control_id, *arrays, request.forced))
            pending[sequence] = tuple(checked)
        committed = draw_counts.astype(np.int64) - pending_counts
        if np.any(committed < 0) or np.any((committed == 0) != (last == -1)):
            raise ValueError("Consumed history disagrees with pending group counts")
        for sampler, rows_by_pool in ((target, target_rows), (control, control_rows)):
            for pool, rows in rows_by_pool.items():
                ids, multiplicity = np.unique(np.concatenate(rows), return_counts=True)
                visits = sampler.delta[sampler._slice(pool)][ids].astype(np.uint64) + sampler.base[pool]
                if np.any(visits < multiplicity):
                    raise ValueError("Pending cell rows disagree with visit counts")
        # Validation is complete before stopping workers or replacing any live state.
        self._stop_workers()
        self._group, self._target, self._control = group, target, control
        self._next_sequence, self._consumed, self._pending = next_sequence, consumed, pending
        self._delivered = set()
        self._last_consumed, self._forced_consumed = last, forced
        self._closed = self._failed = False

    def describe(self, *, include_visit_quantiles=False):
        counts = self._group.draw_counts.astype(np.int64)
        target_pending, control_pending = {}, {}
        for requests in self._pending.values():
            for request in requests:
                index = self._pair_index[request.treated_id]
                counts[index] -= 1
                target_pending.setdefault(index, []).append(request.target_rows)
                control_pending.setdefault(self._control_index[request.control_id], []).append(request.control_rows)
        exposure = self.dataset.cells_per_group * counts / self._group.group_sizes
        result = {
            "consumed_microbatches": self._consumed, "planned_microbatches": self._next_sequence,
            "consumed_groups": self._consumed * self.batch_size,
            "consumed_target_cells": self._consumed * self.batch_size * self.dataset.cells_per_group,
            "group_coverage": float(np.count_nonzero(counts) / len(counts)),
            "target_cell_coverage": self._target.consumed_coverage(target_pending),
            "control_cell_coverage": self._control.consumed_coverage(control_pending),
            "exposure_p10": float(np.quantile(exposure, .1)),
            "exposure_p50": float(np.quantile(exposure, .5)),
            "exposure_p90": float(np.quantile(exposure, .9)),
            "longest_unselected_interval": int((self._consumed - 1 - self._last_consumed).max()),
            "forced_groups": self._forced_consumed,
            "planning_seconds": self._planning_seconds, "loader_wait_seconds": self._wait_seconds,
            "materialize_seconds": self._materialize_seconds,
            "rank": self.rank, "world_size": self.world_size,
        }
        for field in ("condition", "cell_type", "donor"):
            totals = Counter()
            for pair, count in zip(self.pairs, counts):
                totals[pair[field]] += int(count)
            result.update({f"groups_by_{field}/{name}": count for name, count in totals.items()})
        if include_visit_quantiles:
            for name, sampler, pending in (("target", self._target, target_pending),
                                            ("control", self._control, control_pending)):
                for percentile, value in zip((10, 50, 90), sampler.quantiles(pending)):
                    result[f"{name}_cell_visits_p{percentile}"] = value
        return result
