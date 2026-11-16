"""Group scheduling adapted from Perturb's Tahoe visit sampler.

Source: Perturb/src/virtual_cell_perturb/data/tahoe_visit_sampler.py
Copyright (c) 2026. MIT license: LICENSE.perturb in this directory.
Only the group sampler and its pure NumPy helpers are retained here.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np


GROUP_STATE_VERSION = 2
UINT32_MAX = int(np.iinfo(np.uint32).max)
UINT64_MAX = int(np.iinfo(np.uint64).max)


def _positive_integer(name: str, value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError("{} must be an integer".format(name))
    result = int(value)
    if result <= 0:
        raise ValueError("{} must be positive".format(name))
    return result


def _nonnegative_integer(name: str, value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError("{} must be an integer".format(name))
    result = int(value)
    if result < 0:
        raise ValueError("{} must be nonnegative".format(name))
    return result


def _positive_float(name: str, value: Any) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError("{} must be a number".format(name))
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError("{} must be finite and positive".format(name))
    return result


def _probability(name: str, value: Any) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError("{} must be a number".format(name))
    result = float(value)
    if not np.isfinite(result) or result < 0.0 or result > 1.0:
        raise ValueError("{} must be in [0, 1]".format(name))
    return result


def _positive_sizes(name: str, values: Sequence[int]) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError("{} must be a non-empty one-dimensional sequence".format(name))
    if raw.dtype.kind not in ("i", "u"):
        raise ValueError("{} must contain integers".format(name))
    maximum = int(raw.max())
    if maximum > int(np.iinfo(np.int64).max):
        raise ValueError("{} contains a value outside int64".format(name))
    result = raw.astype(np.int64, copy=True)
    if np.any(result <= 0):
        raise ValueError("{} must contain only positive values".format(name))
    return result


def _integer_codes(name: str, values: Sequence[int], size: int) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.size != size:
        raise ValueError("{} must contain one code per group".format(name))
    if raw.dtype.kind not in ("i", "u"):
        raise ValueError("{} must contain integers".format(name))
    if raw.size:
        if raw.dtype.kind == "u" and int(raw.max()) > int(np.iinfo(np.int64).max):
            raise ValueError("{} contains a value outside int64".format(name))
        if raw.dtype.kind == "i" and int(raw.min()) < int(np.iinfo(np.int64).min):
            raise ValueError("{} contains a value outside int64".format(name))
    return raw.astype(np.int64, copy=True)


def _new_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(_nonnegative_integer("seed", seed))


def _validated_rng_state(value: Any) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("rng_state must be a mapping")
    state = copy.deepcopy(dict(value))
    candidate = np.random.default_rng()
    try:
        candidate.bit_generator.state = state
    except (TypeError, ValueError) as exc:
        raise ValueError("rng_state is invalid") from exc
    return state


class _FenwickTree:
    """Mutable prefix-sum tree over nonnegative float64 weights."""

    def __init__(self, weights: Sequence[float]) -> None:
        values = np.asarray(weights, dtype=np.float64)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("Fenwick weights must be a non-empty vector")
        if np.any(~np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("Fenwick weights must be finite and nonnegative")
        self.size = int(values.size)
        self.values = values.copy()
        self.tree = np.zeros(self.size + 1, dtype=np.float64)
        self.tree[1:] = values
        for index in range(1, self.size + 1):
            parent = index + (index & -index)
            if parent <= self.size:
                self.tree[parent] += self.tree[index]
        self._total = 0.0
        index = self.size
        while index > 0:
            self._total += float(self.tree[index])
            index -= index & -index

    @property
    def total(self) -> float:
        return self._total

    def set(self, index: int, value: float) -> None:
        index = int(index)
        value = float(value)
        if index < 0 or index >= self.size:
            raise IndexError("Fenwick index is outside the tree")
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("Fenwick weight must be finite and nonnegative")
        difference = value - float(self.values[index])
        self.values[index] = value
        self._total += difference
        position = index + 1
        while position <= self.size:
            self.tree[position] += difference
            position += position & -position

    def find(self, cumulative: float) -> int:
        """Return the index whose half-open mass interval contains cumulative."""

        total = self.total
        cumulative = float(cumulative)
        if not np.isfinite(cumulative) or cumulative < 0.0 or cumulative >= total:
            raise ValueError("cumulative mass must be in [0, total)")
        index = 0
        bit = 1 << (self.size.bit_length() - 1)
        while bit:
            candidate = index + bit
            if candidate <= self.size and float(self.tree[candidate]) <= cumulative:
                cumulative -= float(self.tree[candidate])
                index = candidate
            bit >>= 1
        if index >= self.size:  # pragma: no cover - guarded by cumulative < total
            raise RuntimeError("Fenwick search exceeded the tree")
        return index

    def sample(self, rng: np.random.Generator) -> int:
        total = self.total
        if not np.isfinite(total) or total <= 0.0:
            raise RuntimeError("Cannot sample a Fenwick tree with zero total weight")
        return self.find(float(rng.random()) * total)


class InfiniteGroupSampler:
    """Draw distinct groups for an unbounded stream of optimizer batches.

    Every group has one global visit count and one global score:

    ``n / (1 + cells_per_group * draw_count / n) ** gamma``.

    Without ``locality_codes``, every position is sampled sequentially from the
    global uniform/score mixture. With locality enabled, the first stochastic
    position is still drawn from that exact global distribution and becomes the
    batch's locality anchor. Remaining stochastic positions use the global
    distribution conditioned on the anchor's locality. The local Fenwick trees
    are derived indexes over the same global scores; they own no counts, decay,
    or deadlines.

    Globally overdue groups are always emitted first. They may cross locality
    boundaries and therefore override the one-locality preference. Any remaining
    positions prefer a locality already represented by those forced groups.
    Consequently locality improves storage access without multiplying either
    visit decay or deadline pressure.
    """

    def __init__(
        self,
        group_sizes: Sequence[int],
        groups_per_batch: int,
        *,
        cells_per_group: int = 256,
        gamma: float = 1.0,
        uniform_probability: float = 0.02,
        deadline_multiplier: Optional[float] = 10.0,
        locality_codes: Optional[Sequence[int]] = None,
        seed: int = 0,
    ) -> None:
        self.group_sizes = _positive_sizes("group_sizes", group_sizes)
        self.groups_per_batch = _positive_integer(
            "groups_per_batch", groups_per_batch
        )
        if self.groups_per_batch > self.group_sizes.size:
            raise ValueError("groups_per_batch cannot exceed the number of groups")
        self.cells_per_group = _positive_integer("cells_per_group", cells_per_group)
        self.gamma = _positive_float("gamma", gamma)
        self.uniform_probability = _probability(
            "uniform_probability", uniform_probability
        )
        self.deadline_multiplier = (
            None
            if deadline_multiplier is None
            else _positive_float("deadline_multiplier", deadline_multiplier)
        )
        self._locality_codes = (
            None
            if locality_codes is None
            else _integer_codes(
                "locality_codes", locality_codes, int(self.group_sizes.size)
            )
        )
        self._build_locality_index()
        self.rng = _new_rng(seed)
        self._draw_counts = np.zeros(self.group_sizes.size, dtype=np.uint64)
        scores = self._calculate_scores(self._draw_counts)
        (
            self._score_tree,
            self._available_tree,
            self._local_score_trees,
            self._local_available_trees,
        ) = self._new_sampling_trees(scores)
        self._take_step = 0
        self._deadline_steps = self._calculate_deadline_steps()
        self._next_due = (
            None if self._deadline_steps is None else self._deadline_steps.copy()
        )

    @property
    def group_count(self) -> int:
        return int(self.group_sizes.size)

    @property
    def draw_counts(self) -> np.ndarray:
        return self._draw_counts.copy()

    @property
    def scores(self) -> np.ndarray:
        return self._calculate_scores(self._draw_counts)

    @property
    def locality_codes(self) -> Optional[np.ndarray]:
        return None if self._locality_codes is None else self._locality_codes.copy()

    @property
    def take_step(self) -> int:
        return self._take_step

    @property
    def deadline_steps(self) -> Optional[np.ndarray]:
        return None if self._deadline_steps is None else self._deadline_steps.copy()

    @property
    def next_due(self) -> Optional[np.ndarray]:
        return None if self._next_due is None else self._next_due.copy()

    def _build_locality_index(self) -> None:
        self._locality_group_indices: Dict[int, np.ndarray] = {}
        self._locality_position = np.full(self.group_count, -1, dtype=np.int64)
        if self._locality_codes is None:
            return
        for raw_code in np.unique(self._locality_codes):
            code = int(raw_code)
            indices = np.flatnonzero(self._locality_codes == raw_code).astype(
                np.int64, copy=False
            )
            self._locality_group_indices[code] = indices
            self._locality_position[indices] = np.arange(
                indices.size, dtype=np.int64
            )

    def _new_sampling_trees(self, scores: np.ndarray):
        score_tree = _FenwickTree(scores)
        available_tree = _FenwickTree(np.ones(self.group_count, dtype=np.float64))
        local_score_trees: Dict[int, _FenwickTree] = {}
        local_available_trees: Dict[int, _FenwickTree] = {}
        for code, indices in self._locality_group_indices.items():
            local_score_trees[code] = _FenwickTree(scores[indices])
            local_available_trees[code] = _FenwickTree(
                np.ones(indices.size, dtype=np.float64)
            )
        return (
            score_tree,
            available_tree,
            local_score_trees,
            local_available_trees,
        )

    def _calculate_deadline_steps(self) -> Optional[np.ndarray]:
        if self.deadline_multiplier is None:
            return None
        total_cells = float(self.group_sizes.sum(dtype=np.int64))
        raw = (
            self.deadline_multiplier
            * total_cells
            / (
                float(self.groups_per_batch)
                * self.group_sizes.astype(np.float64, copy=False)
            )
        )
        if np.any(~np.isfinite(raw)) or np.any(raw > float(UINT64_MAX)):
            raise OverflowError("Group deadline steps are no longer representable")
        return np.maximum(1.0, np.ceil(raw)).astype(np.uint64)

    def _overdue_groups(self, excluded: np.ndarray) -> np.ndarray:
        if self._next_due is None:
            return np.empty(0, dtype=np.int64)
        candidates = np.flatnonzero(
            (~excluded) & (self._next_due <= np.uint64(self._take_step))
        )
        if candidates.size <= 1:
            return candidates.astype(np.int64, copy=False)
        order = np.lexsort((candidates, self._next_due[candidates]))
        return candidates[order].astype(np.int64, copy=False)

    def _calculate_scores(
        self, counts: np.ndarray, group_sizes: Optional[np.ndarray] = None
    ) -> np.ndarray:
        source_sizes = self.group_sizes if group_sizes is None else group_sizes
        sizes = source_sizes.astype(np.float64, copy=False)
        exposure = (
            float(self.cells_per_group)
            * counts.astype(np.float64, copy=False)
            / sizes
        )
        scores = sizes / np.power(1.0 + exposure, self.gamma)
        if np.any(~np.isfinite(scores)) or np.any(scores <= 0.0):
            raise OverflowError("Group scores are no longer representable")
        return scores

    def probabilities(self, excluded: Optional[Sequence[int]] = None) -> np.ndarray:
        """Return the next global-draw probabilities, ignoring locality preference."""

        available = np.ones(self.group_count, dtype=bool)
        if excluded is not None:
            raw = np.asarray(excluded)
            if raw.ndim != 1 or raw.dtype.kind not in ("i", "u"):
                raise ValueError("excluded must be a one-dimensional integer sequence")
            indices = raw.astype(np.int64, copy=False)
            if np.any(indices < 0) or np.any(indices >= self.group_count):
                raise ValueError("excluded contains an invalid group index")
            available[indices] = False
        count = int(available.sum())
        if count == 0:
            raise ValueError("At least one group must remain available")
        overdue = self._overdue_groups(~available)
        if overdue.size:
            result = np.zeros(self.group_count, dtype=np.float64)
            result[int(overdue[0])] = 1.0
            return result
        scores = self.scores
        weighted_total = float(scores[available].sum(dtype=np.float64))
        result = np.zeros(self.group_count, dtype=np.float64)
        result[available] = (
            self.uniform_probability / count
            + (1.0 - self.uniform_probability)
            * scores[available]
            / weighted_total
        )
        return result

    def _draw_global_index(self) -> int:
        if float(self.rng.random()) < self.uniform_probability:
            available = int(round(self._available_tree.total))
            rank = int(self.rng.integers(available))
            return self._available_tree.find(float(rank))
        return self._score_tree.sample(self.rng)

    def _draw_locality_conditioned_index(self, code: int) -> int:
        """Sample the current global mixture conditioned on one locality."""

        local_available_tree = self._local_available_trees[code]
        local_score_tree = self._local_score_trees[code]
        local_count = int(round(local_available_tree.total))
        global_count = int(round(self._available_tree.total))
        if local_count <= 0 or global_count <= 0:
            raise RuntimeError("Cannot sample an empty locality")

        # Conditioning the global mixture requires conditioning its two masses;
        # restarting epsilon inside the locality would define a different law.
        uniform_mass = (
            self.uniform_probability * float(local_count) / float(global_count)
        )
        weighted_mass = (
            (1.0 - self.uniform_probability)
            * local_score_tree.total
            / self._score_tree.total
        )
        local_mass = uniform_mass + weighted_mass
        use_uniform = float(self.rng.random()) < uniform_mass / local_mass
        if use_uniform:
            rank = int(self.rng.integers(local_count))
            local_position = local_available_tree.find(float(rank))
        else:
            local_position = local_score_tree.sample(self.rng)
        return int(self._locality_group_indices[code][local_position])

    def _mark_unavailable(self, group_index: int) -> float:
        old_score = float(self._score_tree.values[group_index])
        self._score_tree.set(group_index, 0.0)
        self._available_tree.set(group_index, 0.0)
        if self._locality_codes is not None:
            code = int(self._locality_codes[group_index])
            local_position = int(self._locality_position[group_index])
            self._local_score_trees[code].set(local_position, 0.0)
            self._local_available_trees[code].set(local_position, 0.0)
        return old_score

    def _mark_available(self, group_index: int, score: float) -> None:
        self._score_tree.set(group_index, score)
        self._available_tree.set(group_index, 1.0)
        if self._locality_codes is not None:
            code = int(self._locality_codes[group_index])
            local_position = int(self._locality_position[group_index])
            self._local_score_trees[code].set(local_position, score)
            self._local_available_trees[code].set(local_position, 1.0)

    def _preferred_forced_locality(
        self, selected: Sequence[int], remaining: int
    ) -> Optional[int]:
        if self._locality_codes is None:
            return None
        first_position: Dict[int, int] = {}
        selected_count: Dict[int, int] = {}
        for position, group_index in enumerate(selected):
            code = int(self._locality_codes[int(group_index)])
            first_position.setdefault(code, position)
            selected_count[code] = selected_count.get(code, 0) + 1
        candidates = []
        for code, count in selected_count.items():
            available = int(round(self._local_available_trees[code].total))
            if available:
                candidates.append(
                    (
                        int(available >= remaining),
                        count,
                        available,
                        -first_position[code],
                        code,
                    )
                )
        return None if not candidates else int(max(candidates)[-1])

    def take(self) -> np.ndarray:
        """Return one optimizer batch of distinct global group indices."""

        if self._take_step == UINT64_MAX:
            raise OverflowError("Group take step reached uint64 capacity")
        selected = []
        old_scores = []
        excluded = np.zeros(self.group_count, dtype=bool)

        try:
            overdue = self._overdue_groups(excluded)
            forced_count = min(self.groups_per_batch, int(overdue.size))
            for raw_index in overdue[:forced_count]:
                group_index = int(raw_index)
                selected.append(group_index)
                old_scores.append(self._mark_unavailable(group_index))
                excluded[group_index] = True

            if self._locality_codes is None:
                while len(selected) < self.groups_per_batch:
                    group_index = self._draw_global_index()
                    selected.append(group_index)
                    old_scores.append(self._mark_unavailable(group_index))
                    excluded[group_index] = True
            else:
                remaining = self.groups_per_batch - len(selected)
                anchor = self._preferred_forced_locality(selected, remaining)
                while len(selected) < self.groups_per_batch:
                    if anchor is None:
                        group_index = self._draw_global_index()
                        anchor = int(self._locality_codes[group_index])
                    elif self._local_available_trees[anchor].total > 0.0:
                        group_index = self._draw_locality_conditioned_index(anchor)
                    else:
                        group_index = self._draw_global_index()
                        anchor = int(self._locality_codes[group_index])
                    selected.append(group_index)
                    old_scores.append(self._mark_unavailable(group_index))
                    excluded[group_index] = True

            selected_int64 = np.asarray(selected, dtype=np.int64)
            if np.any(self._draw_counts[selected_int64] == np.uint64(UINT64_MAX)):
                raise OverflowError("A group draw count reached uint64 capacity")
            new_counts = self._draw_counts[selected_int64] + np.uint64(1)
            new_scores = self._calculate_scores(
                new_counts, self.group_sizes[selected_int64]
            )
            if self._next_due is not None:
                maximum_deadline = int(self._deadline_steps[selected_int64].max())
                if self._take_step > UINT64_MAX - maximum_deadline:
                    raise OverflowError("Group deadline clock reached uint64 capacity")
        except BaseException:
            for group_index, score in zip(selected, old_scores):
                self._mark_available(group_index, score)
            raise

        self._draw_counts[selected_int64] = new_counts
        if self._next_due is not None:
            self._next_due[selected_int64] = (
                np.uint64(self._take_step) + self._deadline_steps[selected_int64]
            )
        for group_index, score in zip(selected, new_scores):
            self._mark_available(group_index, float(score))
        self._take_step += 1
        return selected_int64.astype(np.int32, copy=False)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "version": GROUP_STATE_VERSION,
            "group_sizes": self.group_sizes.copy(),
            "groups_per_batch": self.groups_per_batch,
            "cells_per_group": self.cells_per_group,
            "gamma": self.gamma,
            "uniform_probability": self.uniform_probability,
            "deadline_multiplier": self.deadline_multiplier,
            "locality_codes": (
                None if self._locality_codes is None else self._locality_codes.copy()
            ),
            "draw_counts": self._draw_counts.copy(),
            "take_step": self._take_step,
            "next_due": None if self._next_due is None else self._next_due.copy(),
            "rng_state": copy.deepcopy(self.rng.bit_generator.state),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise ValueError("Group sampler state must be a mapping")
        if int(state.get("version", -1)) != GROUP_STATE_VERSION:
            raise ValueError("Unsupported group sampler state version")
        expected = {
            "groups_per_batch": self.groups_per_batch,
            "cells_per_group": self.cells_per_group,
            "gamma": self.gamma,
            "uniform_probability": self.uniform_probability,
            "deadline_multiplier": self.deadline_multiplier,
        }
        for name, value in expected.items():
            if state.get(name) != value:
                raise ValueError("Group sampler state has a different {}".format(name))
        sizes = _positive_sizes("state group_sizes", state.get("group_sizes", ()))
        if not np.array_equal(sizes, self.group_sizes):
            raise ValueError("Group sampler state has different group sizes")
        raw_locality_codes = state.get("locality_codes")
        if self._locality_codes is None:
            if raw_locality_codes is not None:
                raise ValueError("Group sampler state has different locality codes")
        else:
            if raw_locality_codes is None:
                raise ValueError("Group sampler state has different locality codes")
            locality_codes = _integer_codes(
                "state locality_codes", raw_locality_codes, self.group_count
            )
            if not np.array_equal(locality_codes, self._locality_codes):
                raise ValueError("Group sampler state has different locality codes")
        raw_counts = np.asarray(state.get("draw_counts"))
        if raw_counts.shape != self._draw_counts.shape or raw_counts.dtype.kind not in (
            "i",
            "u",
        ):
            raise ValueError("Group sampler draw_counts are invalid")
        if raw_counts.dtype.kind == "i" and np.any(raw_counts < 0):
            raise ValueError("Group sampler draw_counts cannot be negative")
        if raw_counts.size and int(raw_counts.max()) > UINT64_MAX:
            raise ValueError("Group sampler draw_counts exceed uint64")
        counts = raw_counts.astype(np.uint64, copy=True)
        scores = self._calculate_scores(counts)
        take_step = _nonnegative_integer("take_step", state.get("take_step"))
        if take_step > UINT64_MAX:
            raise ValueError("Group sampler take_step exceeds uint64")
        raw_next_due = state.get("next_due")
        if self._deadline_steps is None:
            if raw_next_due is not None:
                raise ValueError("Group sampler next_due must be None")
            next_due = None
        else:
            candidate_due = np.asarray(raw_next_due)
            if (
                candidate_due.shape != self._deadline_steps.shape
                or candidate_due.dtype.kind not in ("i", "u")
            ):
                raise ValueError("Group sampler next_due is invalid")
            if candidate_due.dtype.kind == "i" and np.any(candidate_due < 0):
                raise ValueError("Group sampler next_due cannot be negative")
            if candidate_due.size and int(candidate_due.max()) > UINT64_MAX:
                raise ValueError("Group sampler next_due exceeds uint64")
            next_due = candidate_due.astype(np.uint64, copy=True)
        rng_state = _validated_rng_state(state.get("rng_state"))
        (
            score_tree,
            available_tree,
            local_score_trees,
            local_available_trees,
        ) = self._new_sampling_trees(scores)

        self._draw_counts = counts
        self._score_tree = score_tree
        self._available_tree = available_tree
        self._local_score_trees = local_score_trees
        self._local_available_trees = local_available_trees
        self._take_step = take_step
        self._next_due = next_due
        self.rng.bit_generator.state = rng_state
