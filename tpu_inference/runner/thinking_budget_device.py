# SPDX-License-Identifier: Apache-2.0
"""Persistent device-side control. Stable decode makes no state transfers."""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec

WINDOW = 4


def benchmark_sampler(plain, controlled, rng, mesh, logits, metadata, prepared):
    """Explicit canary diagnostic, with compilation outside measured iterations."""
    import json
    import statistics
    import time

    args = (rng, mesh, logits, metadata)
    jax.block_until_ready(plain(*args))
    jax.block_until_ready(controlled(*args, *prepared))
    timings = {"plain": [], "controlled": []}
    for _ in range(30):
        for name, fn, extra in (("plain", plain, ()), ("controlled", controlled, prepared)):
            start = time.perf_counter()
            jax.block_until_ready(fn(*args, *extra))
            timings[name].append(time.perf_counter() - start)
    print("THINKING_SAMPLER_MICROBENCHMARK " + json.dumps({
        "median_seconds": {k: statistics.median(v) for k, v in timings.items()},
        "shape": logits.shape, "do_sampling": metadata.do_sampling}), flush=True)


class DeviceBank(NamedTuple):
    history: object
    positions: object
    budgets: object
    starts: object
    ends: object
    transitions: object
    lengths: object


def device_step(tokens, bank, slots, valid, detector=None):
    """Called inside the sampler JIT, not as an additional decode dispatch."""
    from .thinking_budget import apply_budget

    capacity = bank.positions.shape[0]
    safe = jnp.minimum(slots, capacity - 1)
    position = bank.positions[safe]
    states = bank.history[safe, position % WINDOW]
    selected, states, _ = apply_budget(
        states, tokens, bank.budgets[safe], bank.starts[safe], bank.ends[safe],
        bank.transitions[safe], bank.lengths[safe], valid, detector)
    # Invalid/padded rows must not overwrite each other or a real slot.
    target = jnp.where(valid, slots, capacity)
    next_position = position + valid.astype(jnp.int32)
    history = bank.history.at[target, next_position % WINDOW].set(states, mode="drop")
    positions = bank.positions.at[target].set(next_position, mode="drop")
    return selected, bank._replace(history=history, positions=positions)


@jax.jit
def replace_rows(bank, slots, rows):
    return DeviceBank(*(value.at[slots].set(row) for value, row in zip(bank, rows)))


@jax.jit
def rewind_positions(bank, slots, positions):
    return bank._replace(positions=bank.positions.at[slots].set(positions))


class BudgetRunner:
    def __init__(self):
        self.bank = None
        self.capacity = 16
        self.states = {}  # request ID -> (request object, stable device slot)
        self.specs = {}
        self.history = {}  # Available device snapshot positions, no device slices.
        self.expected = {}
        self.inputs = {}
        self.signature = None
        self.batch_arrays = None
        self.assignments = []
        self.detector_key = None
        self.detector = None
        self.step = jax.jit(device_step)  # CPU tests only; production fuses this.

    def prepare(self, runner, scheduler, rows, req_ids_dp):
        from .thinking_budget import (
            KEY, MAX_TRANSITION, MAX_MARKER, ANSWERING, THINKING, BudgetSpec, audit_output, initial_state)

        for key in self.inputs.keys() - runner.requests.keys():
            for mapping in (self.states, self.specs, self.history, self.expected, self.inputs):
                mapping.pop(key, None)
        tracked = []
        for key in runner.input_batch.req_ids:
            request = runner.requests[key]
            params = request.sampling_params
            raw = (params.extra_args or {}).get(KEY) if params else None
            if raw is not None:
                if key not in self.inputs or self.inputs[key][0] is not request:
                    settings = BudgetSpec.parse(raw, runner.input_batch.vocab_size)
                    phase = initial_state(settings, request.prompt_token_ids)[0]
                    self.inputs[key] = (request, settings, phase)
                _, settings, phase = self.inputs[key]
                # These are proofs of completion, not guesses from placeholders:
                # an initially open request must finish by cap + transition.
                # Keep its rollback ring; a rewind below that bound re-enables it.
                if phase == ANSWERING or (phase == THINKING and
                        len(request.output_token_ids) >= settings.budget + len(settings.transition)):
                    continue
                tracked.append((key, request, raw))
        if not tracked:
            return None
        if runner.speculative_config or getattr(runner, "enable_continue_decode", False):
            raise ValueError("thinking budgets require non-speculative single-step decoding")
        sharding = NamedSharding(runner.mesh, PartitionSpec())
        put = lambda x: jax.device_put(x, sharding)
        detectors = {self.inputs[k][1].completer_detector for k, _, _ in tracked}
        if len(detectors) != 1:
            raise ValueError("an engine must use one completer detector")
        encoded = next(iter(detectors))
        if self.detector_key is None:
            from .thinking_budget import text_detector
            self.detector_key = encoded
            if encoded:
                classes, table = text_detector(encoded)
                # Model logits can have a padded vocabulary (Qwen: 248320
                # logits vs 248070 tokenizer entries). Padding is not a marker.
                classes = np.pad(np.asarray(classes, np.int32),
                                 (0, runner.input_batch.vocab_size-len(classes)))
                self.detector = (put(classes), put(np.asarray(table, np.int32)))
        elif encoded != self.detector_key:
            raise ValueError("cannot change an engine's completer detector")
        new = [(k, r, raw) for k, r, raw in tracked
               if k not in self.states or self.states[k][0] is not r]
        required = len(self.states) + sum(k not in self.states for k, _, _ in new)
        if self.bank is None:
            while self.capacity < required:
                self.capacity *= 2
            n = self.capacity
            self.bank = DeviceBank(*(put(x) for x in (
                np.zeros((n, WINDOW, 3), np.int32), np.zeros(n, np.int32),
                np.zeros(n, np.int32), np.full((n, MAX_MARKER), -1, np.int32), np.full((n, MAX_MARKER), -1, np.int32),
                np.zeros((n, MAX_TRANSITION), np.int32), np.ones(n, np.int32))))
        elif required > self.capacity:
            old = self.capacity
            while self.capacity < required:
                self.capacity *= 2
            self.bank = DeviceBank(*(jnp.pad(x, [(0, self.capacity-old)] + [(0, 0)]*(x.ndim-1))
                                     for x in self.bank))
            self.signature = None
        free = iter(sorted(set(range(self.capacity)) - {v[1] for v in self.states.values()}))
        changed = []
        for key, request, raw in new:
            slot = self.states[key][1] if key in self.states else next(free)
            settings = self.inputs[key][1]
            self.states[key] = (request, slot)
            self.specs[key] = settings
            self.history[key] = set()
            self.expected[key] = -1
            changed.append(key)
        rewind = []
        for key, request, _ in tracked:
            count = len(request.output_token_ids)
            if count != self.expected[key]:
                if count in self.history[key]:
                    rewind.append((self.states[key][1], count))
                    self.history[key] = {n for n in self.history[key] if n <= count}
                elif key not in changed:
                    changed.append(key)
                self.expected[key] = count
        if changed:
            slots, values = [], []
            for key in changed:
                request, slot = self.states[key]
                settings = self.specs[key]
                count = len(request.output_token_ids)
                # Only cold start/deep rollback consults committed CPU tokens.
                _, state = audit_output(settings, request.prompt_token_ids, request.output_token_ids)
                history = np.zeros((WINDOW, 3), np.int32)
                history[count % WINDOW] = state
                transition = np.zeros(MAX_TRANSITION, np.int32)
                transition[:len(settings.transition)] = settings.transition
                slots.append(slot)
                start = np.full(MAX_MARKER, -1, np.int32)
                end = np.full(MAX_MARKER, -1, np.int32)
                start[:len(settings.start)], end[:len(settings.end)] = settings.start, settings.end
                values.append((history, count, settings.budget, start,
                               end, transition, len(settings.transition)))
                self.history[key] = {count}
            arrays = DeviceBank(*(put(np.asarray(x, np.int32)) for x in zip(*values)))
            self.bank = replace_rows(self.bank, put(np.asarray(slots, np.int32)), arrays)
        if rewind:
            slots, positions = zip(*rewind)
            self.bank = rewind_positions(self.bank, put(np.asarray(slots, np.int32)),
                                         put(np.asarray(positions, np.int32)))
        groups = req_ids_dp or {0: list(runner.input_batch.req_ids)}
        if rows % len(groups):
            raise ValueError("thinking-budget batch is not evenly padded across DP groups")
        per_group = rows // len(groups)
        slots, valid = [self.capacity] * rows, [False] * rows
        self.assignments = []
        keys = {k for k, _, _ in tracked}
        for group, ids in groups.items():
            for offset, key in enumerate(ids):
                if key not in keys:
                    continue
                index = group * per_group + offset
                request, slot = self.states[key]
                ready = request.num_computed_tokens + scheduler.num_scheduled_tokens[key] >= request.num_tokens
                slots[index], valid[index] = slot, ready
                self.assignments.append((key, ready))
        signature = (tuple(slots), tuple(valid))
        if signature != self.signature:
            self.batch_arrays = (put(np.asarray(slots, np.int32)), put(np.asarray(valid, bool)))
            self.signature = signature
        prepared = (self.bank, *self.batch_arrays)
        return (*prepared, self.detector) if self.detector is not None else prepared

    def commit(self, bank):
        self.bank = bank
        for key, valid in self.assignments:
            count = self.expected[key] + int(valid)
            self.expected[key] = count
            self.history[key].add(count)
            self.history[key] = {n for n in self.history[key] if count-WINDOW < n <= count}

    def apply(self, runner, scheduler, tokens, req_ids_dp):
        """CPU test harness; the TPU runner uses prepare/fused sample/commit."""
        prepared = self.prepare(runner, scheduler, tokens.shape[0], req_ids_dp)
        if prepared is None:
            return tokens
        tokens, bank = self.step(tokens, *prepared)
        self.commit(bank)
        return tokens
