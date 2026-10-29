# SPDX-License-Identifier: Apache-2.0
"""Maximum-only thinking control, independent of vLLM batch positions.

Only generated reasoning consumes the budget. Markers in the prompt establish
the initial phase. Forced tokens are part of the same autoregressive request.
"""
from dataclasses import dataclass
import json
import base64
import zlib
from functools import lru_cache

KEY = "skyrl_thinking_budget"
WAITING, THINKING, FORCING, ANSWERING = range(4)
MAX_TRANSITION = 128
MAX_MARKER = 16


def build_text_detector(tokenizer, marker, vocab_size):
    """Compile the completer's decoded substring test into token transitions.

    The supported markers are ASCII without whitespace. Decode individual
    tokens with special tokens retained: this also recognizes alternate token
    segmentations and a literal marker inside reasoning, just like the client.
    A compact token-class table is shared by every request on the engine.
    """
    if not marker.isascii() or any(c.isspace() for c in marker):
        raise ValueError("completion markers must be non-whitespace ASCII")
    size = len(marker)
    ids = sorted(set(tokenizer.get_vocab().values()))
    classes, effects, indices = [0] * vocab_size, [tuple([0] * size + [size])], {}
    indices[effects[0]] = 0
    for offset in range(0, len(ids), 4096):
        batch = ids[offset:offset+4096]
        pieces = tokenizer.batch_decode([[i] for i in batch], skip_special_tokens=False,
                                         clean_up_tokenization_spaces=False)
        for token, piece in zip(batch, pieces):
            effect = []
            for state in range(size):
                text = marker[:state] + piece
                effect.append(size if marker in text else next(
                    k for k in range(size-1, -1, -1) if text.endswith(marker[:k])))
            signature = tuple(effect + [size])
            if signature not in indices:
                indices[signature] = len(effects)
                effects.append(signature)
            classes[token] = indices[signature]
    data = dict(marker=marker, classes=classes, next=list(map(list, zip(*effects))))
    return base64.b64encode(zlib.compress(json.dumps(data, separators=(",", ":")).encode())).decode()


@lru_cache(maxsize=8)
def text_detector(encoded):
    data = json.loads(zlib.decompress(base64.b64decode(encoded)))
    return data["classes"], data["next"]


@dataclass(frozen=True)
class BudgetSpec:
    budget: int
    start_id: int
    end_id: int
    transition: tuple[int, ...]
    start_sequence: tuple[int, ...] = ()
    end_sequence: tuple[int, ...] = ()
    completer_detector: str = ""

    @property
    def start(self):
        return self.start_sequence or (self.start_id,)

    @property
    def end(self):
        return self.end_sequence or (self.end_id,)

    @classmethod
    def parse(cls, raw, vocab_size=None):
        data = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(data, dict):
            raise ValueError("invalid thinking-budget specification")
        detector = data.get("completer_detector", "")
        fields = set(data) - {"completer_detector"}
        if fields not in ({"budget", "start_id", "end_id", "transition"}, {"budget", "start_id", "end_id", "transition", "start_sequence", "end_sequence"}):
            raise ValueError("invalid thinking-budget specification")
        if not isinstance(detector, str):
            raise ValueError("invalid completer detector")
        if type(data["budget"]) is not int or not 0 <= data["budget"] <= 2**30:
            raise ValueError("thinking budget must be a nonnegative integer")
        tokens = data["transition"]
        if not isinstance(tokens, list) or not 1 <= len(tokens) <= MAX_TRANSITION:
            raise ValueError("transition must contain 1..128 token IDs")
        start = data.get("start_sequence") or [data["start_id"]]
        end = data.get("end_sequence") or [data["end_id"]]
        for marker in (start, end):
            if not isinstance(marker, list) or not 1 <= len(marker) <= MAX_MARKER:
                raise ValueError("markers require 1..16 token IDs")
            # Enables a constant-cost prefix automaton with a one-token fallback.
            if marker[0] in marker[1:]:
                raise ValueError("marker first token must not repeat inside the marker")
        ids = [data["start_id"], data["end_id"], *tokens, *start, *end]
        if any(type(t) is not int or t < 0 or (vocab_size is not None and t >= vocab_size) for t in ids):
            raise ValueError("invalid thinking token ID")
        if start == end or (not detector and not any(tokens[i:i+len(end)] == end for i in range(len(tokens)))):
            raise ValueError("distinct markers and a closing marker in the transition are required")
        if detector:
            classes, states = text_detector(detector)
            if vocab_size is not None and len(classes) > vocab_size:
                raise ValueError("completer detector vocabulary mismatch")
            state = 0
            for token in tokens:
                state = states[state][classes[token]]
            if state != len(states)-1:
                raise ValueError("transition must contain the completer closing marker")
        return cls(data["budget"], data["start_id"], data["end_id"], tuple(tokens), tuple(start), tuple(end), detector)

    def serialize(self):
        return json.dumps(dict(budget=self.budget, start_id=self.start_id,
                               end_id=self.end_id, transition=list(self.transition),
                               start_sequence=list(self.start), end_sequence=list(self.end),
                               **({"completer_detector": self.completer_detector} if self.completer_detector else {})))


def _advance(marker, cursor, token):
    return cursor + 1 if token == marker[cursor] else int(token == marker[0])


def initial_state(spec, prompt):
    if spec.completer_detector:
        # Existing completers count every generated token, ignore markers in
        # the input prompt, and decide whether to force by inspecting output.
        return (THINKING, 0, 0)
    # Inspect only marker patterns, never count prompt tokens as reasoning.
    phase, start_cursor, end_cursor = WAITING, 0, 0
    for token in prompt:
        start_cursor = _advance(spec.start, start_cursor, token)
        end_cursor = _advance(spec.end, end_cursor, token)
        if start_cursor == len(spec.start):
            phase, start_cursor, end_cursor = THINKING, 0, 0
        elif end_cursor == len(spec.end):
            phase, start_cursor, end_cursor = ANSWERING, 0, 0
    # A partial next reasoning header can straddle prompt and generated tokens.
    if start_cursor:
        phase = WAITING
    return (phase, 0, start_cursor if phase == WAITING else end_cursor if phase == THINKING else 0)


def reference_step(spec, state, sampled):
    """CPU oracle and response audit; never used in the TPU decode hot path."""
    phase, count, cursor = state
    forced = phase == FORCING or (phase == THINKING and count >= spec.budget)
    if forced:
        if phase != FORCING:
            # Finish an answer header already in progress instead of duplicating it.
            cursor = (len(spec.transition) - len(spec.end) + cursor
                      if not spec.completer_detector and cursor and spec.transition[-len(spec.end):] == spec.end else 0)
        token = spec.transition[cursor]
        cursor += 1
        return token, (ANSWERING if cursor == len(spec.transition) else FORCING, count, cursor), True
    if spec.completer_detector and phase == THINKING:
        classes, states = text_detector(spec.completer_detector)
        cursor = states[cursor][classes[sampled] if 0 <= sampled < len(classes) else 0]
        count += 1
        if cursor == len(states)-1:
            phase, cursor = ANSWERING, 0
    elif phase == WAITING:
        cursor = _advance(spec.start, cursor, sampled)
        if cursor == len(spec.start):
            phase, cursor = THINKING, 0
    elif phase == THINKING:
        cursor = _advance(spec.end, cursor, sampled)
        if cursor == len(spec.end):
            phase, cursor = ANSWERING, 0
        else:
            count += 1
    return sampled, (phase, count, cursor), False


def audit_output(spec, prompt, tokens):
    state = initial_state(spec, prompt)
    forced_positions = []
    for position, token in enumerate(tokens):
        expected, state, forced = reference_step(spec, state, token)
        if expected != token:
            raise RuntimeError(f"thinking budget was not enforced at output token {position}")
        if forced:
            forced_positions.append(position)
    return forced_positions, state


def apply_budget(states, sampled, budgets, starts, ends, transitions, lengths, valid, detector=None):
    """JAX implementation. All request-specific settings are dynamic arrays."""
    import jax.numpy as jnp

    phase, count, cursor = states[:, 0], states[:, 1], states[:, 2]
    # Legacy single-token callers and the padded device-bank form share one JIT.
    if starts.ndim == 1:
        starts, ends = starts[:, None], ends[:, None]
    start_lengths, end_lengths = (jnp.sum(x >= 0, axis=1) for x in (starts, ends))
    def advance(markers):
        index = jnp.minimum(cursor, markers.shape[1] - 1)
        expected = jnp.take_along_axis(markers, index[:, None], axis=1)[:, 0]
        return jnp.where(sampled == expected, cursor + 1, (sampled == markers[:, 0]).astype(jnp.int32))
    sc, ec = advance(starts), advance(ends)
    if detector is not None:
        classes, table = detector
        ec = table[jnp.minimum(cursor, table.shape[0]-1), classes[sampled]]
        end_lengths = jnp.full_like(end_lengths, table.shape[0]-1)
    forced = valid & ((phase == FORCING) | ((phase == THINKING) & (count >= budgets)))
    idx = jnp.arange(ends.shape[1])[None, :]
    suffix_idx = jnp.clip(lengths[:, None] - end_lengths[:, None] + idx, 0, MAX_TRANSITION-1)
    suffix = jnp.take_along_axis(transitions, suffix_idx, axis=1)
    ends_transition = jnp.all((idx >= end_lengths[:, None]) | (suffix == ends), axis=1)
    partial_cursor = jnp.where((cursor > 0) & ends_transition, lengths - end_lengths + cursor, 0)
    if detector is not None:
        partial_cursor = jnp.zeros_like(cursor)
    force_cursor = jnp.where(phase == FORCING, cursor, partial_cursor)
    forced_ids = jnp.take_along_axis(transitions, jnp.minimum(force_cursor, MAX_TRANSITION-1)[:, None], axis=1)[:, 0]
    tokens = jnp.where(forced, forced_ids, sampled)
    natural_close = valid & ~forced & (phase == THINKING) & (ec == end_lengths)
    natural_start = valid & ~forced & (phase == WAITING) & (sc == start_lengths)
    next_cursor = force_cursor + 1
    new_phase = jnp.where(forced, jnp.where(next_cursor >= lengths, ANSWERING, FORCING), phase)
    new_phase = jnp.where(natural_close, ANSWERING, jnp.where(natural_start, THINKING, new_phase))
    marker_cursor = jnp.where(phase == WAITING, sc, jnp.where(phase == THINKING, ec, cursor))
    new_cursor = jnp.where(valid, jnp.where(forced, next_cursor,
        jnp.where(natural_close | natural_start, 0, marker_cursor)), cursor)
    new_count = count + (valid & ~forced & (phase == THINKING) &
                         (True if detector is not None else ~natural_close)).astype(jnp.int32)
    return tokens, jnp.stack((new_phase, new_count, new_cursor), axis=1), forced


from .thinking_budget_device import BudgetRunner


def validate_backend(runner):
    """Reject unsupported execution before either fast path can bypass control."""
    if not (runner.speculative_config or getattr(runner, "enable_continue_decode", False)):
        return
    for req_id in runner.input_batch.req_ids:
        params = runner.requests[req_id].sampling_params
        if params and KEY in (params.extra_args or {}):
            raise ValueError("thinking budgets require non-speculative single-step decoding")
