# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Numeric tests for the incremental MoE LoRA merge in tpu_runner.

Proves that the incremental merge (subtract-old-delta, add-new-delta;
``apply_moe_lora_deltas_to_state``) is equivalent to a direct full update
from pristine weights, up to bf16 rounding, and that padded regions of the
fused expert tensors are never touched.

Runs on CPU (set ``JAX_PLATFORMS=cpu``); no TPU or model load is needed —
the function under test operates on a plain dict state.
"""

import importlib.util
import os
import sys
from pathlib import Path

# Must be set before jax is first imported (which happens when loading the
# module under test below). Never touch an accelerator from this test.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import ml_dtypes
import numpy as np
import pytest

_REPO_TPU_RUNNER = (Path(__file__).resolve().parents[2] / "tpu_inference" /
                    "runner" / "tpu_runner.py")


def _load_tpu_runner():
    """Load the tpu_runner module under test.

    Prefer the in-repo source sitting next to this test file, so the test
    exercises the checked-out code even on machines whose *installed*
    tpu_inference predates it. Fall back to the installed package (the
    in-repo file's own ``tpu_inference.*`` imports resolve against the
    installed distribution either way).
    """
    if _REPO_TPU_RUNNER.exists():
        spec = importlib.util.spec_from_file_location(
            "_tpu_runner_moe_merge_under_test", str(_REPO_TPU_RUNNER))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return mod
    import tpu_inference.runner.tpu_runner as mod
    return mod


tpu_runner = _load_tpu_runner()

import jax.numpy as jnp  # noqa: E402  (after JAX_PLATFORMS is pinned)

# Small but structurally faithful geometry: padded fused layouts, 2 layers.
E = 4  # experts
H = 16  # hidden
I = 8  # intermediate (per expert)
H_PAD = 32
I_PAD = 32
RANK = 4
NUM_LAYERS = 2
SCALE = 2.0

BF16 = ml_dtypes.bfloat16
# One "rounding unit": bf16 has 8 significand bits (incl. implicit one), so a
# single round-to-nearest costs at most ~2**-8 relative to the magnitude.
BF16_EPS_REL = 2.0**-8

ALL_COMPONENTS = ("wi_0", "wi_1", "wo", "router")


def _w13_key(i):
    return f"vllm_model.model.layers.{i}.mlp.experts.w13_weight"


def _w2_key(i):
    return f"vllm_model.model.layers.{i}.mlp.experts.w2_weight"


def _router_key(i):
    return f"vllm_model.model.layers.{i}.mlp.router.weight"


W13_LAYOUTS = ("3d", "4d")


def _make_state(rng, w13_layout="3d"):
    """bf16 state dict with the padded fused-MoE layouts tpu_runner expects.

    ``w13_layout="3d"`` is the vllm/torchax post-processed layout
    (E, H_pad, 2*I_pad) with uninterleaved [gate | up] halves; "4d" is the
    legacy stacked layout (E, 2, H_pad, I_pad).
    """
    state = {}
    for i in range(NUM_LAYERS):
        w13_shape = ((E, H_PAD, 2 * I_PAD) if w13_layout == "3d" else
                     (E, 2, H_PAD, I_PAD))
        state[_w13_key(i)] = jnp.asarray(rng.standard_normal(w13_shape),
                                         dtype=jnp.bfloat16)
        state[_w2_key(i)] = jnp.asarray(rng.standard_normal((E, I_PAD, H_PAD)),
                                        dtype=jnp.bfloat16)
        state[_router_key(i)] = jnp.asarray(rng.standard_normal((E, H)),
                                            dtype=jnp.bfloat16)
    return state


def _make_factors(rng, components=ALL_COMPONENTS):
    """Flat factor dict keyed layers.{i}.{component}.{lora_a|lora_b} (f32)."""
    factors = {}
    for i in range(NUM_LAYERS):
        for comp in components:
            if comp in ("wi_0", "wi_1"):
                a = rng.standard_normal((H, RANK))
                b = rng.standard_normal((RANK, E, I))
            elif comp == "wo":
                a = rng.standard_normal((E, I, RANK))
                b = rng.standard_normal((RANK, H))
            elif comp == "router":
                a = rng.standard_normal((H, RANK))
                b = rng.standard_normal((RANK, E))
            else:
                raise AssertionError(comp)
            factors[f"layers.{i}.{comp}.lora_a"] = a.astype(np.float32)
            factors[f"layers.{i}.{comp}.lora_b"] = b.astype(np.float32)
    return factors


def _f64(x):
    """Any array (incl. jax bf16) -> numpy float64."""
    return np.asarray(x).astype(np.float64)


def _snapshot(state):
    """Pristine copy of a state dict as numpy bf16 arrays."""
    return {k: np.array(np.asarray(v), dtype=BF16) for k, v in state.items()}


def _naive_delta_f64(component, lora_a, lora_b, scale):
    """Independent f64 reference: explicit loop over experts, np.einsum."""
    a = np.asarray(lora_a, dtype=np.float64)
    b = np.asarray(lora_b, dtype=np.float64)
    if component in ("wi_0", "wi_1"):
        # A (H, r) shared, B (r, E, I) per-expert -> delta (E, H, I).
        return np.stack([
            scale * np.einsum("dr,rf->df", a, b[:, e, :])
            for e in range(b.shape[1])
        ])
    if component == "wo":
        # A (E, I, r) per-expert, B (r, H) shared -> delta (E, I, H).
        return np.stack([
            scale * np.einsum("fr,rd->fd", a[e], b) for e in range(a.shape[0])
        ])
    if component == "router":
        # A (H, r), B (r, E) -> delta (E, H) = (A @ B).T.
        return scale * np.einsum("dr,re->ed", a, b)
    raise AssertionError(component)


def _reference_merged_f64(pristine, factors, scale):
    """Direct full update from pristine weights, entirely in f64.

    Returns {state key: f64 array of the full (padded) shape}. Grouping of
    the flat factor keys is done independently of the implementation.
    """
    ref = {k: _f64(v) for k, v in pristine.items()}
    grouped = {}
    for key, value in factors.items():
        _, i, comp, ab = key.split(".")
        grouped.setdefault((int(i), comp), {})[ab] = value
    for (i, comp), ab in grouped.items():
        delta = _naive_delta_f64(comp, ab["lora_a"], ab["lora_b"], scale)
        if comp in ("wi_0", "wi_1"):
            slot = (tpu_runner.W13_GATE_SLOT
                    if comp == "wi_0" else tpu_runner.W13_UP_SLOT)
            w13 = ref[_w13_key(i)]
            if w13.ndim == 4:
                w13[:, slot, :H, :I] += delta
            else:  # 3-D concat layout: half offset = slot * I_pad.
                off = slot * (w13.shape[-1] // 2)
                w13[:, :H, off:off + I] += delta
        elif comp == "wo":
            ref[_w2_key(i)][:, :I, :H] += delta
        elif comp == "router":
            ref[_router_key(i)] += delta
        else:
            raise AssertionError(comp)
    return ref


def _apply(state, factors, scale, prev_factors=None, prev_scale=None):
    return tpu_runner.apply_moe_lora_deltas_to_state(state, factors,
                                                     {"scale": scale},
                                                     prev_factors, prev_scale)


def _assert_close(state, ref_f64, n_rounding_units, label, eps_ref=None):
    """Assert state matches the f64 reference within n bf16 rounding units.

    ``eps_ref`` optionally supplies additional f64 state dict(s) whose
    magnitudes also bound the rounding unit: bf16 roundings happen at the
    magnitude of each *intermediate* merged state, not just the final one.
    Returns the worst observed error/eps ratio across all state keys.
    """
    worst = 0.0
    for key, want in ref_f64.items():
        got = _f64(state[key])
        magnitude = np.max(np.abs(want))
        for extra in (eps_ref or ()):
            magnitude = max(magnitude, np.max(np.abs(extra[key])))
        eps = magnitude * BF16_EPS_REL
        err = np.max(np.abs(got - want))
        worst = max(worst, err / eps)
        assert err <= n_rounding_units * eps, (
            f"{label}: key {key}: max abs err {err:.3e} exceeds "
            f"{n_rounding_units} * eps = {n_rounding_units * eps:.3e} "
            f"(eps = max|ref| * 2**-8 = {eps:.3e})")
    return worst


def test_delta_math_matches_naive_per_expert_reference():
    """_compute_moe_lora_delta == per-expert scale * A @ B_e (f64 einsum)."""
    rng = np.random.default_rng(0)
    scale = 1.7
    factors = _make_factors(rng)
    for comp in ALL_COMPONENTS:
        a = factors[f"layers.0.{comp}.lora_a"]
        b = factors[f"layers.0.{comp}.lora_b"]
        got = np.asarray(tpu_runner._compute_moe_lora_delta(
            comp, a, b, scale)).astype(np.float64)
        want = _naive_delta_f64(comp, a, b, scale)
        assert got.shape == want.shape, comp
        np.testing.assert_allclose(got,
                                   want,
                                   rtol=1e-5,
                                   atol=1e-5,
                                   err_msg=f"component {comp}")


@pytest.mark.parametrize("w13_layout", W13_LAYOUTS)
def test_single_sync_equals_direct_update(w13_layout):
    """One merge from pristine == W0 + delta within bf16 rounding."""
    rng = np.random.default_rng(1)
    state = _make_state(rng, w13_layout)
    pristine = _snapshot(state)
    factors = _make_factors(rng)

    result = _apply(state, factors, SCALE)
    assert result == {"layers": NUM_LAYERS, "incremental": False}

    ref = _reference_merged_f64(pristine, factors, SCALE)
    # Two roundings (cast the f32 increment to bf16, then the bf16 add),
    # each worth at most ~2 units of max|ref| * 2**-8.
    _assert_close(state, ref, 4, "single sync")


@pytest.mark.parametrize("w13_layout", W13_LAYOUTS)
def test_incremental_three_syncs_equals_direct_from_pristine(w13_layout):
    """THE KEY TEST: F1 -> F2 -> F3 incrementally == pristine + delta(F3).

    The incremental path never sees the pristine weights after sync 1: each
    sync applies (delta_new - delta_old) on top of the already-merged state.
    Equivalence must therefore hold up to bf16 rounding only.
    """
    rng = np.random.default_rng(2)
    state = _make_state(rng, w13_layout)
    pristine = _snapshot(state)
    f1 = _make_factors(rng)
    f2 = _make_factors(rng)
    f3 = _make_factors(rng)

    assert _apply(state, f1, SCALE) == {
        "layers": NUM_LAYERS,
        "incremental": False
    }
    assert _apply(state, f2, SCALE, f1, SCALE) == {
        "layers": NUM_LAYERS,
        "incremental": True
    }
    assert _apply(state, f3, SCALE, f2, SCALE) == {
        "layers": NUM_LAYERS,
        "incremental": True
    }

    # Direct reference: W0 + delta(F3) in f64, then a single cast to bf16.
    ref = _reference_merged_f64(pristine, f3, SCALE)

    worst_ratio = 0.0
    any_bits_differ = False
    for key, want_f64 in ref.items():
        want_bf16 = want_f64.astype(BF16)
        got_bf16 = np.array(np.asarray(state[key]), dtype=BF16)
        err = np.max(np.abs(_f64(got_bf16) - _f64(want_bf16)))
        # Empirical single-cast rounding bound for this tensor.
        eps = np.max(np.abs(want_f64)) * BF16_EPS_REL
        worst_ratio = max(worst_ratio, err / eps)
        assert err <= 4 * eps, (
            f"incremental-vs-direct: key {key}: max abs err {err:.3e} "
            f"exceeds 4 * eps = {4 * eps:.3e} (eps = {eps:.3e})")
        if got_bf16.view(np.uint16).tobytes() != want_bf16.view(
                np.uint16).tobytes():
            any_bits_differ = True
    print(f"\n[incremental-vs-direct] worst max-abs-err / eps ratio over all "
          f"tensors: {worst_ratio:.3f} (allowed 4.0)")

    # Bit-exact equality is NOT expected and NOT required: every sync incurs
    # one bf16 rounding of its increment plus one bf16 rounding of the add,
    # while the direct reference rounds exactly once. Three syncs therefore
    # legitimately drift by a few ULPs from the single-cast result. If this
    # ever becomes bit-identical, the tolerance above is vacuous - flag it.
    assert any_bits_differ, (
        "incremental result is bit-identical to the direct cast; expected "
        "small bf16 rounding differences (one rounding per sync)")


@pytest.mark.parametrize("w13_layout", W13_LAYOUTS)
def test_dropped_component_is_unmerged(w13_layout):
    """Union semantics: a component present in F1 but absent from F2 is
    subtracted back out, leaving its weight at pristine (within rounding)."""
    rng = np.random.default_rng(3)
    state = _make_state(rng, w13_layout)
    pristine = _snapshot(state)
    f1 = _make_factors(rng)  # includes router
    f2 = _make_factors(rng, components=("wi_0", "wi_1", "wo"))  # no router

    _apply(state, f1, SCALE)
    result = _apply(state, f2, SCALE, f1, SCALE)
    assert result == {"layers": NUM_LAYERS, "incremental": True}

    # Router weights must be back at pristine. The tolerance is set by the
    # largest intermediate (W0 + delta(F1)) since the un-merge rounds there.
    ref_after_f1 = _reference_merged_f64(pristine, f1, SCALE)
    for i in range(NUM_LAYERS):
        key = _router_key(i)
        got = _f64(state[key])
        want = _f64(pristine[key])
        eps = np.max(np.abs(ref_after_f1[key])) * BF16_EPS_REL
        err = np.max(np.abs(got - want))
        assert err <= 4 * eps, (
            f"dropped router layer {i}: err {err:.3e} > 4*eps {4 * eps:.3e}")

    # And the surviving components must equal pristine + delta(F2).
    ref = _reference_merged_f64(pristine, f2, SCALE)
    del ref[_router_key(0)], ref[_router_key(1)]
    _assert_close(state, ref, 4, "component drop, surviving components")


@pytest.mark.parametrize("w13_layout", W13_LAYOUTS)
def test_padding_regions_bit_identical_after_syncs(w13_layout):
    """The padded tails of w13 and w2 (E,I_pad,H_pad) must never be
    written: bit-identical to pristine after three syncs."""
    rng = np.random.default_rng(4)
    state = _make_state(rng, w13_layout)
    pristine = _snapshot(state)
    f1 = _make_factors(rng)
    f2 = _make_factors(rng)
    f3 = _make_factors(rng)

    _apply(state, f1, SCALE)
    _apply(state, f2, SCALE, f1, SCALE)
    _apply(state, f3, SCALE, f2, SCALE)

    def bits(x):
        return np.ascontiguousarray(np.array(np.asarray(x),
                                             dtype=BF16)).view(np.uint16)

    for i in range(NUM_LAYERS):
        got13, was13 = bits(state[_w13_key(i)]), bits(pristine[_w13_key(i)])
        if w13_layout == "4d":
            # Rows beyond H and columns beyond I in both w13 slots.
            assert np.array_equal(got13[:, :, H:, :], was13[:, :, H:, :])
            assert np.array_equal(got13[:, :, :, I:], was13[:, :, :, I:])
            # Sanity: the live regions DID change.
            assert not np.array_equal(got13[:, :, :H, :I],
                                      was13[:, :, :H, :I])
        else:
            # 3-D concat layout: rows beyond H, and per-half columns
            # beyond I, are padding.
            assert np.array_equal(got13[:, H:, :], was13[:, H:, :])
            assert np.array_equal(got13[:, :, I:I_PAD], was13[:, :, I:I_PAD])
            assert np.array_equal(got13[:, :, I_PAD + I:],
                                  was13[:, :, I_PAD + I:])
            assert not np.array_equal(got13[:, :H, :I], was13[:, :H, :I])
            assert not np.array_equal(got13[:, :H, I_PAD:I_PAD + I],
                                      was13[:, :H, I_PAD:I_PAD + I])
        got2, was2 = bits(state[_w2_key(i)]), bits(pristine[_w2_key(i)])
        # Rows beyond I and columns beyond H in w2.
        assert np.array_equal(got2[:, I:, :], was2[:, I:, :])
        assert np.array_equal(got2[:, :, H:], was2[:, :, H:])
        assert not np.array_equal(got2[:, :I, :H], was2[:, :I, :H])


def test_factors_as_path_and_nested_lists_match_dict(tmp_path):
    """The three RPC wire forms — safetensors path, dict of ndarrays, dict
    of msgpack-style nested lists — must produce identical merged states."""
    from safetensors.numpy import save_file

    rng = np.random.default_rng(6)
    state_arr = _make_state(rng)
    factors = _make_factors(rng)

    state_path = {k: jnp.array(v) for k, v in state_arr.items()}
    state_list = {k: jnp.array(v) for k, v in state_arr.items()}

    st_file = tmp_path / "moe_lora.safetensors"
    save_file(factors, str(st_file))
    lists = {k: v.tolist() for k, v in factors.items()}

    assert _apply(state_arr, factors, SCALE) == {
        "layers": NUM_LAYERS,
        "incremental": False
    }
    assert _apply(state_path, str(st_file), SCALE) == {
        "layers": NUM_LAYERS,
        "incremental": False
    }
    assert _apply(state_list, lists, SCALE) == {
        "layers": NUM_LAYERS,
        "incremental": False
    }

    for key in state_arr:
        a = np.asarray(state_arr[key]).view(np.uint16)
        b = np.asarray(state_path[key]).view(np.uint16)
        c = np.asarray(state_list[key]).view(np.uint16)
        assert np.array_equal(a, b), f"path form differs at {key}"
        assert np.array_equal(a, c), f"list form differs at {key}"


@pytest.mark.parametrize("w13_layout", W13_LAYOUTS)
def test_scale_change_between_syncs_uses_prev_scale(w13_layout):
    """Sync 2 with a different scale must subtract the old delta at the OLD
    scale (prev_scale path), landing on W0 + delta(F2, new_scale)."""
    rng = np.random.default_rng(5)
    state = _make_state(rng, w13_layout)
    pristine = _snapshot(state)
    f1 = _make_factors(rng)
    f2 = _make_factors(rng)
    old_scale, new_scale = 2.0, 0.5

    _apply(state, f1, old_scale)
    result = _apply(state, f2, new_scale, f1, old_scale)
    assert result == {"layers": NUM_LAYERS, "incremental": True}

    ref = _reference_merged_f64(pristine, f2, new_scale)
    # Sync 1 rounds at the magnitude of the intermediate W0 + delta(F1, 2.0),
    # which (old_scale > new_scale) exceeds the final reference magnitude, so
    # the rounding unit must be based on the larger intermediate as well.
    ref_after_f1 = _reference_merged_f64(pristine, f1, old_scale)
    _assert_close(state, ref, 4, "scale change", eps_ref=(ref_after_f1, ))

    # Discriminator: had the old delta been subtracted at the NEW scale, the
    # state would sit at W0 + delta(F2, new) + delta(F1, old - new), which is
    # far outside the rounding tolerance.
    wrong = _reference_merged_f64(pristine, f2, new_scale)
    wrong_extra = _reference_merged_f64(pristine, f1, old_scale - new_scale)
    key = _w13_key(0)
    wrong_ref = wrong[key] + (wrong_extra[key] - _f64(pristine[key]))
    eps = np.max(np.abs(ref[key])) * BF16_EPS_REL
    assert np.max(np.abs(_f64(state[key]) - wrong_ref)) > 100 * eps
