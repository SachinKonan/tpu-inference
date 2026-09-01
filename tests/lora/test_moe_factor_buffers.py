# Copyright 2026 Google LLC
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
"""CPU tests for replacing, rather than merging, MXFP4 MoE LoRA factors."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np
import pytest

from tpu_inference.runner.tpu_runner import set_moe_lora_factors_in_state

E, H, I, R = 2, 3, 2, 2
H_PAD, I_PAD, R_MAX = 4, 4, 4


def _key(layer, field):
    return (f"vllm_model.model.layers.{layer}.mlp.experts.routed_experts."
            f"tpu_moe_lora_{field}")


def _state():
    state = {
        "immutable.mxfp4.blocks": jnp.arange(12, dtype=jnp.uint8),
    }
    for layer in range(2):
        state.update({
            _key(layer, "gate_a"):
            jnp.ones((H_PAD, R_MAX), jnp.bfloat16),
            _key(layer, "gate_b"):
            jnp.ones((E, R_MAX, I_PAD), jnp.bfloat16),
            _key(layer, "up_a"):
            jnp.ones((H_PAD, R_MAX), jnp.bfloat16),
            _key(layer, "up_b"):
            jnp.ones((E, R_MAX, I_PAD), jnp.bfloat16),
            _key(layer, "down_a"):
            jnp.ones((E, I_PAD, R_MAX), jnp.bfloat16),
            _key(layer, "down_b"):
            jnp.ones((R_MAX, H_PAD), jnp.bfloat16),
            _key(layer, "scale"):
            jnp.asarray(0., jnp.bfloat16),
        })
    return state


def _factors(seed=0):
    rng = np.random.default_rng(seed)
    result = {}
    for layer in range(2):
        for component in ("wi_0", "wi_1"):
            result[f"layers.{layer}.{component}.lora_a"] = rng.normal(
                size=(H, R)).astype(np.float32)
            result[f"layers.{layer}.{component}.lora_b"] = rng.normal(
                size=(R, E, I)).astype(np.float32)
        result[f"layers.{layer}.wo.lora_a"] = rng.normal(size=(E, I,
                                                               R)).astype(
                                                                   np.float32)
        result[f"layers.{layer}.wo.lora_b"] = rng.normal(size=(R, H)).astype(
            np.float32)
    return result


def test_factor_update_preserves_base_and_pads_compile_time_buffers():
    state = _state()
    base_before = np.asarray(state["immutable.mxfp4.blocks"]).copy()
    factors = _factors()
    result = set_moe_lora_factors_in_state(state, factors, {"scale": 0.5})

    assert result == {
        "layers": 2,
        "cleared": False,
        "base_weights_mutated": False,
    }
    np.testing.assert_array_equal(state["immutable.mxfp4.blocks"], base_before)
    for layer in range(2):
        gate_a = np.asarray(state[_key(layer, "gate_a")], dtype=np.float32)
        gate_b = np.asarray(state[_key(layer, "gate_b")], dtype=np.float32)
        expected_a = factors[f"layers.{layer}.wi_0.lora_a"]
        expected_b = factors[f"layers.{layer}.wi_0.lora_b"].transpose(1, 0, 2)
        np.testing.assert_allclose(gate_a[:H, :R],
                                   expected_a,
                                   rtol=1e-2,
                                   atol=1e-2)
        np.testing.assert_allclose(gate_b[:, :R, :I],
                                   expected_b,
                                   rtol=1e-2,
                                   atol=1e-2)
        assert np.count_nonzero(gate_a[H:, :]) == 0
        assert np.count_nonzero(gate_a[:, R:]) == 0
        assert np.count_nonzero(gate_b[:, R:, :]) == 0
        assert np.count_nonzero(gate_b[:, :, I:]) == 0
        assert float(state[_key(layer, "scale")]) == pytest.approx(0.5)


def test_new_adapter_replaces_old_values_and_clear_zeros_every_buffer():
    state = _state()
    first = _factors(1)
    second = _factors(2)
    set_moe_lora_factors_in_state(state, first, {"scale": 1.0})
    set_moe_lora_factors_in_state(state, second, {"scale": 0.25})
    actual = np.asarray(state[_key(0, "down_b")], dtype=np.float32)
    np.testing.assert_allclose(actual[:R, :H],
                               second["layers.0.wo.lora_b"],
                               rtol=1e-2,
                               atol=1e-2)
    assert not np.allclose(
        actual[:R, :H],
        first["layers.0.wo.lora_b"] + second["layers.0.wo.lora_b"])

    result = set_moe_lora_factors_in_state(state, None, {"scale": 0.0})
    assert result["cleared"] is True
    for key, value in state.items():
        if "tpu_moe_lora_" in key:
            assert np.count_nonzero(np.asarray(value)) == 0


def test_rank_mismatch_fails_instead_of_installing_ambiguous_factors():
    state = _state()
    factors = _factors()
    factors["layers.0.wi_0.lora_b"] = np.zeros((R + 1, E, I), dtype=np.float32)
    with pytest.raises(ValueError, match="rank mismatch"):
        set_moe_lora_factors_in_state(state, factors, {"scale": 1.0})
