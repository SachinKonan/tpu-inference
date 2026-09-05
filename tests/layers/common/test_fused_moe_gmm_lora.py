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
"""CPU numeric tests for separate GPT-OSS expert-LoRA GMM factors."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np

from tpu_inference.kernels.megablox.gmm_v2 import apply_act_fn
from tpu_inference.layers.common import fused_moe_gmm
from tpu_inference.layers.common.moe_lora import FusedMoELoRAWeights


def _reference_gmm(lhs, rhs, _scale, _bias, group_sizes, group_offset,
                   **_kwargs):
    """Tiny grouped matmul reference for already expert-sorted rows."""

    sizes = np.asarray(group_sizes)
    offset = int(np.asarray(group_offset).reshape(-1)[0])
    cursor = 0
    chunks = []
    for local_expert in range(rhs.shape[0]):
        rows = int(sizes[offset + local_expert])
        chunks.append(lhs[cursor:cursor + rows] @ rhs[local_expert])
        cursor += rows
    assert cursor == lhs.shape[0]
    return jnp.concatenate(chunks, axis=0)


def _weights():
    # Two experts, rank 2, hidden 3, intermediate 2. Rows 0-1 route to
    # expert 0; row 2 routes to expert 1.
    gate_a = jnp.asarray([[1., 0.], [0., 1.], [1., -1.]])
    gate_b = jnp.asarray([[[1., 2.], [0., 1.]],
                          [[-1., 1.], [2., 0.]]])
    up_a = jnp.asarray([[0., 1.], [1., 0.], [1., 1.]])
    up_b = jnp.asarray([[[1., -1.], [2., 1.]],
                        [[0., 2.], [1., -2.]]])
    down_a = jnp.asarray([[[1., 0.], [0., 1.]],
                          [[2., 1.], [-1., 1.]]])
    down_b = jnp.asarray([[1., 0., 2.], [0., -1., 1.]])
    return FusedMoELoRAWeights(
        gate_a=jnp.stack((gate_a, gate_a * -0.5)),
        gate_b=jnp.stack((gate_b, gate_b + 0.75)),
        up_a=jnp.stack((up_a, up_a * 1.5)),
        up_b=jnp.stack((up_b, up_b - 0.25)),
        down_a=jnp.stack((down_a, down_a + 0.5)),
        down_b=jnp.stack((down_b, down_b * -1.25)),
        scale=jnp.asarray([0.25, 0.5]),
    )


def _expert_indices(group_sizes):
    return np.repeat(np.arange(len(group_sizes)), np.asarray(group_sizes))


def _mixed_w13_reference(x, base, lora, group_sizes, adapter_indices):
    expected = np.asarray(base).copy()
    for row, (expert, slot) in enumerate(
            zip(_expert_indices(group_sizes), np.asarray(adapter_indices))):
        if slot < 0:
            continue
        gate = np.asarray(x[row]) @ np.asarray(lora.gate_a[slot])
        gate = gate @ np.asarray(lora.gate_b[slot, expert])
        up = np.asarray(x[row]) @ np.asarray(lora.up_a[slot])
        up = up @ np.asarray(lora.up_b[slot, expert])
        expected[row] += float(lora.scale[slot]) * np.concatenate((gate, up))
    return expected


def test_w13_lora_is_added_before_gptoss_activation(monkeypatch):
    monkeypatch.setattr(fused_moe_gmm, "gmm_wrapper", _reference_gmm)
    x = jnp.asarray([[1., 2., 3.], [-1., 1., 0.], [2., 0., -1.]])
    base_w13 = jnp.asarray([[0.1, 0.2, 0.3, 0.4], [0.5, -0.2, 0.7, 0.1],
                            [-0.1, 0.8, -0.5, 0.6]])
    sizes = jnp.asarray([2, 1], dtype=jnp.int32)
    offset = jnp.asarray([0], dtype=jnp.int32)
    adapter_indices = jnp.asarray([0, -1, 1], dtype=jnp.int32)
    lora = _weights()

    actual_pre = fused_moe_gmm.apply_moe_lora_w13(
        x, base_w13, lora, adapter_indices, sizes, offset)
    expected_pre = _mixed_w13_reference(x, base_w13, lora, sizes,
                                        adapter_indices)
    np.testing.assert_allclose(actual_pre, expected_pre, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(actual_pre[1], base_w13[1])

    # This specifically catches the invalid ordering activation(base) + delta.
    actual = apply_act_fn(actual_pre, "swigluoai")
    expected = apply_act_fn(jnp.asarray(expected_pre), "swigluoai")
    wrong = (apply_act_fn(base_w13, "swigluoai")
             + 0.5 * (expected_pre - np.asarray(base_w13))[:, :2])
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    assert not np.allclose(actual, wrong)


def test_w2_lora_is_added_before_route_weight_reduction(monkeypatch):
    monkeypatch.setattr(fused_moe_gmm, "gmm_wrapper", _reference_gmm)
    activated = jnp.asarray([[1., 2.], [-1., 0.5], [3., -2.]])
    base_w2 = jnp.asarray([[0.1, 0.2, 0.3], [-0.5, 0.7, 0.9], [1.0, -1.0,
                                                               0.0]])
    sizes = jnp.asarray([2, 1], dtype=jnp.int32)
    offset = jnp.asarray([0], dtype=jnp.int32)
    adapter_indices = jnp.asarray([0, -1, 1], dtype=jnp.int32)
    lora = _weights()

    actual = fused_moe_gmm.apply_moe_lora_w2(
        activated, base_w2, lora, adapter_indices, sizes, offset)
    expected = np.asarray(base_w2).copy()
    for row, (expert, slot) in enumerate(
            zip(_expert_indices(sizes), np.asarray(adapter_indices))):
        if slot < 0:
            continue
        rank = (np.asarray(activated[row])
                @ np.asarray(lora.down_a[slot, expert]))
        expected[row] += (float(lora.scale[slot]) * rank
                          @ np.asarray(lora.down_b[slot]))
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(actual[1], base_w2[1])
