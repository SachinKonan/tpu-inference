# SPDX-License-Identifier: Apache-2.0
"""sample_with_budget must trace: `sample` carries compiler_options, which JAX rejects on a nested jit."""
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.jax.sample.sampling import sample_with_budget
from tpu_inference.layers.jax.sample.sampling_metadata import \
    TPUSupportedSamplingMetadata
from tpu_inference.runner.thinking_budget import MAX_MARKER, MAX_TRANSITION
from tpu_inference.runner.thinking_budget_device import WINDOW, DeviceBank


def test_sample_with_budget_traces_with_compiler_options():
    mesh = Mesh(np.array(jax.devices()[:1]), (ShardingAxisName.ATTN_DATA, ))
    batch, vocab, capacity = 2, 256, 4
    bank = DeviceBank(
        jnp.zeros((capacity, WINDOW, 3), jnp.int32),
        jnp.zeros(capacity, jnp.int32), jnp.zeros(capacity, jnp.int32),
        jnp.full((capacity, MAX_MARKER), -1, jnp.int32),
        jnp.full((capacity, MAX_MARKER), -1, jnp.int32),
        jnp.zeros((capacity, MAX_TRANSITION), jnp.int32),
        jnp.ones(capacity, jnp.int32))
    metadata = TPUSupportedSamplingMetadata(
        temperature=jnp.array([0.7, 0.0], jnp.float32),
        top_k=jnp.array([20, 20], jnp.int32),
        top_p=jnp.array([0.9, 0.9], jnp.float32),
        do_sampling=True,
        logprobs=False,
    )
    logits = jnp.zeros((batch, vocab), jnp.float32)
    with jax.set_mesh(mesh):
        # Lowering traces the whole program (the nested-jit error is a trace-time error); compiling
        # would reject the TPU-only compiler options on CPU.
        sample_with_budget.lower(jax.random.key(0), mesh, logits, metadata, bank,
                                 jnp.array([0, 1], jnp.int32), jnp.array([True, True]))
