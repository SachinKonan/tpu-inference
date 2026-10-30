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
"""The pure-JAX conv1d + GDN fallback against a per-token numpy loop and
against the GDN v3 kernel in TPU interpret mode.

Runs on CPU. The reference below is written from the GDN equations, one
sequence and one token at a time, independently of the fallback's
vectorized conv and chunked delta rule.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src import tpu_info as jax_tpu_info
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh

from tpu_inference.kernels.gdn.reference import fused_conv1d_gdn_jax
from tpu_inference.kernels.gdn.v3 import vmem_ldst
from tpu_inference.kernels.gdn.v3 import wrapper as v3_gdn
from tpu_inference.layers.common import attention_backend, gdn_attention
from tpu_inference.layers.common.sharding import MESH_AXIS_NAMES


@dataclasses.dataclass(frozen=True)
class Dims:
    n_kq: int
    n_v: int
    d_k: int
    d_v: int
    kernel_size: int = 4

    @property
    def dim(self) -> int:
        return 2 * self.n_kq * self.d_k + self.n_v * self.d_v

    def kwargs(self) -> dict[str, int]:
        return dict(n_kq=self.n_kq,
                    n_v=self.n_v,
                    d_k=self.d_k,
                    d_v=self.d_v,
                    kernel_size=self.kernel_size)


# d_k != d_v catches transposed state layouts.
SMALL = Dims(n_kq=2, n_v=4, d_k=16, d_v=8)
# The v3 kernel needs lane-aligned head dims.
LANE_ALIGNED = Dims(n_kq=1, n_v=2, d_k=128, d_v=128)


def reference_conv1d_gdn(qkv, b, a, conv_state, recurrent_state, conv_weight,
                         conv_bias, a_log, dt_bias, query_start_loc,
                         state_indices, distribution, seq_lens,
                         read_state_indices, *, n_kq, n_v, d_k, d_v,
                         kernel_size):
    """Per-sequence, per-token, per-head GDN in float64 numpy."""
    qkv, b, a = (np.asarray(t, np.float64) for t in (qkv, b, a))
    conv_state = np.asarray(conv_state, np.float64)
    recurrent_state = np.asarray(recurrent_state, np.float64)
    dim = qkv.shape[1]
    w = np.asarray(conv_weight, np.float64)[:, 0, :]  # [dim, kernel]
    bias = (np.zeros(dim) if conv_bias is None else np.asarray(
        conv_bias, np.float64))
    a_log = np.asarray(a_log, np.float64)
    dt_bias = np.asarray(dt_bias, np.float64)
    qsl = np.asarray(query_start_loc)
    num_seqs = len(state_indices)

    new_conv = conv_state.copy()
    new_rec = recurrent_state.copy()
    out = np.zeros((qkv.shape[0], n_v * d_v))
    key_dim = n_kq * d_k
    v_per_kq = n_v // n_kq

    for s in range(min(int(distribution[2]), num_seqs)):
        start, end = int(qsl[s]), int(qsl[s + 1])
        query_len = end - start
        if query_len <= 0:
            continue
        has_init = int(seq_lens[s]) - query_len > 0
        read, write = int(read_state_indices[s]), int(state_indices[s])
        window = list(conv_state[read] if has_init else np.zeros((kernel_size -
                                                                  1, dim)))
        state = (recurrent_state[read].copy() if has_init else np.zeros(
            (n_v, d_k, d_v)))
        for t in range(start, end):
            window.append(qkv[t])
            taps = np.stack(window[-kernel_size:])  # oldest first
            y = (taps * w.T).sum(axis=0) + bias
            y = y / (1 + np.exp(-y))
            q = y[:key_dim].reshape(n_kq, d_k)
            k = y[key_dim:2 * key_dim].reshape(n_kq, d_k)
            v = y[2 * key_dim:].reshape(n_v, d_v)
            for h in range(n_v):
                qh = q[h // v_per_kq]
                kh = k[h // v_per_kq]
                qh = qh / np.sqrt((qh * qh).sum() + 1e-6) / np.sqrt(d_k)
                kh = kh / np.sqrt((kh * kh).sum() + 1e-6)
                beta = 1 / (1 + np.exp(-b[t, h]))
                g = -np.exp(a_log[h]) * np.log1p(np.exp(a[t, h] + dt_bias[h]))
                st = state[h] * np.exp(g)
                delta = beta * (v[h] - kh @ st)
                st = st + np.outer(kh, delta)
                out[t, h * d_v:(h + 1) * d_v] = qh @ st
                state[h] = st
        new_conv[write] = np.stack(window[-(kernel_size - 1):])
        new_rec[write] = state
    return (new_conv, new_rec), out


@dataclasses.dataclass
class Batch:
    """A batch: query lengths per real sequence plus padding."""
    lens: list[int]
    distribution: list[int]
    # Tokens already computed before this step; 0 = no initial state.
    context_lens: list[int]
    state_indices: list[int]
    read_state_indices: list[int]
    num_padded_seqs: int = 0
    num_padded_tokens: int = 0
    # Slot the padded sequences point at (must not be written).
    padded_state_index: int = 0


BATCHES = {
    # Five 1-token decodes, one without initial state, one resuming from a
    # different slot than it writes; three padded requests pointing at a
    # real slot, and three padded tokens.
    "decode":
    Batch(lens=[1] * 5,
          distribution=[5, 5, 5],
          context_lens=[6, 2, 0, 9, 1],
          state_indices=[3, 1, 5, 2, 4],
          read_state_indices=[3, 6, 5, 2, 4],
          num_padded_seqs=3,
          num_padded_tokens=3,
          padded_state_index=7),
    # Prefills: fresh, chunked-prefill continuation from another slot,
    # a 2-token continuation (shorter than the conv window), and a fresh
    # one spanning several chunks.
    "prefill":
    Batch(lens=[5, 40, 2, 17],
          distribution=[0, 4, 4],
          context_lens=[0, 20, 2, 0],
          state_indices=[1, 2, 3, 4],
          read_state_indices=[1, 5, 3, 4],
          num_padded_seqs=2,
          num_padded_tokens=6),
    # Decodes (one fresh, one from another slot), prefills, a zero-length
    # request and a mixed request; padded requests on the null block.
    "mixed":
    Batch(lens=[1, 1, 1, 9, 0, 20],
          distribution=[3, 5, 6],
          context_lens=[4, 0, 7, 0, 3, 33],
          state_indices=[6, 2, 3, 4, 5, 1],
          read_state_indices=[6, 2, 7, 4, 5, 8],
          num_padded_seqs=2,
          num_padded_tokens=4,
          padded_state_index=0),
}


def make_inputs(batch: Batch,
                dims: Dims = SMALL,
                dtype=jnp.float32,
                seed=0,
                with_bias=True):
    num_tokens = sum(batch.lens) + batch.num_padded_tokens
    num_blocks = max(batch.state_indices + batch.read_state_indices +
                     [batch.padded_state_index]) + 2
    rng = np.random.default_rng(seed)

    qsl = np.concatenate([[0], np.cumsum(batch.lens)])
    # Garbage (non-monotonic) locations past the real sequences, like the
    # padded batches vLLM sends.
    qsl = np.concatenate([qsl, np.ones(batch.num_padded_seqs, np.int64)])
    seq_lens = np.array(
        [c + n for c, n in zip(batch.context_lens, batch.lens)] +
        [0] * batch.num_padded_seqs)
    pad = [batch.padded_state_index] * batch.num_padded_seqs

    def normal(*shape, scale=1.0):
        return jnp.asarray(rng.normal(size=shape) * scale, jnp.float32)

    return dict(
        qkv=normal(num_tokens, dims.dim).astype(dtype),
        b=normal(num_tokens, dims.n_v).astype(dtype),
        a=normal(num_tokens, dims.n_v).astype(dtype),
        conv_state=normal(num_blocks, dims.kernel_size - 1,
                          dims.dim).astype(dtype),
        recurrent_state=normal(num_blocks,
                               dims.n_v,
                               dims.d_k,
                               dims.d_v,
                               scale=0.5),
        conv_weight=normal(dims.dim, 1, dims.kernel_size, scale=0.5),
        conv_bias=normal(dims.dim, scale=0.1) if with_bias else None,
        a_log=normal(dims.n_v, scale=0.5),
        dt_bias=normal(dims.n_v, scale=0.5),
        query_start_loc=jnp.asarray(qsl, jnp.int32),
        state_indices=jnp.asarray(batch.state_indices + pad, jnp.int32),
        distribution=jnp.asarray(batch.distribution, jnp.int32),
        seq_lens=jnp.asarray(seq_lens, jnp.int32),
        read_state_indices=jnp.asarray(batch.read_state_indices + pad,
                                       jnp.int32),
    )


def copy_states(inputs):
    # The kernels donate the state caches; pass copies.
    return dict(inputs,
                conv_state=jnp.array(inputs["conv_state"], copy=True),
                recurrent_state=jnp.array(inputs["recurrent_state"],
                                          copy=True))


def run_fallback(inputs, dims: Dims = SMALL, **options):
    return fused_conv1d_gdn_jax.fused_conv1d_gdn(**copy_states(inputs),
                                                 **dims.kwargs(), **options)


def assert_close(result, expected, atol=1e-5, rtol=1e-5):
    (conv, rec), out = result
    (exp_conv, exp_rec), exp_out = expected
    for name, actual, want in (("new_conv_state", conv, exp_conv),
                               ("new_recurrent_state", rec,
                                exp_rec), ("output", out, exp_out)):
        np.testing.assert_allclose(np.asarray(actual, np.float64),
                                   np.asarray(want, np.float64),
                                   atol=atol,
                                   rtol=rtol,
                                   err_msg=name)


def reference(inputs, dims: Dims = SMALL):
    return reference_conv1d_gdn(**inputs, **dims.kwargs())


@pytest.mark.parametrize("mixed_tile_size", [8, 64])
@pytest.mark.parametrize("name", list(BATCHES))
def test_matches_reference(name, mixed_tile_size):
    inputs = make_inputs(BATCHES[name])
    assert_close(run_fallback(inputs, mixed_tile_size=mixed_tile_size),
                 reference(inputs))


def test_without_conv_bias():
    inputs = make_inputs(BATCHES["mixed"], with_bias=False)
    assert_close(run_fallback(inputs, mixed_tile_size=8), reference(inputs))


def test_untouched_slots_keep_their_state():
    """Only the write slots of real requests change: not the null block, not
    the slot padded requests point at, not the read-only slots."""
    batch = BATCHES["decode"]
    inputs = make_inputs(batch)
    (conv, rec), out = run_fallback(inputs)
    written = set(batch.state_indices)
    for slot in range(inputs["conv_state"].shape[0]):
        if slot in written:
            continue
        np.testing.assert_array_equal(conv[slot], inputs["conv_state"][slot])
        np.testing.assert_array_equal(rec[slot],
                                      inputs["recurrent_state"][slot])
    np.testing.assert_array_equal(out[sum(batch.lens):], 0)


def test_read_slot_is_used_without_writing_it():
    """A request resuming from a cached block reads that block and writes
    its own; reading the write slot instead gives a different result."""
    inputs = make_inputs(BATCHES["prefill"])
    (_, rec), out = run_fallback(inputs)
    same_slot = dict(inputs, read_state_indices=inputs["state_indices"])
    (_, rec_same), out_same = run_fallback(same_slot)
    # Request 1 (tokens 5..44) reads slot 5 and writes slot 2.
    assert not np.allclose(out[5], out_same[5])
    np.testing.assert_array_equal(out[:5], out_same[:5])
    np.testing.assert_array_equal(rec[5], inputs["recurrent_state"][5])
    assert not np.allclose(rec[2], inputs["recurrent_state"][2])


def test_has_initial_state_ignores_stale_slot():
    """A fresh request (seq_len == query_len) ignores whatever its slot
    holds."""
    inputs = make_inputs(BATCHES["prefill"])
    _, out = run_fallback(inputs)
    garbage = dict(inputs,
                   conv_state=inputs["conv_state"].at[1].set(100.),
                   recurrent_state=inputs["recurrent_state"].at[1].set(100.))
    _, out_garbage = run_fallback(garbage)
    np.testing.assert_array_equal(out[:5], out_garbage[:5])


def test_bf16_keeps_dtypes():
    inputs = make_inputs(BATCHES["mixed"], dtype=jnp.bfloat16)
    inputs["recurrent_state"] = inputs["recurrent_state"].astype(jnp.bfloat16)
    result = run_fallback(inputs, mixed_tile_size=8)
    (conv, rec), out = result
    assert conv.dtype == rec.dtype == out.dtype == jnp.bfloat16
    assert conv.shape == inputs["conv_state"].shape
    assert rec.shape == inputs["recurrent_state"].shape
    assert out.shape == (inputs["qkv"].shape[0], SMALL.n_v * SMALL.d_v)
    assert_close(result, reference(inputs), atol=5e-2, rtol=5e-2)


def test_spec_decode_is_not_supported():
    inputs = make_inputs(BATCHES["decode"])
    num_seqs = inputs["state_indices"].size
    with pytest.raises(NotImplementedError):
        run_fallback(inputs,
                     read_offsets=jnp.zeros((num_seqs, ), jnp.int32),
                     num_spec_tokens=1)


def test_has_no_pallas_call():
    inputs = make_inputs(BATCHES["mixed"])
    jaxpr = jax.make_jaxpr(lambda kw: fused_conv1d_gdn_jax.fused_conv1d_gdn(
        **kw, **SMALL.kwargs()))(inputs)
    assert "pallas_call" not in str(jaxpr)


class _LoadedRef:
    """A loaded VMEM block that reshapes as a value."""

    def __init__(self, value):
        self.value = value
        self.shape = value.shape
        self.dtype = value.dtype

    def reshape(self, *shape):
        return self.value.reshape(*shape)


@pytest.fixture
def v3_interpret(monkeypatch):
    """Run the v3 kernel in TPU interpret mode on CPU, as on a v5p.

    The TPU interpreter of this JAX cannot reshape a ref, which
    vmem_ldst.load_as_qkv_large and load_compact_to_large do (prefill and
    mixed tiles), so those two loads read the block and reshape the value
    instead. They only move data; all compute and state DMA is v3's own.
    """
    info = jax_tpu_info.get_tpu_info_for_chip(jax_tpu_info.ChipVersion.TPU_V5P,
                                              1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)
    for name in ("load_as_qkv_large", "load_compact_to_large"):
        load = getattr(vmem_ldst, name)
        monkeypatch.setattr(
            vmem_ldst,
            name,
            lambda ref, *args, load=load: load(_LoadedRef(ref[...]), *args))
    with pltpu.force_tpu_interpret_mode():
        yield


@pytest.mark.parametrize("name", list(BATCHES))
def test_matches_v3_kernel_in_interpret_mode(v3_interpret, name):
    inputs = make_inputs(BATCHES[name], dims=LANE_ALIGNED)
    v3_result = v3_gdn.fused_conv1d_gdn(**copy_states(inputs),
                                        **LANE_ALIGNED.kwargs(),
                                        mixed_tile_size=16)
    assert_close(v3_result, reference(inputs, LANE_ALIGNED))
    assert_close(run_fallback(inputs, LANE_ALIGNED, mixed_tile_size=16),
                 v3_result)


@pytest.fixture
def mesh() -> Mesh:
    """One CPU device over every axis the shard_map specs name."""
    devices = np.array([jax.devices("cpu")[0]])
    return Mesh(devices.reshape((1, ) * len(MESH_AXIS_NAMES)), MESH_AXIS_NAMES)


@pytest.mark.parametrize("pass_kernel", [True, False])
def test_run_jax_gdn_attention_runs_the_fallback(mesh, monkeypatch,
                                                 pass_kernel):
    """The call site: run_jax_gdn_attention shard-maps the fallback the
    default linear attention backend picks."""
    monkeypatch.setenv("SKYRL_GDN_V4_FALLBACK", "1")
    kernel = attention_backend.GDNLinearAttentionBackend().conv1d_gdn_kernel()
    assert kernel is fused_conv1d_gdn_jax.fused_conv1d_gdn
    inputs = make_inputs(BATCHES["mixed"])
    args = copy_states(inputs)

    result = gdn_attention.run_jax_gdn_attention(
        args["qkv"],
        args["b"],
        args["a"],
        args["conv_state"],
        args["recurrent_state"],
        args["conv_weight"],
        args["conv_bias"],
        args["a_log"],
        args["dt_bias"],
        args["state_indices"],
        args["query_start_loc"],
        args["distribution"],
        args["seq_lens"],
        **SMALL.kwargs(),
        mesh=mesh,
        read_state_indices=args["read_state_indices"],
        conv1d_gdn_kernel=kernel if pass_kernel else None,
    )

    assert_close(result, reference(inputs))


def _tpu_info_with_vmem(vmem_bytes):

    def get_tpu_info():
        if vmem_bytes is None:
            raise ValueError("Unsupported TPU device kind: cpu.")
        return dataclasses.replace(jax_tpu_info.get_tpu_info_for_chip(
            jax_tpu_info.ChipVersion.TPU_V5P, 1),
                                   vmem_capacity_bytes=vmem_bytes)

    return get_tpu_info


@pytest.mark.parametrize(("env", "vmem_mib", "expected"), [
    (None, 16, True),
    (None, 64, False),
    (None, 128, False),
    (None, None, False),
    ("0", 16, False),
    ("1", 64, True),
    ("1", None, True),
    ("", 16, True),
])
def test_fallback_selection(monkeypatch, env, vmem_mib, expected):
    if env is None:
        monkeypatch.delenv("SKYRL_GDN_V4_FALLBACK", raising=False)
    else:
        monkeypatch.setenv("SKYRL_GDN_V4_FALLBACK", env)
    monkeypatch.setattr(
        pltpu, "get_tpu_info",
        _tpu_info_with_vmem(None if vmem_mib is None else vmem_mib * 2**20))

    assert attention_backend.use_gdn_jax_fallback() is expected
    kernel = attention_backend.GDNLinearAttentionBackend().conv1d_gdn_kernel()
    assert kernel is (fused_conv1d_gdn_jax.fused_conv1d_gdn
                      if expected else v3_gdn.fused_conv1d_gdn)
