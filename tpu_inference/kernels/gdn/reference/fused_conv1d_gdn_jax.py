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
"""Pure-JAX fused conv1d + GDN with the contract of the GDN v3 kernel.

`fused_conv1d_gdn` here is a drop-in replacement for
`kernels.gdn.v3.wrapper.fused_conv1d_gdn` (same signature, same outputs) that
uses no Pallas. It exists for TPU v4 (16 MiB VMEM), where the v3 kernel halts
TPU cores under serving load; see
`layers.common.attention_backend.GDNLinearAttentionBackend`.

Layout of the computation:
1. Causal depthwise conv1d over the flat token stream. Taps that would cross
   into the previous sequence are zeroed, and the contribution of the conv
   state (the last `kernel_size - 1` inputs before the query) is added to the
   first `kernel_size - 1` tokens of each sequence by a scatter-add.
2. SiLU, q/k L2 norm, q scaling, beta = sigmoid(b) and the log decay
   g = -exp(a_log) * softplus(a + dt_bias), for every token at once.
3. Decode sequences (the first `distribution[0]`, one token each) take one
   recurrent step, vectorized over sequences.
4. Prefill / mixed sequences (`distribution[0]` to `distribution[2]`) run
   the chunked delta rule: a `lax.fori_loop` over the chunks of all those
   sequences (dynamic trip count, so a decode-only batch runs no chunk) with
   the intra-chunk part solved by a unit lower-triangular solve.

Every sequence reads its initial state from `read_state_indices` (zero when
it has no initial state, i.e. `seq_lens == query_lens`) and writes its final
state to `state_indices`. Sequences past `distribution[2]` or with no tokens
write nothing, and tokens outside every sequence give zero output.
"""

import jax
import jax.numpy as jnp
from jax import lax

_HIGHEST = lax.Precision.HIGHEST


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def _l2_norm(x: jax.Array, eps: float = 1e-6) -> jax.Array:
    """Same normalization as the v3 kernel: x / sqrt(sum(x^2) + eps)."""
    return x * lax.rsqrt(jnp.sum(x * x, axis=-1, keepdims=True) + eps)


def _chunk_delta_rule(
        q: jax.Array,  # [chunk, n_v, d_k], normalized and scaled
        k: jax.Array,  # [chunk, n_v, d_k], normalized
        v: jax.Array,  # [chunk, n_v, d_v]
        g: jax.Array,  # [chunk, n_v], log decay
        beta: jax.Array,  # [chunk, n_v]
        mask: jax.Array,  # [chunk], True for the real tokens of the chunk
        h: jax.Array,  # [n_v, d_k, d_v], state before the chunk
) -> tuple[jax.Array, jax.Array]:
    """Gated delta rule over one chunk of one sequence.

    Per token: h = exp(g) * h; delta = beta * (v - k^T h);
    h = h + k delta^T; out = q^T h. Rows past the end of the sequence are
    masked to k = v = q = 0, beta = 0, g = 0, which leaves the state as is.

    Returns the output [chunk, n_v, d_v] and the state after the chunk.
    """
    chunk = q.shape[0]
    mask3 = mask[:, None, None]
    # [n_v, chunk, d]
    q = jnp.where(mask3, q, 0).swapaxes(0, 1)
    k = jnp.where(mask3, k, 0).swapaxes(0, 1)
    v = jnp.where(mask3, v, 0).swapaxes(0, 1)
    # [n_v, chunk]
    g = jnp.where(mask[:, None], g, 0).T
    beta = jnp.where(mask[:, None], beta, 0).T

    g_cum = jnp.cumsum(g, axis=-1)
    # [n_v, chunk, chunk]: g_cum[i] - g_cum[j], the decay from j to i.
    g_diff = g_cum[:, :, None] - g_cum[:, None, :]
    row = lax.broadcasted_iota(jnp.int32, (chunk, chunk), 0)
    col = lax.broadcasted_iota(jnp.int32, (chunk, chunk), 1)
    strictly_lower = row > col
    lower = row >= col
    # Mask before exp: above the diagonal g_diff > 0 and could overflow.
    decay_strict = jnp.where(strictly_lower,
                             jnp.exp(jnp.where(strictly_lower, g_diff, 0)), 0)
    decay_lower = jnp.where(lower, jnp.exp(jnp.where(lower, g_diff, 0)), 0)

    k_beta = k * beta[..., None]
    # (I + strictly_lower(beta_i k_i.k_j decay_ij)) [u | w] =
    #     [beta v | beta k exp(g_cum)]
    kk = jnp.einsum("hik,hjk->hij", k_beta, k, precision=_HIGHEST)
    t = jnp.eye(chunk, dtype=jnp.float32) + kk * decay_strict
    rhs = jnp.concatenate(
        [v * beta[..., None], k_beta * jnp.exp(g_cum)[..., None]], axis=-1)
    uw = lax.linalg.triangular_solve(t,
                                     rhs,
                                     left_side=True,
                                     lower=True,
                                     unit_diagonal=True)
    d_v = v.shape[-1]
    u, w = uw[..., :d_v], uw[..., d_v:]

    # delta_i = beta_i (v_i - k_i^T h_i^-), with h_i^- the decayed state
    # just before token i.
    delta = u - jnp.einsum("hck,hkv->hcv", w, h, precision=_HIGHEST)

    qk = jnp.einsum("hik,hjk->hij", q, k, precision=_HIGHEST) * decay_lower
    out = (jnp.einsum(
        "hck,hkv->hcv", q * jnp.exp(g_cum)[..., None], h, precision=_HIGHEST) +
           jnp.einsum("hij,hjv->hiv", qk, delta, precision=_HIGHEST))

    g_last = g_cum[:, -1]
    k_decayed = k * jnp.exp(g_last[:, None] - g_cum)[..., None]
    h_new = (h * jnp.exp(g_last)[:, None, None] +
             jnp.einsum("hck,hcv->hkv", k_decayed, delta, precision=_HIGHEST))

    return out.swapaxes(0, 1), h_new


@jax.jit(
    donate_argnames=("conv_state", "recurrent_state"),
    static_argnames=(
        "n_kq",
        "n_v",
        "d_k",
        "d_v",
        "kernel_size",
        "num_spec_tokens",
        "decode_tile_size",
        "mixed_tile_size",
        "zero_initialize_out",
        "compute_precision",
    ),
)
def fused_conv1d_gdn(
    qkv: jax.Array,  # [batch_size, n_kq * d_k * 2 + n_v * d_v = dim_size]
    b: jax.Array,  # [batch_size, n_v]
    a: jax.Array,  # [batch_size, n_v]
    conv_state: jax.Array,  # [num_blocks, kernel_size - 1, dim_size]
    recurrent_state: jax.Array,  # [num_blocks, nv, dk, dv]
    conv_weight: jax.Array,  # [dim_size, 1, kernel_size]
    conv_bias: jax.Array | None,  # [dim_size]
    a_log: jax.Array,  # [n_v]
    dt_bias: jax.Array,  # [n_v]
    query_start_loc: jax.Array,  # [num_seqs + 1]
    state_indices: jax.Array,  # [num_seqs]
    distribution: jax.Array,  # [3]
    seq_lens: jax.Array,  # [num_seqs]
    read_state_indices: jax.Array,  # [num_seqs]
    read_offsets: jax.Array | None = None,  # [num_seqs]
    *,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    num_spec_tokens: int = 0,
    zero_initialize_out: bool = True,
    compute_precision: jnp.dtype = jnp.float32.dtype,
    decode_tile_size: int = 4,
    mixed_tile_size: int = 64,
) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
    """Conv1d + GDN in pure JAX, with the contract of the v3 kernel.

    See `kernels.gdn.v3.wrapper.fused_conv1d_gdn` for the arguments. The
    differences:
        - `num_spec_tokens > 0` (speculative verify windows) raises
          NotImplementedError.
        - `mixed_tile_size` is the chunk size of the chunked delta rule for
          prefill / mixed sequences; `decode_tile_size` is unused.
        - Rows outside every sequence are always zero, whatever
          `zero_initialize_out` says.
        - Computation is always float32 (`compute_precision` must be
          float32, as for the v3 kernel).

    Returns:
        (new_conv_state, new_recurrent_state): Updated convolution state cache and
            recurrent state cache tensors.
        out: Output of shape [batch_size, n_v * d_v] in the dtype of qkv.
    """
    if num_spec_tokens > 0:
        raise NotImplementedError(
            "The pure-JAX conv1d + GDN fallback does not support speculative "
            "decoding (num_spec_tokens > 0).")
    if jnp.dtype(compute_precision) != jnp.float32:
        raise NotImplementedError(
            f"compute_precision {compute_precision} is not supported.")
    del decode_tile_size, zero_initialize_out

    f32 = jnp.float32
    i32 = jnp.int32
    act_out_dtype = qkv.dtype
    num_tokens, dim = qkv.shape
    num_seqs = state_indices.size
    num_blocks = conv_state.shape[0]
    prev_k = kernel_size - 1
    key_dim = n_kq * d_k

    # Step 1: Validate inputs.
    assert dim == 2 * key_dim + n_v * d_v
    assert conv_weight.shape == (dim, 1, kernel_size)
    if conv_bias is not None:
        assert conv_bias.shape == (dim, )
    assert conv_state.shape == (num_blocks, prev_k, dim)
    assert recurrent_state.shape == (num_blocks, n_v, d_k, d_v)
    assert query_start_loc.shape == (num_seqs + 1, )
    assert distribution.shape == (3, )
    assert read_state_indices.shape == (num_seqs, )
    assert n_v % n_kq == 0
    if read_offsets is None:
        read_offsets = jnp.zeros((num_seqs, ), dtype=i32)
    assert read_offsets.shape == (num_seqs, )

    # Step 2: Per-sequence metadata.
    seq_ids = jnp.arange(num_seqs, dtype=i32)
    query_start_loc = query_start_loc.astype(i32)
    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    # Same rule as the v3 kernel: the slot holds a state iff some tokens of
    # the sequence were computed before this step.
    has_initial_state = (seq_lens.astype(i32) - query_lens) > 0
    decode_end = distribution[0].astype(i32)
    mixed_end = distribution[2].astype(i32)
    is_valid = (seq_ids < mixed_end) & (query_lens > 0)
    is_decode = seq_ids < decode_end
    is_prefill = is_valid & ~is_decode
    lens = jnp.where(is_valid, query_lens, 0)
    starts = jnp.where(is_valid, query_start_loc[:-1], 0)
    # Like the v3 kernel, only decode sequences apply read_offsets.
    read_slots = (read_state_indices.astype(i32) +
                  jnp.where(is_decode, read_offsets.astype(i32), 0))
    # Out-of-range slots are dropped by the scatters below.
    write_slots = jnp.where(is_valid, state_indices.astype(i32), num_blocks)

    # Token -> sequence. query_start_loc past distribution[2] may hold
    # garbage, so clamp it to a non-decreasing sequence first.
    num_valid_seqs = jnp.minimum(mixed_end, num_seqs)
    loc_ids = jnp.arange(num_seqs + 1, dtype=i32)
    valid_loc = jnp.where(loc_ids <= num_valid_seqs, query_start_loc,
                          query_start_loc[num_valid_seqs])
    token_ids = jnp.arange(num_tokens, dtype=i32)
    token_seq = jnp.clip(
        jnp.searchsorted(valid_loc, token_ids, side="right").astype(i32) - 1,
        0, num_seqs - 1)
    token_pos = token_ids - starts[token_seq]
    token_valid = ((token_ids >= valid_loc[0]) &
                   (token_ids < valid_loc[num_valid_seqs])
                   & is_valid[token_seq])

    # Step 3: Causal depthwise conv1d.
    x = qkv.astype(f32)
    conv_w = conv_weight[:, 0, :].astype(f32)  # [dim, kernel_size]
    # [num_seqs, prev_k, dim]
    init_conv = jnp.where(has_initial_state[:, None, None],
                          conv_state[read_slots].astype(f32), 0)

    # Tap k reads the token `prev_k - k` positions back. Zero the taps that
    # reach before the start of the sequence; the conv state covers them.
    x_padded = jnp.pad(x, ((prev_k, 0), (0, 0)))
    conv_out = jnp.zeros((num_tokens, dim), f32)
    for k in range(kernel_size):
        shift = prev_k - k
        x_tap = x_padded[k:k + num_tokens]
        if shift > 0:
            x_tap = jnp.where((token_pos >= shift)[:, None], x_tap, 0)
        conv_out += x_tap * conv_w[:, k]

    # Conv state contribution to position p < prev_k of each sequence: tap k
    # with shift > p reads state row prev_k + p - shift.
    state_contrib = []
    for p in range(prev_k):
        contrib = jnp.zeros((num_seqs, dim), f32)
        for k in range(kernel_size):
            shift = prev_k - k
            if p < shift:
                contrib += init_conv[:, prev_k + p - shift] * conv_w[:, k]
        state_contrib.append(contrib)
    state_contrib = jnp.stack(state_contrib, axis=1)  # [num_seqs, prev_k, dim]
    pos = jnp.arange(prev_k, dtype=i32)[None, :]
    contrib_rows = jnp.where(pos < lens[:, None], starts[:, None] + pos,
                             num_tokens)
    conv_out = conv_out.at[contrib_rows].add(state_contrib, mode="drop")
    if conv_bias is not None:
        conv_out += conv_bias.astype(f32)

    # New conv state: the last prev_k inputs of [conv state, query tokens].
    src_pos = lens[:, None] + pos - prev_k  # position in the query
    from_tokens = src_pos >= 0
    token_rows = jnp.clip(starts[:, None] + src_pos, 0, num_tokens - 1)
    state_rows = jnp.clip(lens[:, None] + pos, 0, prev_k - 1)
    new_conv = jnp.where(
        from_tokens[..., None], x[token_rows],
        jnp.take_along_axis(init_conv, state_rows[..., None], axis=1))
    new_conv = new_conv.astype(conv_state.dtype)
    conv_state = conv_state.at[write_slots].set(new_conv, mode="drop")

    # Step 4: Activations, q/k normalization and gating.
    act = jax.nn.silu(conv_out)
    v_per_kq = n_v // n_kq
    q = _l2_norm(act[:, :key_dim].reshape(num_tokens, n_kq, d_k))
    q = q * (d_k**-0.5)
    k = _l2_norm(act[:, key_dim:2 * key_dim].reshape(num_tokens, n_kq, d_k))
    q = jnp.repeat(q, v_per_kq, axis=1)  # [num_tokens, n_v, d_k]
    k = jnp.repeat(k, v_per_kq, axis=1)
    v = act[:, 2 * key_dim:].reshape(num_tokens, n_v, d_v)
    beta = jax.nn.sigmoid(b.astype(f32))  # [num_tokens, n_v]
    g = -jnp.exp(a_log.astype(f32)) * jax.nn.softplus(
        a.astype(f32) + dt_bias.astype(f32))

    # Step 5: Decode sequences, one recurrent step each. Each holds exactly
    # one token, so there are at most min(num_seqs, num_tokens) of them.
    chunk_size = max(1, min(mixed_tile_size, num_tokens))
    out = jnp.zeros((num_tokens + chunk_size, n_v, d_v), f32)

    num_decode = min(num_seqs, num_tokens)
    decode_valid = (is_valid & is_decode)[:num_decode]
    decode_rows = jnp.clip(starts[:num_decode], 0, num_tokens - 1)
    h = jnp.where(has_initial_state[:num_decode, None, None, None],
                  recurrent_state[read_slots[:num_decode]].astype(f32), 0)
    q_d, k_d, v_d = q[decode_rows], k[decode_rows], v[decode_rows]
    h = h * jnp.exp(g[decode_rows])[..., None, None]
    k_h = jnp.einsum("shk,shkv->shv", k_d, h, precision=_HIGHEST)
    delta = beta[decode_rows][..., None] * (v_d - k_h)
    h = h + k_d[..., None] * delta[..., None, :]
    out_d = jnp.einsum("shk,shkv->shv", q_d, h, precision=_HIGHEST)
    recurrent_state = recurrent_state.at[jnp.where(
        decode_valid, write_slots[:num_decode],
        num_blocks)].set(h.astype(recurrent_state.dtype), mode="drop")
    out = out.at[jnp.where(decode_valid, decode_rows,
                           num_tokens + chunk_size)].set(out_d, mode="drop")

    # Step 6: Prefill / mixed sequences, chunk by chunk. Each sequence has
    # cdiv(len, chunk_size) chunks, which sum to at most
    # cdiv(num_tokens, chunk_size) + min(num_seqs, num_tokens).
    seq_num_chunks = jnp.where(is_prefill,
                               (lens + chunk_size - 1) // chunk_size, 0)
    max_chunks = _cdiv(num_tokens, chunk_size) + min(num_seqs, num_tokens)
    chunk_seq = jnp.repeat(seq_ids,
                           seq_num_chunks,
                           total_repeat_length=max_chunks)
    seq_first_chunk = jnp.cumulative_sum(seq_num_chunks, include_initial=True)
    chunk_offset = (jnp.arange(max_chunks, dtype=i32) -
                    seq_first_chunk[chunk_seq]) * chunk_size
    num_chunks = seq_num_chunks.sum()

    def pad_tokens(t):
        return jnp.pad(t, ((0, chunk_size), ) + ((0, 0), ) * (t.ndim - 1))

    q_p, k_p, v_p, g_p, beta_p = map(pad_tokens, (q, k, v, g, beta))
    chunk_iota = jnp.arange(chunk_size, dtype=i32)

    def chunk_body(c, carry):
        h, recurrent_state, out = carry
        s = chunk_seq[c]
        offset = chunk_offset[c]
        start = starts[s] + offset
        mask = chunk_iota < lens[s] - offset

        def take(t):
            return lax.dynamic_slice_in_dim(t, start, chunk_size, axis=0)

        h_init = lax.dynamic_index_in_dim(recurrent_state,
                                          read_slots[s],
                                          axis=0,
                                          keepdims=False).astype(f32)
        h_init = jnp.where(has_initial_state[s], h_init, 0)
        h = jnp.where(offset == 0, h_init, h)

        out_c, h = _chunk_delta_rule(take(q_p), take(k_p), take(v_p),
                                     take(g_p), take(beta_p), mask, h)

        # Every chunk writes its state; the last chunk's is the final state.
        recurrent_state = lax.dynamic_update_index_in_dim(
            recurrent_state, h.astype(recurrent_state.dtype),
            state_indices[s].astype(i32), 0)
        out_c = jnp.where(mask[:, None, None], out_c, take(out))
        out = lax.dynamic_update_slice_in_dim(out, out_c, start, axis=0)
        return h, recurrent_state, out

    h0 = jnp.zeros((n_v, d_k, d_v), f32)
    _, recurrent_state, out = lax.fori_loop(0, num_chunks, chunk_body,
                                            (h0, recurrent_state, out))

    out = out[:num_tokens].reshape(num_tokens, n_v * d_v)
    out = jnp.where(token_valid[:, None], out, 0).astype(act_out_dtype)
    return (conv_state, recurrent_state), out
