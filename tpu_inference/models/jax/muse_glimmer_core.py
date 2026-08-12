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
"""Framework-free numerical core for ``meta-models/Muse-Glimmer-30B`` (text only).

Everything in this module depends on **jax/jax.numpy only** -- no ``vllm``, no
``tpu_inference``, no TPU kernels.  That is deliberate:

* ``muse_glimmer.py`` (the tpu-inference serving model) delegates every
  model-specific numeric op to the functions here, so proving these functions
  correct proves the serving numerics correct.
* The CPU parity harness (``tpu/muse_glimmer/parity_check.py``) can import this
  module in a throwaway venv that has jax but neither vllm nor tpu-inference.
* The phase-2 MaxText/tunix port can reuse the same functions verbatim.

Ground truth is ``transformers`` main,
``src/transformers/models/muse_glimmer/modeling_muse_glimmer.py``
(``transformers`` 5.15.0.dev0).  Line references below are to that file.
"""

from __future__ import annotations

import dataclasses
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SLIDING = "sliding_attention"
FULL = "full_attention"


@dataclasses.dataclass(frozen=True)
class MuseGlimmerTextParams:
    """The subset of ``MuseGlimmerTextConfig`` the text stack actually needs.

    Kept as a plain frozen dataclass so the core is importable without
    ``transformers``.  Build one from a real HF config with
    :func:`params_from_hf_config`.
    """

    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    sliding_window: int
    layer_types: Tuple[str, ...]
    layer_rope_theta: Tuple[float, ...]
    rope_theta: float
    rms_norm_eps: float = 1e-5
    # NOTE: distinct from rms_norm_eps -- applies to the two POST norms only.
    post_norm_eps: float = 1e-8
    qk_scale_factor: float = 3.87
    final_logit_softcapping: Optional[float] = 20.0
    output_multiplier: float = 0.19611613513818404

    def __post_init__(self):
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"layer_types has {len(self.layer_types)} entries but "
                f"num_hidden_layers={self.num_hidden_layers}")
        if len(self.layer_rope_theta) != self.num_hidden_layers:
            raise ValueError(
                f"layer_rope_theta has {len(self.layer_rope_theta)} entries but "
                f"num_hidden_layers={self.num_hidden_layers}")
        bad = set(self.layer_types) - {SLIDING, FULL}
        if bad:
            raise ValueError(f"unsupported layer_types {sorted(bad)}")

    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def attn_scaling(self) -> float:
        """The softmax scale.  Separate from, and multiplied on top of, the
        ``qk_scale_factor`` that is folded into q (see :func:`qk_normalize`)."""
        return self.head_dim**-0.5

    def is_sliding(self, layer_idx: int) -> bool:
        return self.layer_types[layer_idx] == SLIDING

    def uses_rope(self, layer_idx: int) -> bool:
        """``layer_rope_theta[i] == 0`` marks a NoPE layer (spec trap 5)."""
        return bool(self.layer_rope_theta[layer_idx])


def params_from_hf_config(text_config: Any) -> MuseGlimmerTextParams:
    """Build :class:`MuseGlimmerTextParams` from a HF ``MuseGlimmerTextConfig``."""
    rope_parameters = getattr(text_config, "rope_parameters", None) or {}
    rope_theta = rope_parameters.get(
        "rope_theta", getattr(text_config, "rope_theta", 10000.0))
    return MuseGlimmerTextParams(
        hidden_size=text_config.hidden_size,
        num_hidden_layers=text_config.num_hidden_layers,
        num_attention_heads=text_config.num_attention_heads,
        num_key_value_heads=text_config.num_key_value_heads,
        head_dim=getattr(
            text_config, "head_dim",
            text_config.hidden_size // text_config.num_attention_heads),
        intermediate_size=text_config.intermediate_size,
        vocab_size=text_config.vocab_size,
        sliding_window=text_config.sliding_window,
        layer_types=tuple(text_config.layer_types),
        layer_rope_theta=tuple(float(t) for t in text_config.layer_rope_theta),
        rope_theta=float(rope_theta),
        rms_norm_eps=float(text_config.rms_norm_eps),
        post_norm_eps=float(text_config.post_norm_eps),
        qk_scale_factor=float(text_config.qk_scale_factor),
        final_logit_softcapping=(
            None if text_config.final_logit_softcapping is None else float(
                text_config.final_logit_softcapping)),
        output_multiplier=float(text_config.output_multiplier),
    )


# ---------------------------------------------------------------------------
# Norms.  Muse-Glimmer ships THREE distinct flavours; mixing them up silently
# produces plausible-but-wrong output.
# ---------------------------------------------------------------------------


def rms_norm_no_scale(x: jax.Array, eps: float) -> jax.Array:
    """``MuseGlimmerRMSNorm(with_scale=False)`` -- parameter free.

    Used by (a) the embedding norm and (b) the per-head q/k norm.  Reference
    computes in float32 and casts back; and deliberately writes
    ``x * pow(mean_sq + eps, -0.5)`` rather than ``rsqrt`` "to address compiler
    differences between Torch and JAX" (modeling_muse_glimmer.py:113-115), so we
    mirror ``pow(..., -0.5)`` here.
    """
    orig_dtype = x.dtype
    xf = x.astype(jnp.float32)
    mean_squared = jnp.mean(jnp.square(xf), axis=-1, keepdims=True) + eps
    return (xf * jnp.power(mean_squared, -0.5)).astype(orig_dtype)


def rms_norm_scaled(x: jax.Array, weight: jax.Array, eps: float) -> jax.Array:
    """``MuseGlimmerRMSNorm(with_scale=True)`` -- weight init ONES, ``n * w``.

    This is the flavour used by the FINAL ``model.language_model.norm`` only.
    It is *not* the ``(1 + w)`` centred variant used inside the decoder layers.
    """
    orig_dtype = x.dtype
    xf = x.astype(jnp.float32)
    mean_squared = jnp.mean(jnp.square(xf), axis=-1, keepdims=True) + eps
    normed = xf * jnp.power(mean_squared, -0.5)
    return (normed * weight.astype(jnp.float32)).astype(orig_dtype)


def centered_rms_norm(x: jax.Array, weight: jax.Array,
                      eps: float) -> jax.Array:
    """``MuseGlimmerTextCenteredRMSNorm`` -- weight init ZEROS, ``n * (1 + w)``.

    The four per-decoder-layer norms.  Reference uses ``rsqrt`` here (unlike
    :func:`rms_norm_no_scale`, which uses ``pow(-0.5)``); see
    modeling_muse_glimmer.py:131.  Both are exact in float32 up to the last ulp.
    """
    orig_dtype = x.dtype
    xf = x.astype(jnp.float32)
    normed = xf * jax.lax.rsqrt(
        jnp.mean(jnp.square(xf), axis=-1, keepdims=True) + eps)
    return (normed * (1.0 + weight.astype(jnp.float32))).astype(orig_dtype)


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------


def rope_cos_sin(positions: jax.Array, head_dim: int,
                 theta: float) -> Tuple[jax.Array, jax.Array]:
    """Half-split ("rotate_half") RoPE tables.

    ``positions``: ``[..., T]``.  Returns ``cos``/``sin`` of shape
    ``[..., T, head_dim // 2]`` in float32.
    """
    inv_freq = 1.0 / (theta**(
        jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    freqs = positions.astype(jnp.float32)[..., None] * inv_freq
    return jnp.cos(freqs), jnp.sin(freqs)


def apply_rope(x: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
    """Apply half-split RoPE to ``x`` of shape ``[..., T, n_heads, head_dim]``.

    ``cos``/``sin`` are ``[..., T, head_dim // 2]``; a head axis is inserted.
    Matches HF ``apply_rotary_pos_emb`` + ``rotate_half`` (which duplicates the
    ``head_dim//2`` frequency table into a full-width ``cat((freqs, freqs))``),
    and matches tpu-inference ``apply_rope(..., rope_input_ordering="split")``.

    Dtype note: ``cos``/``sin`` stay float32 and the rotation promotes, whereas
    HF casts the table down to the activation dtype first
    (``cos.to(dtype=x.dtype)``).  Identical in float32/float64; in bfloat16 this
    is *more* accurate than the reference, and it is what tpu-inference's own
    ``apply_rope`` does, so the serving path and this core agree with each
    other.
    """
    cos = cos[..., None, :]
    sin = sin[..., None, :]
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    out1 = x1 * cos - x2 * sin
    out2 = x2 * cos + x1 * sin
    return jnp.concatenate([out1, out2], axis=-1).astype(x.dtype)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


def qk_normalize(q: jax.Array, k: jax.Array, eps: float,
                 qk_scale_factor: float) -> Tuple[jax.Array, jax.Array]:
    """Per-head, parameter-free q/k norm; ``qk_scale_factor`` on **q only**.

    ``q``/``k`` are ``[..., T, heads, head_dim]``; the norm reduces over
    ``head_dim``.  The scale factor is applied *after* the norm and is entirely
    separate from the ``head_dim ** -0.5`` softmax scaling
    (modeling_muse_glimmer.py:341-342).
    """
    q = rms_norm_no_scale(q, eps) * qk_scale_factor
    k = rms_norm_no_scale(k, eps)
    return q, k


def causal_window_mask(q_positions: jax.Array, kv_positions: jax.Array,
                       sliding_window: Optional[int]) -> jax.Array:
    """Boolean "may attend" mask ``[..., T_q, T_kv]``.

    Causal: ``kv <= q``.  Sliding (matching HF ``sliding_window_overlay``, which
    ANDs ``q_idx - kv_idx < sliding_window`` onto the causal mask): a query
    attends to itself plus the ``sliding_window - 1`` preceding tokens, i.e.
    ``sliding_window`` tokens in total.
    """
    q_pos = q_positions[..., :, None]
    kv_pos = kv_positions[..., None, :]
    allowed = kv_pos <= q_pos
    if sliding_window is not None:
        allowed = jnp.logical_and(allowed, (q_pos - kv_pos) < sliding_window)
    return allowed


def repeat_kv(x: jax.Array, n_rep: int) -> jax.Array:
    """``[..., T, K, H] -> [..., T, K * n_rep, H]`` with each kv head repeated
    ``n_rep`` times *contiguously* (HF ``repeat_kv`` semantics)."""
    if n_rep == 1:
        return x
    return jnp.repeat(x, n_rep, axis=-2)


def dense_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    mask: jax.Array,
    scaling: float,
) -> jax.Array:
    """Reference (non-paged) attention.  ``q``: ``[B, T, N, H]``,
    ``k``/``v``: ``[B, S, K, H]``, ``mask``: ``[B, T, S]`` (bool).

    This is the CPU/eager path used by the parity harness and by anything that
    wants a dependency-free forward.  Serving uses tpu-inference's ragged paged
    attention kernel instead, which implements the same masked softmax.
    """
    n_rep = q.shape[-2] // k.shape[-2]
    k = repeat_kv(k, n_rep)
    v = repeat_kv(v, n_rep)

    # Dtype discipline mirrors HF `eager_attention_forward`
    # (modeling_muse_glimmer.py:274-280): both matmuls run in the activation
    # dtype, and ONLY the softmax is forced to float32 before being cast back.
    # [B, T, S, N]
    logits = jnp.einsum("btnh,bsnh->btsn", q, k) * scaling
    logits = jnp.where(mask[..., None], logits, jnp.finfo(logits.dtype).min)
    probs = jax.nn.softmax(logits.astype(jnp.float32),
                           axis=-2).astype(q.dtype)
    # Rows that are fully masked cannot occur for causal+sliding masks (a query
    # always sees itself), so no renormalisation guard is needed.
    return jnp.einsum("btsn,bsnh->btnh", probs, v)


# ---------------------------------------------------------------------------
# Logits post-processing
# ---------------------------------------------------------------------------


def apply_output_transform(logits: jax.Array,
                           output_multiplier: float,
                           final_logit_softcapping: Optional[float]
                           ) -> jax.Array:
    """``output_multiplier`` **then** tanh softcapping (order matters).

    ``T * tanh(logits * multiplier / T)`` -- modeling_muse_glimmer.py:1182-1185.
    """
    logits = logits * output_multiplier
    if final_logit_softcapping is not None:
        logits = jnp.tanh(logits / final_logit_softcapping)
        logits = logits * final_logit_softcapping
    return logits


# ---------------------------------------------------------------------------
# Weight names.  HF layout is kept verbatim ([out, in] for every Linear) so the
# loader is a pure rename and the parity harness compares like with like.
# ---------------------------------------------------------------------------

#: HF leaf name (under ``layers.{i}.``) -> our per-layer key.
#: NOTE the collision hazard: BOTH ``self_attn.gate_proj`` (the sigmoid
#: attention gate) and ``mlp.gate_proj`` (the SwiGLU gate) exist.  Any mapping
#: keyed on the leaf ``gate_proj`` alone silently aliases them.
LAYER_WEIGHT_MAP: Dict[str, str] = {
    "input_layernorm.weight": "input_layernorm",
    "post_attention_layernorm.weight": "post_attention_layernorm",
    "pre_feedforward_layernorm.weight": "pre_feedforward_layernorm",
    "post_feedforward_layernorm.weight": "post_feedforward_layernorm",
    "self_attn.q_proj.weight": "q_proj",
    "self_attn.k_proj.weight": "k_proj",
    "self_attn.v_proj.weight": "v_proj",
    "self_attn.o_proj.weight": "o_proj",
    "self_attn.gate_proj.weight": "attn_gate_proj",
    "mlp.gate_proj.weight": "mlp_gate_proj",
    "mlp.up_proj.weight": "mlp_up_proj",
    "mlp.down_proj.weight": "mlp_down_proj",
}

#: Prefixes stripped from checkpoint keys before matching.  The 30B checkpoint
#: uses ``model.language_model.*``; a bare ``MuseGlimmerTextModel`` state_dict
#: uses no prefix at all.
_TEXT_PREFIXES = ("model.language_model.", "language_model.", "model.text_model.",
                  "model.", "")

_LAYER_RE = re.compile(r"^layers\.(\d+)\.(.+)$")

#: Checkpoint key prefixes that belong to the (out-of-scope) vision stack.
VISION_KEY_PREFIXES = (
    "model.vision_tower.",
    "model.vision_adapter.",
    "model.vision_projection.",
    "vision_tower.",
    "vision_adapter.",
    "vision_projection.",
)


def is_vision_weight(name: str) -> bool:
    return name.startswith(VISION_KEY_PREFIXES)


def _strip_text_prefix(name: str) -> Optional[str]:
    for prefix in _TEXT_PREFIXES:
        if prefix and name.startswith(prefix):
            return name[len(prefix):]
    return name


def classify_weight(name: str) -> Optional[Tuple[Any, ...]]:
    """Map a checkpoint key to a location in the params pytree.

    Returns ``("embed_tokens",)`` / ``("norm",)`` / ``("lm_head",)`` /
    ``("layers", i, key)``, or ``None`` if the key is not part of the text
    stack (vision, buffers, ...).
    """
    if is_vision_weight(name):
        return None
    if name == "lm_head.weight":
        return ("lm_head", )

    stripped = _strip_text_prefix(name)
    if stripped is None:
        return None
    if stripped == "embed_tokens.weight":
        return ("embed_tokens", )
    if stripped == "norm.weight":
        return ("norm", )

    m = _LAYER_RE.match(stripped)
    if m is None:
        return None
    layer_idx, leaf = int(m.group(1)), m.group(2)
    key = LAYER_WEIGHT_MAP.get(leaf)
    if key is None:
        return None
    return ("layers", layer_idx, key)


REQUIRED_LAYER_KEYS = frozenset(LAYER_WEIGHT_MAP.values())


def load_params(state_dict: Dict[str, Any],
                params: MuseGlimmerTextParams,
                dtype: jnp.dtype = jnp.float32) -> Dict[str, Any]:
    """Build the params pytree from an HF ``state_dict``-like mapping.

    Weights keep their HF ``[out, in]`` orientation.  ``lm_head`` falls back to
    ``embed_tokens`` when the checkpoint omits it (tied-embedding case); the
    30B checkpoint ships a *separate* ``lm_head.weight``, so the fallback is
    only exercised by tied variants.
    """
    out: Dict[str, Any] = {
        "layers": [dict() for _ in range(params.num_hidden_layers)],
    }
    for name, tensor in state_dict.items():
        loc = classify_weight(name)
        if loc is None:
            continue
        array = jnp.asarray(_to_numpy(tensor), dtype=dtype)
        if loc[0] == "layers":
            _, idx, key = loc
            if idx >= params.num_hidden_layers:
                continue
            out["layers"][idx][key] = array
        else:
            out[loc[0]] = array

    if "embed_tokens" not in out:
        raise ValueError("checkpoint has no embed_tokens weight")
    if "norm" not in out:
        raise ValueError("checkpoint has no final norm weight")
    if "lm_head" not in out:
        # Tied embeddings: lm_head shares the (un-normed) embedding matrix.
        out["lm_head"] = out["embed_tokens"]
    for idx, layer in enumerate(out["layers"]):
        missing = REQUIRED_LAYER_KEYS - set(layer)
        if missing:
            raise ValueError(
                f"layer {idx} missing weights: {sorted(missing)}")
    return out


def _to_numpy(tensor: Any):
    if hasattr(tensor, "detach"):  # torch.Tensor
        tensor = tensor.detach().to("cpu")
        # numpy has no bfloat16/float8; upcast only those. Do NOT blanket-call
        # `.float()`, which would silently truncate a float64 tensor and put a
        # float32 floor under any double-precision check.
        if "bfloat16" in str(tensor.dtype) or "float8" in str(tensor.dtype):
            tensor = tensor.float()
        return tensor.numpy()
    return tensor


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------


def embed(weights: Dict[str, Any], input_ids: jax.Array,
          cfg: MuseGlimmerTextParams) -> jax.Array:
    """``MuseGlimmerTextNormedEmbedding``: lookup then parameter-free RMSNorm.

    There is deliberately **no** ``sqrt(hidden_size)`` multiplier here -- unlike
    Gemma, whose embedding scaling must not be copied over (spec trap 9).
    """
    x = jnp.take(weights["embed_tokens"], input_ids, axis=0)
    return rms_norm_no_scale(x, cfg.rms_norm_eps)


def _linear(x: jax.Array, w: jax.Array) -> jax.Array:
    """``x @ w.T`` for an HF-layout ``[out, in]`` weight."""
    return jnp.einsum("...i,oi->...o", x, w)


def attention_block(layer: Dict[str, Any], x: jax.Array,
                    positions: jax.Array, mask: jax.Array, layer_idx: int,
                    cfg: MuseGlimmerTextParams) -> jax.Array:
    """One attention block.  ``x`` is the **post-input_layernorm** activation.

    Returns the o_proj output (pre post_attention_layernorm, pre residual).
    """
    b, t = x.shape[0], x.shape[1]
    n, kvh, h = (cfg.num_attention_heads, cfg.num_key_value_heads,
                 cfg.head_dim)

    q = _linear(x, layer["q_proj"]).reshape(b, t, n, h)
    k = _linear(x, layer["k_proj"]).reshape(b, t, kvh, h)
    v = _linear(x, layer["v_proj"]).reshape(b, t, kvh, h)

    q, k = qk_normalize(q, k, cfg.rms_norm_eps, cfg.qk_scale_factor)

    if cfg.uses_rope(layer_idx):
        cos, sin = rope_cos_sin(positions, cfg.head_dim,
                                cfg.layer_rope_theta[layer_idx])
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

    o = dense_attention(q, k, v, mask, cfg.attn_scaling)
    o = o.reshape(b, t, n * h)

    # Sigmoid gate computed from `x` (the input_layernorm output), NOT from the
    # attention output, and applied BEFORE o_proj (spec trap 4).
    o = o * jax.nn.sigmoid(_linear(x, layer["attn_gate_proj"]))
    return _linear(o, layer["o_proj"])


def mlp_block(layer: Dict[str, Any], x: jax.Array) -> jax.Array:
    gate = jax.nn.silu(_linear(x, layer["mlp_gate_proj"]))
    up = _linear(x, layer["mlp_up_proj"])
    return _linear(gate * up, layer["mlp_down_proj"])


def decoder_layer(layer: Dict[str, Any], h: jax.Array, positions: jax.Array,
                  mask: jax.Array, layer_idx: int,
                  cfg: MuseGlimmerTextParams) -> jax.Array:
    """Sandwich-norm decoder layer.  The two POST norms use ``post_norm_eps``
    (1e-8), the two PRE norms use ``rms_norm_eps`` (1e-5) -- spec trap 6."""
    residual = h
    x = centered_rms_norm(h, layer["input_layernorm"], cfg.rms_norm_eps)
    attn_out = attention_block(layer, x, positions, mask, layer_idx, cfg)
    attn_out = centered_rms_norm(attn_out, layer["post_attention_layernorm"],
                                 cfg.post_norm_eps)
    h = residual + attn_out

    residual = h
    y = centered_rms_norm(h, layer["pre_feedforward_layernorm"],
                          cfg.rms_norm_eps)
    y = mlp_block(layer, y)
    y = centered_rms_norm(y, layer["post_feedforward_layernorm"],
                          cfg.post_norm_eps)
    return residual + y


def build_masks(mask_index: jax.Array,
                cfg: MuseGlimmerTextParams,
                attention_mask: Optional[jax.Array] = None
                ) -> Dict[str, jax.Array]:
    """Per-layer-type masks.  ``mask_index``: ``[B, T]``.  ``attention_mask``:
    optional ``[B, T]`` padding mask (1 = real token).

    ``mask_index`` is deliberately *not* ``positions``: HF builds its causal and
    sliding masks from ``cache_position`` (slot indices), while ``position_ids``
    only feed RoPE.  Keeping them separate is what makes the NoPE layers
    demonstrably position-invariant (spec trap 5) instead of accidentally
    picking the shift up through the mask.
    """
    masks = {
        FULL: causal_window_mask(mask_index, mask_index, None),
        SLIDING:
        causal_window_mask(mask_index, mask_index, cfg.sliding_window),
    }
    if attention_mask is not None:
        keep = attention_mask.astype(bool)[:, None, :]
        masks = {k: jnp.logical_and(v, keep) for k, v in masks.items()}
    return masks


def forward_hidden(weights: Dict[str, Any],
                   input_ids: jax.Array,
                   cfg: MuseGlimmerTextParams,
                   positions: Optional[jax.Array] = None,
                   attention_mask: Optional[jax.Array] = None) -> jax.Array:
    """Full text stack -> final hidden states ``[B, T, D]``.

    Equivalent to HF ``MuseGlimmerTextModel(...).last_hidden_state``.  This is
    the dependency-free *prefill* reference (no KV cache): masks come from the
    token index, ``positions`` only drives RoPE.
    """
    if input_ids.ndim != 2:
        raise ValueError(f"expected [B, T] input_ids, got {input_ids.shape}")
    b, t = input_ids.shape
    mask_index = jnp.broadcast_to(jnp.arange(t), (b, t))
    if positions is None:
        positions = mask_index

    h = embed(weights, input_ids, cfg)
    masks = build_masks(mask_index, cfg, attention_mask)
    for i, layer in enumerate(weights["layers"]):
        h = decoder_layer(layer, h, positions, masks[cfg.layer_types[i]], i,
                          cfg)
    # The FINAL norm is a *scaled* RMSNorm (weight init ones, `n * w`), NOT the
    # centred `(1 + w)` variant used inside the decoder layers.
    return rms_norm_scaled(h, weights["norm"], cfg.rms_norm_eps)


def compute_logits(weights: Dict[str, Any], hidden: jax.Array,
                   cfg: MuseGlimmerTextParams) -> jax.Array:
    logits = _linear(hidden, weights["lm_head"])
    return apply_output_transform(logits, cfg.output_multiplier,
                                  cfg.final_logit_softcapping)


def forward(weights: Dict[str, Any],
            input_ids: jax.Array,
            cfg: MuseGlimmerTextParams,
            positions: Optional[jax.Array] = None,
            attention_mask: Optional[jax.Array] = None
            ) -> Tuple[jax.Array, jax.Array]:
    """Returns ``(hidden_states, logits)``."""
    hidden = forward_hidden(weights, input_ids, cfg, positions, attention_mask)
    return hidden, compute_logits(weights, hidden, cfg)


__all__ = [
    "FULL",
    "SLIDING",
    "MuseGlimmerTextParams",
    "params_from_hf_config",
    "rms_norm_no_scale",
    "rms_norm_scaled",
    "centered_rms_norm",
    "rope_cos_sin",
    "apply_rope",
    "qk_normalize",
    "causal_window_mask",
    "repeat_kv",
    "dense_attention",
    "apply_output_transform",
    "LAYER_WEIGHT_MAP",
    "VISION_KEY_PREFIXES",
    "is_vision_weight",
    "classify_weight",
    "load_params",
    "embed",
    "attention_block",
    "mlp_block",
    "decoder_layer",
    "build_masks",
    "forward_hidden",
    "compute_logits",
    "forward",
]
