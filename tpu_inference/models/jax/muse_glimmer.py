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
"""JAX-native, **text-only** ``meta-models/Muse-Glimmer-30B``.

Architecture (see ``muse_glimmer_core`` for the numerics and for the reference
line numbers):

* ``NormedEmbedding``  -- embedding lookup followed by a parameter-free
  RMSNorm.  Unlike Gemma there is **no** ``sqrt(hidden_size)`` scaling.
* 52 sandwich-norm decoder layers.  The two PRE norms use ``rms_norm_eps``
  (1e-5), the two POST norms use ``post_norm_eps`` (1e-8), and all four are the
  *centred* ``n * (1 + w)`` variant with zero-initialised weights.
* GQA attention (32 q heads / 2 kv heads / head_dim 128) with a parameter-free
  per-head q/k norm; ``qk_scale_factor`` (3.87) multiplies **q only** and is
  separate from the ``head_dim ** -0.5`` softmax scale.
* A sigmoid gate ``sigmoid(gate_proj(x))`` where ``x`` is the *input_layernorm
  output*, applied elementwise to the attention output **before** ``o_proj``.
* Hybrid attention: ``layer_types`` alternates [S, S, S, F] x 13; the 13 full
  layers carry ``layer_rope_theta == 0`` and are therefore **NoPE**.
* The final ``model.norm`` is a *scaled* RMSNorm (``n * w``, ones-init), not the
  centred variant.
* Logits: ``lm_head`` -> ``* output_multiplier`` -> tanh softcap at 20.0.

The vision tower, projector and image/video token handling are out of scope;
the loader drops vision weights and :meth:`MuseGlimmerForConditionalGeneration`
rejects multimodal inputs with a clear error.
"""

from typing import Any, Iterable, List, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh
from vllm.config import VllmConfig

from tpu_inference import utils
from tpu_inference.distributed.jax_parallel_state import get_pp_group
from tpu_inference.layers.common.attention_interface import attention
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.common.quantization import quantize_kv
from tpu_inference.layers.jax import JaxModule
from tpu_inference.layers.jax.embed import JaxEmbed
from tpu_inference.layers.jax.linear import JaxEinsum, JaxLinear, JaxLmHead
from tpu_inference.layers.jax.pp_utils import PPMissingLayer, make_layers
from tpu_inference.layers.jax.rope_interface import apply_rope
from tpu_inference.layers.vllm.quantization.configs import VllmQuantConfig
from tpu_inference.logger import init_logger
from tpu_inference.models.jax.jax_intermediate_tensor import \
    JaxIntermediateTensors
from tpu_inference.models.jax.muse_glimmer_core import (SLIDING,
                                                        apply_output_transform,
                                                        centered_rms_norm,
                                                        is_vision_weight,
                                                        qk_normalize,
                                                        rms_norm_no_scale,
                                                        rms_norm_scaled)
from tpu_inference.models.jax.utils.weight_utils import (LoadableWithIterator,
                                                         StandardWeightLoader)

logger = init_logger(__name__)

init_fn = nnx.initializers.uniform()


def _text_config(hf_config: Any) -> Any:
    """Muse-Glimmer's checkpoint config is the composite (vision + text) one."""
    return getattr(hf_config, "text_config", hf_config)


# ---------------------------------------------------------------------------
# Norms
# ---------------------------------------------------------------------------


class MuseGlimmerCenteredRmsNorm(JaxModule):
    """``MuseGlimmerTextCenteredRMSNorm`` -- ``n * (1 + w)``, weights zero-init.

    Cannot reuse ``JaxRmsNorm``/``nnx.RMSNorm``: those compute ``n * w``, which
    for this checkpoint's zero-centred weights collapses the activation to ~0.
    """

    def __init__(self,
                 hidden_size: int,
                 epsilon: float,
                 dtype: jnp.dtype,
                 rngs: nnx.Rngs,
                 prefix: str = ""):
        self.epsilon = epsilon
        self.weight = nnx.Param(
            nnx.with_partitioning(nnx.initializers.zeros_init(),
                                  (None, ))(rngs.params(), (hidden_size, ),
                                            dtype),
            eager_sharding=False,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return centered_rms_norm(x, self.weight.get_value(), self.epsilon)


class MuseGlimmerScaledRmsNorm(JaxModule):
    """``MuseGlimmerRMSNorm(with_scale=True)`` -- ``n * w``, weights ones-init.

    Only the final ``model.norm`` uses this flavour.  It differs from
    ``JaxRmsNorm`` only in using ``pow(mean_sq + eps, -0.5)`` instead of
    ``rsqrt``, which the reference does deliberately for Torch/JAX agreement.
    """

    def __init__(self,
                 hidden_size: int,
                 epsilon: float,
                 dtype: jnp.dtype,
                 rngs: nnx.Rngs,
                 prefix: str = ""):
        self.epsilon = epsilon
        self.weight = nnx.Param(
            nnx.with_partitioning(nnx.initializers.ones_init(),
                                  (None, ))(rngs.params(), (hidden_size, ),
                                            dtype),
            eager_sharding=False,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return rms_norm_scaled(x, self.weight.get_value(), self.epsilon)


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------


class MuseGlimmerMLP(JaxModule):
    """SwiGLU MLP.

    NOTE: ``gate_proj``/``up_proj`` are deliberately *not* fused into a merged
    column-parallel linear.  ``packed_modules_mapping`` routing matches on the
    bare substring ``gate_proj``, which would also catch the attention gate
    (``self_attn.gate_proj``).  Keeping them unfused removes the hazard; fusing
    them later requires a name-qualified routing rule.
    """

    def __init__(self,
                 config: Any,
                 dtype: jnp.dtype,
                 rng: nnx.Rngs,
                 quant_config: VllmQuantConfig,
                 prefix: str = ""):
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size

        self.gate_proj = JaxLinear(
            hidden_size,
            intermediate_size,
            use_bias=False,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model")),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".gate_proj",
        )
        self.up_proj = JaxLinear(
            hidden_size,
            intermediate_size,
            use_bias=False,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model")),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".up_proj",
        )
        self.down_proj = JaxLinear(
            intermediate_size,
            hidden_size,
            use_bias=False,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, ("model", None)),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".down_proj",
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.down_proj(jax.nn.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class MuseGlimmerAttention(JaxModule):

    def __init__(self,
                 config: Any,
                 layer_idx: int,
                 dtype: jnp.dtype,
                 rng: nnx.Rngs,
                 mesh: Mesh,
                 kv_cache_dtype: str,
                 quant_config: VllmQuantConfig,
                 prefix: str = ""):
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.rms_norm_eps = config.rms_norm_eps
        self.qk_scale_factor = float(config.qk_scale_factor)

        self.head_dim_original = getattr(
            config, "head_dim", self.hidden_size // config.num_attention_heads)
        self.head_dim = utils.get_padded_head_dim(self.head_dim_original)

        # The softmax scale is `head_dim ** -0.5`; `qk_scale_factor` is applied
        # to q separately, before the kernel.
        self.scaling = self.head_dim_original**-0.5

        self.layer_type = config.layer_types[layer_idx]
        self.is_sliding = self.layer_type == SLIDING
        self.sliding_window = config.sliding_window if self.is_sliding else None

        # `layer_rope_theta[i] == 0` marks a NoPE layer: no rotation at all.
        layer_theta = float(config.layer_rope_theta[layer_idx])
        self.uses_rope = bool(layer_theta)
        self.rope_theta = layer_theta
        # CAREFUL: on transformers v5 configs `config.rope_scaling` is an alias
        # for `rope_parameters`, i.e. a *truthy* dict
        # `{"rope_theta": ..., "rope_type": "default"}`.  Passing that straight
        # to `apply_rope` makes it take the `if rope_scaling:` branch and apply
        # llama3-style frequency rescaling (scale_factor 8.0 by default) to a
        # model that asks for plain RoPE.  Only forward a scaling dict when the
        # rope type actually is a scaled one.
        rope_parameters = getattr(config, "rope_parameters", None) or {}
        rope_type = rope_parameters.get("rope_type", "default")
        self.rope_scaling = None if rope_type in (None,
                                                  "default") else rope_parameters

        sharding_size = mesh.shape["model"]
        padded_num_heads = utils.get_padded_num_heads(self.num_heads,
                                                      sharding_size)
        padded_num_kv_heads = utils.get_padded_num_heads(
            self.num_kv_heads, sharding_size)
        self.num_heads = padded_num_heads
        self.num_kv_heads = padded_num_kv_heads
        self.mesh = mesh

        self.q_proj = JaxEinsum(
            "TD,DNH->TNH",
            (self.hidden_size, self.num_heads, self.head_dim),
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model", None)),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".q_proj",
        )
        self.k_proj = JaxEinsum(
            "TD,DKH->TKH",
            (self.hidden_size, self.num_kv_heads, self.head_dim),
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model", None)),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".k_proj",
        )
        self.v_proj = JaxEinsum(
            "TD,DKH->TKH",
            (self.hidden_size, self.num_kv_heads, self.head_dim),
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model", None)),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".v_proj",
        )
        self.o_proj = JaxEinsum(
            "TNH,NHD->TD",
            (self.num_heads, self.head_dim, self.hidden_size),
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, ("model", None, None)),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".o_proj",
        )
        # Sigmoid gate over heads*head_dim, computed from the layer input.
        # Sharded on the output axis so it lines up with q_proj/o_proj's head
        # sharding (the product is elementwise over N*H).
        #
        # NAMING: the checkpoint calls this `self_attn.gate_proj`, but the JAX
        # module is deliberately named `attn_gate_proj` and `load_weights`
        # renames the incoming key. Reason: vLLM's LoRA manager matches
        # `target_modules` against the *last dotted component* of a module path
        # (`re.match(rf".*\.{target}$", name)`), so an adapter that targets the
        # ordinary SwiGLU `gate_proj` would also latch onto this attention gate
        # -- a different tensor with a different shape and no adapter weights.
        # The same suffix collision would hit `packed_modules_mapping` routing.
        # Renaming the module makes `gate_proj` unambiguously the MLP gate.
        self.attn_gate_proj = JaxLinear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            use_bias=False,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model")),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".attn_gate_proj",
        )

        self._q_scale = 1.0
        self._k_scale = 1.0
        self._v_scale = 1.0
        self.kv_cache_quantized_dtype = None
        if kv_cache_dtype != "auto":
            self.kv_cache_quantized_dtype = utils.get_jax_dtype_from_str_dtype(
                kv_cache_dtype)

    def _qk_norm(self, q: jax.Array, k: jax.Array):
        """Parameter-free per-head RMSNorm over the *unpadded* head_dim, with
        ``qk_scale_factor`` on q only."""
        hd = self.head_dim_original
        if self.head_dim == hd:
            return qk_normalize(q, k, self.rms_norm_eps, self.qk_scale_factor)
        # head_dim was padded for the kernel: normalise only the real slice and
        # leave the padding untouched (it is masked out downstream anyway).
        q_real, k_real = qk_normalize(q[..., :hd], k[..., :hd],
                                      self.rms_norm_eps, self.qk_scale_factor)
        q = q.at[..., :hd].set(q_real.astype(q.dtype))
        k = k.at[..., :hd].set(k_real.astype(k.dtype))
        return q, k

    def __call__(
        self,
        kv_cache: Optional[jax.Array],
        x: jax.Array,
        attention_metadata: AttentionMetadata,
    ) -> Tuple[jax.Array, jax.Array]:
        md = attention_metadata

        q = self.q_proj(x)  # (T, N, H)
        k = self.k_proj(x)  # (T, K, H)
        v = self.v_proj(x)  # (T, K, H)

        q, k = self._qk_norm(q, k)
        q = q.astype(x.dtype)
        k = k.astype(x.dtype)

        if self.uses_rope:
            q = apply_rope(q, md.input_positions, self.head_dim_original,
                           self.rope_theta, self.rope_scaling)
            k = apply_rope(k, md.input_positions, self.head_dim_original,
                           self.rope_theta, self.rope_scaling)

        q_scale = k_scale = v_scale = None
        if self.kv_cache_quantized_dtype:
            k_scale = self._k_scale
            v_scale = self._v_scale
            k, v = quantize_kv(self.kv_cache_quantized_dtype, k, v, k_scale,
                               v_scale)

        new_kv_cache, outputs = attention(
            kv_cache,
            q,
            k,
            v,
            attention_metadata,
            self.mesh,
            self.head_dim_original,
            sm_scale=self.scaling,
            attention_chunk_size=self.sliding_window,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
        )

        # Gate from the *layer input* `x` (post input_layernorm), elementwise
        # over N*H, applied BEFORE o_proj.
        gate = jax.nn.sigmoid(self.attn_gate_proj(x))
        outputs = outputs.reshape(outputs.shape[0], -1) * gate
        outputs = outputs.reshape(-1, self.num_heads, self.head_dim)

        o = self.o_proj(outputs)  # (T, D)
        return new_kv_cache, o


# ---------------------------------------------------------------------------
# Decoder layer / model
# ---------------------------------------------------------------------------


class MuseGlimmerDecoderLayer(JaxModule):

    def __init__(self,
                 config: Any,
                 layer_idx: int,
                 dtype: jnp.dtype,
                 rng: nnx.Rngs,
                 mesh: Mesh,
                 kv_cache_dtype: str,
                 quant_config: VllmQuantConfig,
                 prefix: str = ""):
        hidden_size = config.hidden_size
        rms_norm_eps = config.rms_norm_eps
        post_norm_eps = config.post_norm_eps

        self.input_layernorm = MuseGlimmerCenteredRmsNorm(
            hidden_size, rms_norm_eps, dtype, rng,
            prefix + ".input_layernorm")
        self.self_attn = MuseGlimmerAttention(
            config=config,
            layer_idx=layer_idx,
            dtype=dtype,
            rng=rng,
            mesh=mesh,
            kv_cache_dtype=kv_cache_dtype,
            quant_config=quant_config,
            prefix=prefix + ".self_attn",
        )
        self.post_attention_layernorm = MuseGlimmerCenteredRmsNorm(
            hidden_size, post_norm_eps, dtype, rng,
            prefix + ".post_attention_layernorm")
        self.pre_feedforward_layernorm = MuseGlimmerCenteredRmsNorm(
            hidden_size, rms_norm_eps, dtype, rng,
            prefix + ".pre_feedforward_layernorm")
        self.mlp = MuseGlimmerMLP(config=config,
                                  dtype=dtype,
                                  rng=rng,
                                  quant_config=quant_config,
                                  prefix=prefix + ".mlp")
        self.post_feedforward_layernorm = MuseGlimmerCenteredRmsNorm(
            hidden_size, post_norm_eps, dtype, rng,
            prefix + ".post_feedforward_layernorm")

    def __call__(
        self,
        kv_cache: Optional[jax.Array],
        x: jax.Array,
        attention_metadata: AttentionMetadata,
    ) -> Tuple[jax.Array, jax.Array]:
        residual = x
        h = self.input_layernorm(x)
        kv_cache, h = self.self_attn(kv_cache, h, attention_metadata)
        h = self.post_attention_layernorm(h)
        x = residual + h

        residual = x
        h = self.pre_feedforward_layernorm(x)
        h = self.mlp(h)
        h = self.post_feedforward_layernorm(h)
        return kv_cache, residual + h


class MuseGlimmerModel(JaxModule):
    """The text stack.  Named ``model`` so checkpoint keys line up after the
    ``language_model.`` segment is stripped by the loader."""

    def __init__(self,
                 vllm_config: VllmConfig,
                 rng: nnx.Rngs,
                 mesh: Mesh,
                 prefix: str = "model") -> None:
        model_config = vllm_config.model_config
        config = _text_config(model_config.hf_config)
        dtype = model_config.dtype
        hidden_size = config.hidden_size

        self.is_first_rank = get_pp_group().is_first_rank
        self.is_last_rank = get_pp_group().is_last_rank

        vocab_size = model_config.get_vocab_size()
        tp_size = (vllm_config.parallel_config.tensor_parallel_size
                   if vllm_config.parallel_config is not None else 1)
        padded_vocab_size = utils.align_to(vocab_size, tp_size)

        if self.is_first_rank:
            self.embed_tokens = JaxEmbed(
                num_embeddings=padded_vocab_size,
                features=hidden_size,
                param_dtype=dtype,
                embedding_init=nnx.with_partitioning(init_fn,
                                                     ("model", None)),
                rngs=rng,
                quant_config=vllm_config.quant_config,
                prefix=prefix + ".embed_tokens",
            )
            # Parameter-free RMSNorm applied on top of the embedding lookup
            # (`MuseGlimmerTextNormedEmbedding`).  NOT a sqrt(d) scale.
            self.embed_norm_eps = config.rms_norm_eps
        else:
            self.embed_tokens = PPMissingLayer()
            self.embed_norm_eps = config.rms_norm_eps

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda layer_index: MuseGlimmerDecoderLayer(
                config=config,
                layer_idx=layer_index,
                dtype=dtype,
                rng=rng,
                mesh=mesh,
                kv_cache_dtype=vllm_config.cache_config.cache_dtype,
                quant_config=vllm_config.quant_config,
                prefix=f"{prefix}.layers.{layer_index}",
            ))

        if self.is_last_rank:
            self.norm = MuseGlimmerScaledRmsNorm(hidden_size,
                                                 config.rms_norm_eps, dtype,
                                                 rng, prefix + ".norm")
        else:
            self.norm = PPMissingLayer()

    def __call__(
        self,
        kv_caches: List[jax.Array],
        input_ids: Optional[jax.Array],
        attention_metadata: AttentionMetadata,
        inputs_embeds: Optional[jax.Array] = None,
    ) -> Tuple[List[jax.Array], jax.Array]:
        if inputs_embeds is not None:
            # Already normed by the caller (the PP-rank hand-off passes final
            # hidden states, not raw embeddings).
            x = inputs_embeds
        else:
            x = self.embed_tokens(input_ids)
            # `MuseGlimmerTextNormedEmbedding`: parameter-free RMSNorm on top of
            # the lookup. NOT a sqrt(hidden_size) scale (spec trap 9).
            x = rms_norm_no_scale(x, self.embed_norm_eps)

        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            kv_caches[i], x = layer(kv_caches[i], x, attention_metadata)

        if self.is_last_rank:
            x = self.norm(x)
        return kv_caches, x


class MuseGlimmerForConditionalGeneration(JaxModule, LoadableWithIterator):
    """Text-only Muse-Glimmer.

    The checkpoint's ``architectures`` entry is
    ``MuseGlimmerForConditionalGeneration`` (the multimodal class), so this must
    carry that name to match; image/video inputs are rejected explicitly.
    """

    WeightLoader = StandardWeightLoader

    # Deliberately empty: `gate_proj`/`up_proj` are NOT fused into a merged
    # column-parallel linear. Fusion would (a) require a routing rule keyed on
    # the bare substring `gate_proj`, which also matches the attention gate,
    # and (b) force every LoRA adapter through `packed_modules_mapping` shard
    # splitting. Keeping them separate makes both the loader and LoRA
    # targeting unambiguous, at the cost of one extra matmul per layer.
    packed_modules_mapping: dict = {}

    def __init__(self, vllm_config: VllmConfig, rng_key: jax.Array,
                 mesh: Mesh) -> None:
        self.vllm_config = vllm_config
        rng = nnx.Rngs(rng_key)
        self.mesh = mesh

        model_config = vllm_config.model_config
        config = _text_config(model_config.hf_config)

        self.model = MuseGlimmerModel(vllm_config=vllm_config,
                                      rng=rng,
                                      mesh=mesh,
                                      prefix="model")

        self.output_multiplier = float(getattr(config, "output_multiplier",
                                               1.0))
        self.final_logit_softcapping = getattr(config,
                                               "final_logit_softcapping", None)
        if self.final_logit_softcapping is not None:
            self.final_logit_softcapping = float(self.final_logit_softcapping)

        # The 30B checkpoint stores a *separate* `lm_head.weight` and sets
        # `text_config.tie_word_embeddings = false`; the tied path exists only
        # for variants that omit the tensor.
        self.tie_word_embeddings = bool(
            getattr(config, "tie_word_embeddings", False))
        if not self.tie_word_embeddings and self.model.is_last_rank:
            vocab_size = model_config.get_vocab_size()
            tp_size = (vllm_config.parallel_config.tensor_parallel_size
                       if vllm_config.parallel_config is not None else 1)
            self.lm_head = JaxLmHead(
                hidden_size=config.hidden_size,
                vocab_size=utils.align_to(vocab_size, tp_size),
                param_dtype=model_config.dtype,
                # Shard the CONTRACTING (hidden) axis, as gemma4_mm does: the
                # 202048 x 6656 head is far too big to replicate, and this
                # layout leaves the [T, V] logits replicated so the sampler
                # sees full rows without a gather. Sharding the vocab axis
                # instead would halve the all-reduce but hand the sampler a
                # vocab-sharded output.
                kernel_init=nnx.with_partitioning(init_fn, ("model", None)),
                rngs=rng,
                prefix="lm_head",
            )
        elif not self.tie_word_embeddings:
            self.lm_head = PPMissingLayer()

    # -- weight loading ----------------------------------------------------

    def load_weights(self, weights: Iterable[Tuple[str, Any]]):
        """Drop the vision stack and flatten ``model.language_model.*`` down to
        ``model.*`` so the auto-loader's module walk lines up.

        Also renames ``self_attn.gate_proj`` -> ``self_attn.attn_gate_proj`` to
        match the module tree; see ``MuseGlimmerAttention.attn_gate_proj`` for
        why that module is not called ``gate_proj``.
        """
        allowed_layers = tuple(f"layers.{i}."
                               for i in range(len(self.model.layers)))

        def _filtered():
            for name, tensor in weights:
                if is_vision_weight(name):
                    continue
                clean = name.replace("language_model.", "")
                clean = clean.replace("self_attn.gate_proj",
                                      "self_attn.attn_gate_proj")
                if not clean.startswith(("model.", "lm_head")):
                    continue
                if "layers." in clean and not any(p in clean
                                                  for p in allowed_layers):
                    continue
                yield clean, tensor

        return super().load_weights(_filtered())

    # -- forward -----------------------------------------------------------

    def __call__(
        self,
        kv_caches: List[jax.Array],
        input_ids: jax.Array,
        attention_metadata: AttentionMetadata,
        inputs_embeds: Optional[jax.Array] = None,
        _input_positions=None,
        _layer_name_to_kv_cache=None,
        _lora_metadata=None,
        intermediate_tensors: JaxIntermediateTensors | None = None,
        is_first_rank: bool = True,
        is_last_rank: bool = True,
        *args,
    ) -> Tuple[List[jax.Array], jax.Array | JaxIntermediateTensors,
               List[jax.Array], Optional[jax.Array]]:
        if not is_first_rank:
            assert intermediate_tensors is not None
            inputs_embeds = intermediate_tensors["hidden_states"]

        kv_caches, x = self.model(kv_caches, input_ids, attention_metadata,
                                  inputs_embeds)

        if not is_last_rank:
            x = JaxIntermediateTensors(tensors={"hidden_states": x})
        return kv_caches, x, [], None

    def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
        if hasattr(self, "lm_head") and not isinstance(self.lm_head,
                                                       PPMissingLayer):
            logits = self.lm_head(hidden_states)
        else:
            logits = self.model.embed_tokens.decode(hidden_states)
        # output_multiplier FIRST, then the tanh softcap.
        return apply_output_transform(logits, self.output_multiplier,
                                      self.final_logit_softcapping)

    def get_multimodal_embeddings(self, **kwargs):
        raise NotImplementedError(
            "The JAX Muse-Glimmer port is TEXT-ONLY: the vision tower, the "
            "projector and image/video token handling are not implemented. "
            "Serve this checkpoint with text-only requests, or use the "
            "PyTorch path for multimodal inference.")


__all__ = [
    "MuseGlimmerAttention",
    "MuseGlimmerCenteredRmsNorm",
    "MuseGlimmerDecoderLayer",
    "MuseGlimmerForConditionalGeneration",
    "MuseGlimmerMLP",
    "MuseGlimmerModel",
    "MuseGlimmerScaledRmsNorm",
]
