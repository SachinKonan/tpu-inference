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
"""vLLM-native (torch) **text-only** ``meta-models/Muse-Glimmer-30B``.

Why this exists alongside ``models/jax/muse_glimmer.py``
-------------------------------------------------------
The JAX-native model is the faster, proven serving path, but
``model_loader.get_flax_model`` returns ``lora_manager=None`` unconditionally,
so ``--enable-lora`` cannot work under ``MODEL_IMPL_TYPE=flax_nnx``.  RL weight
sync in SkyRL uploads LoRA adapters to vLLM, so the training loop needs a path
that has the LoRA stack.

That path is ``MODEL_IMPL_TYPE=vllm``: ``VllmModelWrapper.load_weights`` runs
``load_lora_model`` + ``replace_set_lora``, which walk ``named_modules()`` and
wrap every ``BaseLayerWithLoRA``.  A model assembled out of vLLM's own
``QKVParallelLinear`` / ``MergedColumnParallelLinear`` / ``RowParallelLinear`` /
``VocabParallelEmbedding`` / ``ParallelLMHead`` therefore gets adapters for
free; torchax executes the torch graph on TPU and tpu-inference's OOT layer
overrides (``layers/vllm/custom_ops``, ``layers/vllm/quantization``) do the
sharding.

**This file is additive.**  ``MODEL_IMPL_TYPE=flax_nnx`` (and ``auto``, which
resolves to ``flax_nnx`` for this architecture) still selects the JAX model out
of ``model_loader._MODEL_REGISTRY``; vLLM's own ``ModelRegistry`` -- which is
what this class is registered into -- is consulted only on the ``vllm`` branch.

Numerics
--------
``tpu/muse_glimmer/SPEC.md`` is the contract and ``models/jax/muse_glimmer_core``
is the second reference; the two implement the same forward pass.  The twelve
parity traps are called out inline where they bite.

Text only: no vision tower, projector or image/video tokens.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
from vllm.config import CacheConfig, VllmConfig
from vllm.model_executor.layers.activation import get_act_and_mul_fn
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import SupportsLoRA
from vllm.model_executor.models.utils import (AutoWeightsLoader,
                                              extract_layer_index,
                                              maybe_prefix)
from vllm.sequence import IntermediateTensors

from tpu_inference.logger import init_logger

logger = init_logger(__name__)

SLIDING = "sliding_attention"

#: Checkpoint prefixes belonging to the (out-of-scope) vision stack.
VISION_KEY_PREFIXES = (
    "model.vision_tower.",
    "model.vision_adapter.",
    "model.vision_projection.",
    "vision_tower.",
    "vision_adapter.",
    "vision_projection.",
)


def _text_config(config: Any) -> Any:
    """Muse-Glimmer ships the composite (vision + text) config."""
    return getattr(config, "text_config", config)


# ---------------------------------------------------------------------------
# Norms.  Muse-Glimmer has THREE flavours; mixing them up silently produces
# plausible-but-wrong output (SPEC.md, "THREE norm flavours").
#
#   * parameter-free  ``n``            -- embedding norm and the q/k norms
#   * scaled          ``n * w``        -- the FINAL model.norm ONLY (ones init)
#   * centred         ``n * (1 + w)``  -- the four per-decoder-layer norms
#                                         (zeros init)  == vLLM's GemmaRMSNorm
#
# Every flavour reduces in float32 regardless of activation dtype; that is what
# sets the achievable parity floor.  HF deliberately writes
# ``pow(mean_sq + eps, -0.5)`` rather than ``rsqrt`` for the two ``MuseGlimmer
# RMSNorm`` flavours "to address compiler differences between Torch and JAX",
# and ``rsqrt`` for the centred one; we mirror each choice exactly so this port
# and ``muse_glimmer_core`` agree with the reference to the last ulp.
# ---------------------------------------------------------------------------


def _rms_norm_no_scale(x: torch.Tensor, eps: float) -> torch.Tensor:
    """``MuseGlimmerRMSNorm(with_scale=False)`` -- no parameters."""
    orig_dtype = x.dtype
    xf = x.float()
    mean_squared = xf.pow(2).mean(dim=-1, keepdim=True) + eps
    return (xf * torch.pow(mean_squared, -0.5)).to(orig_dtype)


class MuseGlimmerRMSNormNoScale(nn.Module):
    """Parameter-free RMSNorm.  Used by the normed embedding and, per head over
    ``head_dim``, by the q/k norms (SPEC trap 3 -- these weights are absent from
    the checkpoint, so vLLM's weight-carrying ``RMSNorm`` cannot be used)."""

    def __init__(self, eps: float) -> None:
        super().__init__()
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _rms_norm_no_scale(x, self.variance_epsilon)

    def extra_repr(self) -> str:
        return f"eps={self.variance_epsilon}, has_weight=False"


class MuseGlimmerScaledRMSNorm(nn.Module):
    """``MuseGlimmerRMSNorm(with_scale=True)`` -- ``n * w``, weight init ONES.

    SPEC trap 10: the final ``model.norm`` is this flavour, *not* the centred
    ``(1 + w)`` one used inside the decoder layers.  Getting it wrong scales the
    final hidden state by roughly 2x.
    """

    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        xf = x.float()
        mean_squared = xf.pow(2).mean(dim=-1, keepdim=True) + self.variance_epsilon
        normed = xf * torch.pow(mean_squared, -0.5)
        return (normed * self.weight.float()).to(orig_dtype)

    def extra_repr(self) -> str:
        return f"hidden_size={self.weight.size(0)}, eps={self.variance_epsilon}"


# ``GemmaRMSNorm`` is ``(x.float() * rsqrt(var + eps)).to(w.dtype) * (1 + w)``
# cast back to the input dtype, with a zeros-initialised weight -- byte for byte
# the ``MuseGlimmerTextCenteredRMSNorm`` of the reference (SPEC trap 1).
MuseGlimmerCenteredRMSNorm = GemmaRMSNorm


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------


class MuseGlimmerMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_activation: str,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_activation not in ("silu", "swish"):
            raise ValueError(
                f"Muse-Glimmer expects a SwiGLU MLP; got hidden_activation="
                f"{hidden_activation!r}")
        self.act_fn = get_act_and_mul_fn(hidden_activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class MuseGlimmerAttention(nn.Module):
    """GQA + parameter-free per-head qk-norm + a sigmoid output gate.

    Three things here are not standard vLLM shapes and are worth stating
    plainly:

    1. ``qk_scale_factor`` (3.87) multiplies **q only**, after the q norm, and
       is entirely separate from the ``head_dim ** -0.5`` softmax scale that is
       handed to the kernel (SPEC trap 2).  Effective q scale = 3.87 x 0.0884.
    2. The output gate is ``sigmoid(attn_gate_proj(x))`` where ``x`` is the
       **input_layernorm output**, not the attention output, and it is applied
       elementwise over ``heads * head_dim`` **before** ``o_proj``
       (SPEC trap 4).
    3. ``layer_rope_theta[i] == 0`` marks a NoPE layer -- the 13 full-attention
       layers get no rotation at all (SPEC trap 5).
    """

    def __init__(
        self,
        config: Any,
        layer_idx: int,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim",
                                config.hidden_size // config.num_attention_heads)
        self.rms_norm_eps = float(config.rms_norm_eps)
        self.qk_scale_factor = float(config.qk_scale_factor)

        # The softmax scale handed to the attention kernel.  `qk_scale_factor`
        # is applied to q *before* this, in the model.
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=getattr(config, "attention_bias", False),
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        # HEAD COUNTS: STOCK geometry, deliberately (2026-08-17 crash root
        # cause).  tpu-inference's OOT `VllmQKVParallelLinear` inflates its
        # WEIGHT BUFFER to `mesh TP` KV heads when TP > num_key_value_heads
        # (Muse-Glimmer: 2 KV heads, TP=4), replicating each head with
        # `repeat_interleave` ([h0 h0 h1 h1]) — but its `forward` then
        # COLLAPSES the replica sub-axis back out of the global view
        # (`dedup_replicated_kv`) and returns the STOCK widths
        # `q + 2 * total_num_kv_heads * head_dim`, so that model code written
        # against stock vLLM (torch world_size=1) keeps working — exactly
        # what upstream vLLM's own muse_glimmer model assumes.  A previous
        # revision read the layer's INFLATED `num_kv_heads` here instead;
        # under torchax, `split` lowers to clamped JAX slicing, so splitting
        # the collapsed (narrower) tensor by inflated sizes silently produced
        # an EMPTY v — the first-request TPU crash at flash_attn reshape.
        # `total_num_kv_heads` is the REAL checkpoint count (the OOT layer
        # restores it after its inflated super-init; stock vLLM layers carry
        # the same attribute), and the rest of the stack agrees with stock
        # geometry: the runner pads the KV cache head count up to TP
        # (`get_padded_num_heads`) and `sharded_ragged_paged_attention`
        # replicates k/v at kernel entry.
        self.num_heads = self.qkv_proj.num_heads
        self.num_kv_heads = self.qkv_proj.total_num_kv_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=getattr(config, "attention_bias", False),
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # NAMING, deliberately: the checkpoint calls this `self_attn.gate_proj`,
        # but vLLM matches LoRA `target_modules` on the LAST dotted component
        # (`re.match(rf".*\.{target}$", name)`), so `self_attn.gate_proj` would
        # collide with the MLP's SwiGLU `gate_proj` -- a different tensor with a
        # different shape.  `load_weights` renames the incoming key.
        #
        # SHARDING: this is a plain ColumnParallelLinear, so its output is
        # split into contiguous blocks along `heads * head_dim` on the same
        # mesh axis (`ShardingAxisName.ATTN_HEAD`) that the attention output
        # and `o_proj`'s input use.  Rank r owns q heads
        # [r * H/tp, (r+1) * H/tp) in all three, so the elementwise product
        # lines up without any collective.
        self.attn_gate_proj = ColumnParallelLinear(
            self.hidden_size,
            self.total_num_heads * self.head_dim,
            bias=False,
            gather_output=False,
            quant_config=quant_config,
            prefix=f"{prefix}.attn_gate_proj",
        )

        # Parameter-free, per head, over head_dim.  Not vLLM's `RMSNorm`:
        # that one carries a weight, and these do not exist in the checkpoint.
        self.q_norm = MuseGlimmerRMSNormNoScale(self.rms_norm_eps)
        self.k_norm = MuseGlimmerRMSNormNoScale(self.rms_norm_eps)

        layer_type = config.layer_types[layer_idx]
        self.is_sliding = layer_type == SLIDING
        sliding_window = config.sliding_window if self.is_sliding else None

        # NoPE: `layer_rope_theta[i] == 0` on the 13 full-attention layers.
        layer_theta = float(config.layer_rope_theta[layer_idx])
        if layer_theta:
            rope_parameters = dict(getattr(config, "rope_parameters", None) or {})
            # CAREFUL: on transformers v5 configs `rope_scaling` is an alias for
            # `rope_parameters`, i.e. a truthy dict `{"rope_theta": ...,
            # "rope_type": "default"}`.  Forwarding a scaled rope type this
            # model does not ask for would apply llama3-style frequency
            # rescaling.  Force the per-layer theta and keep the declared type.
            rope_parameters["rope_theta"] = layer_theta
            rope_parameters.setdefault("rope_type", "default")
            self.rotary_emb = get_rope(
                self.head_dim,
                max_position=config.max_position_embeddings,
                rope_parameters=rope_parameters,
                is_neox_style=True,  # HF "rotate_half" == neox half-split
            )
        else:
            self.rotary_emb = None

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=sliding_window,
            prefix=f"{prefix}.attn",
        )
        # The window must reach the KERNEL but must NOT reach the KV-cache
        # spec.  `PallasAttentionBackendImpl` keeps its own copy of
        # `sliding_window` (taken above, at construction) and forwards it to the
        # ragged-paged-attention kernel; `runner/kv_cache_manager.get_kv_cache_
        # spec` reads `attn_module.sliding_window` and, when it is not None,
        # emits a `SlidingWindowSpec` -- i.e. a hybrid KV cache.  That hybrid
        # layout was implemented, measured and REVERTED on the JAX path
        # (E2E.md section 5): a 4.5% capacity *regression* at 4096, and one
        # sequence in five corrupted under concurrency.  Clearing the attribute
        # keeps a uniform single-group cache, exactly matching the JAX model
        # that decoded 4/5 prompts token-for-token at 3609 tokens.
        # `kv_cache_manager` itself uses this same idiom for its
        # `disable_sliding_window` workaround.
        self.attn.sliding_window = None

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # HARDENING: under torchax, `split` lowers to JAX slicing, which
        # CLAMPS out-of-range slices silently -- a qkv narrower than
        # q_size + 2*kv_size does not raise, it hands back a truncated k and
        # an EMPTY v, which then dies far away in the attention backend as
        # `cannot reshape (0, kv_heads, head_dim)` (the 2026-08-17 crash).
        # Fail here, loudly, naming the actual width.
        if (q.shape[-1], k.shape[-1], v.shape[-1]) != (self.q_size,
                                                       self.kv_size,
                                                       self.kv_size):
            raise AssertionError(
                f"qkv_proj returned width {qkv.shape[-1]} but the model "
                f"declares q_size + 2*kv_size = {self.q_size} + "
                f"2*{self.kv_size} = {self.q_size + 2 * self.kv_size} "
                f"(split produced q={q.shape[-1]} k={k.shape[-1]} "
                f"v={v.shape[-1]}); the layer's runtime output width "
                f"diverged from the model's declared head geometry")

        # Per-head, parameter-free norm over head_dim; `qk_scale_factor` on q
        # only, applied AFTER the norm (SPEC trap 2).
        q = q.unflatten(-1, (self.num_heads, self.head_dim))
        q = self.q_norm(q) * self.qk_scale_factor
        q = q.flatten(-2, -1).to(hidden_states.dtype)

        k = k.unflatten(-1, (self.num_kv_heads, self.head_dim))
        k = self.k_norm(k)
        k = k.flatten(-2, -1).to(hidden_states.dtype)

        if self.rotary_emb is not None:
            q, k = self.rotary_emb(positions, q, k)

        attn_output = self.attn(q, k, v)

        # SPEC trap 4: gate from `hidden_states` (the input_layernorm output),
        # NOT from the attention output, and applied BEFORE o_proj.
        gate, _ = self.attn_gate_proj(hidden_states)
        attn_output = attn_output * torch.sigmoid(gate)

        output, _ = self.o_proj(attn_output)
        return output


# ---------------------------------------------------------------------------
# Decoder layer
# ---------------------------------------------------------------------------


class MuseGlimmerDecoderLayer(nn.Module):
    """Sandwich norms.  The two PRE norms use ``rms_norm_eps`` (1e-5), the two
    POST norms use ``post_norm_eps`` (1e-8) -- SPEC trap 6."""

    def __init__(
        self,
        config: Any,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        layer_idx = extract_layer_index(prefix)
        self.layer_idx = layer_idx
        hidden_size = config.hidden_size
        rms_norm_eps = float(config.rms_norm_eps)
        post_norm_eps = float(getattr(config, "post_norm_eps", rms_norm_eps))

        self.input_layernorm = MuseGlimmerCenteredRMSNorm(hidden_size,
                                                          eps=rms_norm_eps)
        self.self_attn = MuseGlimmerAttention(
            config=config,
            layer_idx=layer_idx,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.post_attention_layernorm = MuseGlimmerCenteredRMSNorm(
            hidden_size, eps=post_norm_eps)
        self.pre_feedforward_layernorm = MuseGlimmerCenteredRMSNorm(
            hidden_size, eps=rms_norm_eps)
        self.mlp = MuseGlimmerMLP(
            hidden_size=hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_activation=getattr(config, "hidden_activation", "silu"),
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.post_feedforward_layernorm = MuseGlimmerCenteredRMSNorm(
            hidden_size, eps=post_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(positions=positions,
                                       hidden_states=hidden_states,
                                       **kwargs)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        return residual + hidden_states


# ---------------------------------------------------------------------------
# Text stack
# ---------------------------------------------------------------------------


class MuseGlimmerModel(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = _text_config(vllm_config.model_config.hf_config)
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.config = config

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.embed_tokens",
        )
        # `MuseGlimmerTextNormedEmbedding`: a parameter-free RMSNorm on top of
        # the lookup.  SPEC trap 9: there is deliberately NO sqrt(hidden_size)
        # multiplier here -- do not copy Gemma's embedding scaling.
        self.embed_norm = MuseGlimmerRMSNormNoScale(float(config.rms_norm_eps))

        # No pipeline parallelism: `VllmModelWrapper._apply_pp_patch` only
        # rewrites `get_pp_group` inside `vllm.model_executor.models.*`, so an
        # out-of-tree model that called it would reach vLLM's real PP group,
        # which tpu-inference never initialises.  PP is out of scope here; the
        # JAX model keeps that capability.
        self.layers = nn.ModuleList([
            MuseGlimmerDecoderLayer(
                config=config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.layers.{i}",
            ) for i in range(config.num_hidden_layers)
        ])

        # SPEC trap 10: `n * w`, ones init -- not the centred variant.
        self.norm = MuseGlimmerScaledRMSNorm(config.hidden_size,
                                             eps=float(config.rms_norm_eps))

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_norm(self.embed_tokens(input_ids))

    # vLLM's newer name for the same thing; keep both so either interface
    # probe finds it.
    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_input_ids(input_ids)

        for layer in self.layers:
            hidden_states = layer(positions, hidden_states, **kwargs)

        return self.norm(hidden_states)


# ---------------------------------------------------------------------------
# Top level
# ---------------------------------------------------------------------------


class MuseGlimmerForCausalLM(nn.Module, SupportsLoRA):
    """Text-only Muse-Glimmer for the torch/torchax serving path.

    Registered under the checkpoint's advertised architecture name,
    ``MuseGlimmerForConditionalGeneration`` (see
    ``models/common/oot_registration.py``); the class name says CausalLM
    because this implementation has no vision stack.
    """

    # `qkv_proj` / `gate_up_proj` are the packed modules an adapter's
    # `q_proj`/`k_proj`/`v_proj` and `gate_proj`/`up_proj` targets route into.
    #
    # The attention gate is NOT in this table on purpose: it lives at
    # `self_attn.attn_gate_proj`, so vLLM's last-component matching keeps
    # `gate_proj` unambiguously the MLP gate.  The flip side is that an adapter
    # trained against raw HF module names (where the gate is
    # `self_attn.gate_proj`) would be routed into `gate_up_proj` and fail on
    # shape; adapters produced by this stack use `attn_gate_proj` and are fine.
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    embedding_modules: dict[str, str] = {}

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = _text_config(vllm_config.model_config.hf_config)
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config

        self.model = MuseGlimmerModel(vllm_config=vllm_config,
                                      prefix=maybe_prefix(prefix, "model"))

        # SPEC trap 8: `text_config.tie_word_embeddings` is false and the 30B
        # ships a separate `lm_head.weight` (measured
        # `max|lm_head - embed| = 3.09`).  Only fall back to the tie when the
        # config actually asks for it.
        self.tie_word_embeddings = bool(
            getattr(config, "tie_word_embeddings", False))
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if self.tie_word_embeddings:
            self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)

        # SPEC trap 7: `logits * output_multiplier` FIRST, then the tanh
        # softcap.  vLLM's LogitsProcessor applies its `soft_cap` BEFORE its
        # `scale`, i.e. the opposite order, so neither knob can be used --
        # the processor is left as a plain gather (TP all-gather + vocab
        # de-padding) and the transform is applied here.
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.output_multiplier = float(
            getattr(config, "output_multiplier", 1.0))
        final_logit_softcapping = getattr(config, "final_logit_softcapping",
                                          None)
        self.final_logit_softcapping = (None if final_logit_softcapping is None
                                        else float(final_logit_softcapping))

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, intermediate_tensors,
                          inputs_embeds, **kwargs)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        if logits is None:
            return None
        logits = logits * self.output_multiplier
        if self.final_logit_softcapping is not None:
            cap = self.final_logit_softcapping
            logits = torch.tanh(logits / cap) * cap
        return logits

    # -- weight loading ----------------------------------------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """Map the 30B checkpoint's key layout onto this module tree.

        The checkpoint stores the text stack under ``model.language_model.*``
        plus a top-level ``lm_head.weight``, and separate ``q_proj``/``k_proj``/
        ``v_proj`` and ``gate_proj``/``up_proj`` tensors.
        """
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        def _weight_iterator():
            for name, weight in weights:
                if name.startswith(VISION_KEY_PREFIXES):
                    continue
                name = name.replace("language_model.", "")
                # MUST run before the stacked mapping below: the attention gate
                # is `self_attn.gate_proj` on disk, and `gate_proj` is also the
                # MLP's SwiGLU gate.  Renaming first makes the substring test
                # in the stacked mapping unambiguous.
                name = name.replace("self_attn.gate_proj",
                                    "self_attn.attn_gate_proj")
                yield name, weight

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in _weight_iterator():
            if self.tie_word_embeddings and name.startswith("lm_head."):
                continue

            for param_name, shard_name, shard_id in stacked_params_mapping:
                if shard_name not in name:
                    continue
                mapped = name.replace(shard_name, param_name)
                if mapped not in params_dict:
                    continue
                param = params_dict[mapped]
                param.weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(mapped)
                break
            else:
                if name not in params_dict:
                    logger.debug("Skipping unexpected checkpoint key %s", name)
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

        return loaded_params


# `AutoWeightsLoader` is imported for parity with the in-tree vLLM models and
# to keep the import surface stable for anything that introspects this module;
# the hand-rolled loader above is used because Muse-Glimmer needs the
# `self_attn.gate_proj` rename to happen before packed-module routing.
_ = AutoWeightsLoader

__all__ = [
    "MuseGlimmerAttention",
    "MuseGlimmerDecoderLayer",
    "MuseGlimmerForCausalLM",
    "MuseGlimmerMLP",
    "MuseGlimmerModel",
    "MuseGlimmerRMSNormNoScale",
    "MuseGlimmerScaledRMSNorm",
]
