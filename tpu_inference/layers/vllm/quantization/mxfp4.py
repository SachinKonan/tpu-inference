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

from typing import Optional

import jax
import jax.numpy as jnp
import torch
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from torch.nn.parameter import Parameter
from torchax.interop import jax_view, torch_view
from vllm.config import get_current_vllm_config_or_none
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import (FusedMoEMethodBase,
                                                  RoutedExperts)
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig, FusedMoEQuantConfig, mxfp4_w4a16_moe_quant_config)
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import Mxfp4MoeBackend
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization import \
    register_quantization_config
from vllm.model_executor.layers.quantization.base_config import \
    QuantizeMethodBase
from vllm.model_executor.layers.quantization.mxfp4 import \
    GptOssMxfp4Config as Mxfp4Config
from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4MoEMethod
from vllm.model_executor.layers.quantization.utils.quant_utils import \
    is_layer_skipped

import tpu_inference.envs as envs
from tpu_inference.layers.common.moe import MoEBackend
from tpu_inference.layers.common.moe_lora import FusedMoELoRAWeights
from tpu_inference.layers.common.process_weights.moe_weights import (
    FusedMoEWeights, get_gmm_tp_w2_block_size, process_moe_weights,
    quantize_moe_weights, shard_moe_weights)
from tpu_inference.layers.common.quant_methods import MXFP4
from tpu_inference.layers.common.quantization import \
    dequantize_tensor_from_mxfp4_packed
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.common.utils import general_device_put
from tpu_inference.layers.vllm.interface.moe import (
    select_moe_backend_from_fused_moe_config, vllm_moe_apply)
from tpu_inference.layers.vllm.quantization.configs import VllmQuantConfig
from tpu_inference.layers.vllm.quantization.unquantized import \
    VllmUnquantizedLinearMethod
from tpu_inference.logger import init_logger
from tpu_inference.utils import get_mesh_shape_product, t2j, to_jax_dtype

REQUANTIZED_BLOCK_SIZE = 512

P = PartitionSpec

logger = init_logger(__name__)

_TPU_MOE_LORA_BUFFER_NAMES = {
    "gate_a": "tpu_moe_lora_gate_a",
    "gate_b": "tpu_moe_lora_gate_b",
    "up_a": "tpu_moe_lora_up_a",
    "up_b": "tpu_moe_lora_up_b",
    "down_a": "tpu_moe_lora_down_a",
    "down_b": "tpu_moe_lora_down_b",
    "scale": "tpu_moe_lora_scale",
}


@register_quantization_config(MXFP4)
class VllmMxfp4Config(Mxfp4Config, VllmQuantConfig):

    @classmethod
    def get_name(cls):
        return MXFP4

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["QuantizeMethodBase"]:

        if isinstance(layer, LinearBase):
            linear_config = self.get_linear_config(layer)
            if self.ignored_layers and is_layer_skipped(
                    prefix=prefix,
                    ignored_layers=self.ignored_layers,
                    fused_mapping=self.packed_modules_mapping,
            ):
                return VllmUnquantizedLinearMethod(linear_config)
            logger.warning_once(
                "MXFP4 linear layer is not implemented - falling back to "
                "UnquantizedLinearMethod.")
            return VllmUnquantizedLinearMethod(linear_config)
        elif isinstance(layer, RoutedExperts):
            moe_config = self.get_moe_config(layer)
            return VllmMxfp4MoEMethod(moe_config, self.mesh)
        elif isinstance(layer, Attention):
            logger.warning_once("MXFP4 attention layer is not implemented. "
                                "Skipping quantization for this layer.")
        return None


class VllmMxfp4MoEMethod(Mxfp4MoEMethod):

    def __init__(
        self,
        moe: FusedMoEConfig,
        mesh: Mesh,
        ep_axis_name: str = "model",
    ):
        FusedMoEMethodBase.__init__(self, moe)

        # We piggyback on triton implementation as it applies minimal hardware
        # specific post processing to the weights.
        self.mxfp4_backend = Mxfp4MoeBackend.TRITON

        self.mesh = mesh
        self.moe_backend = select_moe_backend_from_fused_moe_config(self.moe)
        current_config = get_current_vllm_config_or_none()
        self.lora_config = (None if current_config is None else
                            current_config.lora_config)

        self.extra_backend_kwargs = {}
        if self.moe_backend == MoEBackend.FUSED_MOE:
            # When fused moe kernle is used, we pass extra arguments like
            # tuned block sizes to the kernel.
            self.extra_backend_kwargs = dict(ep_axis_name=ep_axis_name, )

    def _create_lora_buffers(self, layer: RoutedExperts,
                             weights: FusedMoEWeights) -> None:
        """Create fixed-shape BF16 factor banks before JAX compilation.

        Buffer shapes use ``max_loras`` and ``max_lora_rank`` so loading or
        evicting adapters changes values, never the model pytree or executable
        signature.  The slot axis matches vLLM/Punica's physical LoRA slots.
        """

        if self.lora_config is None:
            return
        if hasattr(layer, _TPU_MOE_LORA_BUFFER_NAMES["gate_a"]):
            return
        if self.moe_backend not in (MoEBackend.GMM_EP, MoEBackend.GMM_TP):
            raise NotImplementedError(
                "MXFP4 expert LoRA requires the GMM_TP or GMM_EP backend; "
                f"got {self.moe_backend.value}.")
        if layer.activation != MoEActivation.SWIGLUOAI:
            raise NotImplementedError(
                "The TPU MXFP4 expert-LoRA path currently implements the "
                "GPT-OSS SwiGLU activation contract only.")

        num_experts, padded_hidden, fused_intermediate = (
            weights.w13_weight.shape)
        padded_intermediate = weights.w2_weight.shape[1]
        if fused_intermediate != 2 * padded_intermediate:
            raise ValueError(
                "Expected fused GPT-OSS W13 output to be twice W2's "
                "intermediate input, got "
                f"{fused_intermediate} and {padded_intermediate}.")
        max_rank = int(self.lora_config.max_lora_rank)
        max_loras = int(self.lora_config.max_loras)

        if self.moe_backend == MoEBackend.GMM_EP:
            expert_spec = P(ShardingAxisName.EXPERT)
            gate_b_spec = expert_spec
            down_a_spec = expert_spec
        else:
            gate_b_spec = P(None, None, ShardingAxisName.MLP_TENSOR)
            down_a_spec = P(None, ShardingAxisName.MLP_TENSOR, None)

        shapes_and_specs = {
            "gate_a": ((max_loras, padded_hidden, max_rank), P()),
            "gate_b":
            ((max_loras, num_experts, max_rank, padded_intermediate),
             P(None, *gate_b_spec)),
            "up_a": ((max_loras, padded_hidden, max_rank), P()),
            "up_b":
            ((max_loras, num_experts, max_rank, padded_intermediate),
             P(None, *gate_b_spec)),
            "down_a":
            ((max_loras, num_experts, padded_intermediate, max_rank),
             P(None, *down_a_spec)),
            "down_b": ((max_loras, max_rank, padded_hidden), P()),
            "scale": ((max_loras, ), P()),
        }
        for field, (shape, spec) in shapes_and_specs.items():
            value = general_device_put(
                jnp.zeros(shape, dtype=jnp.bfloat16),
                NamedSharding(self.mesh, spec),
            )
            layer.register_buffer(_TPU_MOE_LORA_BUFFER_NAMES[field],
                                  torch_view(value),
                                  persistent=False)
        logger.info_once(
            "Allocated separate BF16 MXFP4 expert-LoRA banks "
            "(max_loras=%d, max_rank=%d, backend=%s); base expert weights "
            "remain immutable.", max_loras, max_rank, self.moe_backend.value)

    def get_fused_moe_quant_config(
            self, layer: torch.nn.Module) -> FusedMoEQuantConfig | None:
        return mxfp4_w4a16_moe_quant_config(
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            w1_bias=getattr(layer, "w13_bias", None),
            w2_bias=getattr(layer, "w2_bias", None),
        )

    @property
    def is_monolithic(self) -> bool:
        return True

    def process_weights_after_loading(self, layer: torch.nn.Module):
        assert isinstance(layer, RoutedExperts)
        has_bias = layer.moe_config.has_bias

        w13_weight = t2j(layer.w13_weight, use_dlpack=False)
        w13_weight_scale = t2j(layer.w13_weight_scale, use_dlpack=False)
        w13_bias = t2j(layer.w13_bias, use_dlpack=False) if has_bias else None

        w2_weight = t2j(layer.w2_weight, use_dlpack=False)
        w2_weight_scale = t2j(layer.w2_weight_scale, use_dlpack=False)
        w2_bias = t2j(layer.w2_bias, use_dlpack=False) if has_bias else None

        desired_quant_dtype = (
            to_jax_dtype(envs.MOE_REQUANTIZE_WEIGHT_DTYPE)
            if envs.MOE_REQUANTIZE_WEIGHT_DTYPE else jnp.float4_e2m1fn)
        preferred_requant_block_size = (envs.MOE_REQUANTIZE_BLOCK_SIZE
                                        or REQUANTIZED_BLOCK_SIZE)
        logger.info_once(
            "Requantizing native MXFP4 experts for TPU execution "
            "(dtype=%s, preferred_block_size=%d).", desired_quant_dtype,
            preferred_requant_block_size)

        @jax.jit
        def process_mxfp4_moe_weights(
            w13_weight: jax.Array,
            w13_weight_scale: jax.Array,
            w13_bias: jax.Array | None,
            w2_weight: jax.Array,
            w2_weight_scale: jax.Array,
            w2_bias: jax.Array | None,
        ) -> FusedMoEWeights:
            # Dequantize fp4 weights into fp32.
            w13_weight = dequantize_tensor_from_mxfp4_packed(
                w13_weight, w13_weight_scale, 2, jnp.float32)
            w2_weight = dequantize_tensor_from_mxfp4_packed(
                w2_weight, w2_weight_scale, 2, jnp.float32)
            w13_interleave = layer.activation == MoEActivation.SWIGLUOAI
            w13_reorder_size = get_mesh_shape_product(
                self.mesh, ShardingAxisName.MLP_TENSOR)

            requant_block_size = preferred_requant_block_size
            if self.moe_backend == MoEBackend.GMM_TP:
                # W2 scales are sharded by their block-count dimension.  The
                # generic cap used by other quantized paths is not reached by
                # this native MXFP4 loader, so choose a block count divisible
                # by TP here as well (GPT-OSS 20B: 2880 / TP8 -> block 384,
                # eight scale blocks).
                w2_block_size = get_gmm_tp_w2_block_size(
                    w2_weight.shape[2], requant_block_size,
                    w13_reorder_size)
                requant_block_size = (requant_block_size, w2_block_size)

            weights = quantize_moe_weights(
                FusedMoEWeights(
                    w13_weight=w13_weight,
                    w13_weight_scale=None,
                    w13_bias=w13_bias,
                    w2_weight=w2_weight,
                    w2_weight_scale=None,
                    w2_bias=w2_bias,
                ),
                desired_quant_dtype,
                requant_block_size,
                w13_interleave=w13_interleave,
            )
            return process_moe_weights(
                weights,
                moe_backend=self.moe_backend,
                w13_reorder_size=w13_reorder_size,
                w13_interleave=w13_interleave,
            )

        weights = process_mxfp4_moe_weights(
            w13_weight,
            w13_weight_scale,
            w13_bias,
            w2_weight,
            w2_weight_scale,
            w2_bias,
        )
        weights = torch_view(
            shard_moe_weights(weights, self.moe_backend, self.mesh))

        layer.w13_weight = Parameter(weights.w13_weight, requires_grad=False)
        layer.w2_weight = Parameter(weights.w2_weight, requires_grad=False)

        layer.w13_weight_scale = Parameter(weights.w13_weight_scale,
                                           requires_grad=False)
        layer.w2_weight_scale = Parameter(weights.w2_weight_scale,
                                          requires_grad=False)

        if has_bias:
            layer.w13_bias = Parameter(weights.w13_bias, requires_grad=False)
            layer.w2_bias = Parameter(weights.w2_bias, requires_grad=False)

        self._create_lora_buffers(layer, weights)

    def apply_monolithic(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:

        has_bias = layer.moe_config.has_bias
        weights = FusedMoEWeights(
            w13_weight=jax_view(layer.w13_weight),
            w13_weight_scale=jax_view(layer.w13_weight_scale),
            w13_bias=jax_view(layer.w13_bias) if has_bias else None,
            w2_weight=jax_view(layer.w2_weight),
            w2_weight_scale=jax_view(layer.w2_weight_scale),
            w2_bias=jax_view(layer.w2_bias) if has_bias else None,
        )

        lora_weights = None
        if hasattr(layer, _TPU_MOE_LORA_BUFFER_NAMES["gate_a"]):
            lora_weights = FusedMoELoRAWeights(
                **{
                    field: jax_view(getattr(layer, name))
                    for field, name in _TPU_MOE_LORA_BUFFER_NAMES.items()
                })

        return vllm_moe_apply(layer=layer,
                              weights=weights,
                              quant_method_instance=self,
                              x=x,
                              router_logits=router_logits,
                              input_ids=input_ids,
                              lora_weights=lora_weights)
