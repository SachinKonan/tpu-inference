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
"""Architectures that exist in tpu-inference but not in vLLM's own registry.

vLLM resolves ``config.architectures`` against **its own** ``ModelRegistry``
inside ``ModelConfig.__post_init__`` (``is_text_generation_model`` ->
``inspect_model_cls`` -> ``_raise_for_unsupported``).  That happens in
``EngineArgs.create_engine_config()``, i.e. *before* any tpu-inference code
runs, so an architecture that is registered only in
``tpu_inference.models.common.model_loader._MODEL_REGISTRY`` never gets a
chance: ``vllm serve`` dies with

    ValueError: Model architectures ['MuseGlimmerForConditionalGeneration']
    are not supported for now.

Registering a torch class under the same name clears that gate.

Which implementation actually serves is decided later and independently, by
``model_loader.get_model``:

* ``MODEL_IMPL_TYPE=flax_nnx`` (and ``auto``, which resolves to ``flax_nnx``
  for every architecture outside ``_VLLM_PREFERRED_ARCHITECTURES``) takes the
  ``get_flax_model`` branch and looks the class up in tpu-inference's own
  ``_MODEL_REGISTRY``.  vLLM's registry is never consulted there, so what is
  registered here has no effect on the JAX path.
* ``MODEL_IMPL_TYPE=vllm`` takes ``get_vllm_model``, which calls vLLM's
  ``get_model`` and therefore instantiates exactly the class registered here.

For Muse-Glimmer the registered target is a real torch model
(``models/vllm/muse_glimmer.py``).  It exists because the JAX path returns
``lora_manager=None`` unconditionally, so ``--enable-lora`` is impossible under
``flax_nnx``; the torch model is built from vLLM's own parallel-linear classes
and therefore picks up the whole LoRA stack.  Architectures that have no torch
implementation keep pointing at ``JaxOnlyTextGenerationShim`` below, which
satisfies ``vllm.model_executor.models.interfaces_base.is_text_generation_model``
(``__init__`` accepting ``vllm_config``, plus callable ``embed_input_ids``,
``forward(input_ids, positions)`` and ``compute_logits``) and raises if
anything ever executes it.

Registration happens from the ``vllm.general_plugins`` entry point
(``tpu_inference.layers.vllm:register_layers``).  vLLM re-runs
``load_general_plugins()`` in every process it spawns -- the API server, the
EngineCore child, each worker, and the model-inspection subprocess -- so the
registration survives the ``spawn`` boundary.  The target is given as a
``"module:ClassName"`` string so that process 0 does not import torch/JAX model
code just to answer a registry question.
"""
from torch import nn

from tpu_inference.logger import init_logger

logger = init_logger(__name__)

# arch name -> "module:ClassName" resolved lazily by vLLM.
OUT_OF_TREE_ARCHITECTURES: dict[str, str] = {
    # meta-models/Muse-Glimmer-30B. The checkpoint advertises the multimodal
    # class name; both implementations here are text-only -- the JAX one in
    # tpu_inference/models/jax/muse_glimmer.py (served under
    # MODEL_IMPL_TYPE=flax_nnx, which does not read this table) and the torch
    # one below (served under MODEL_IMPL_TYPE=vllm, which does).
    "MuseGlimmerForConditionalGeneration":
    "tpu_inference.models.vllm.muse_glimmer:MuseGlimmerForCausalLM",
}

_UNSUPPORTED = (
    "This is a placeholder for a JAX-only architecture. It exists so vLLM's "
    "ModelRegistry can resolve the architecture name; the real implementation "
    "lives in tpu_inference.models.jax and runs under MODEL_IMPL_TYPE=flax_nnx. "
    "Reaching this code means the engine fell back to the PyTorch path, which "
    "this architecture does not support.")


class JaxOnlyTextGenerationShim(nn.Module):
    """Satisfies vLLM's structural checks; raises if anything actually calls it."""

    def __init__(self, vllm_config=None, prefix: str = "", **kwargs) -> None:
        super().__init__()
        self.vllm_config = vllm_config
        self.prefix = prefix

    def embed_input_ids(self, input_ids, *args, **kwargs):
        raise NotImplementedError(_UNSUPPORTED)

    def forward(self,
                input_ids,
                positions,
                intermediate_tensors=None,
                inputs_embeds=None,
                **kwargs):
        raise NotImplementedError(_UNSUPPORTED)

    def compute_logits(self, hidden_states, *args, **kwargs):
        raise NotImplementedError(_UNSUPPORTED)

    def load_weights(self, *args, **kwargs):
        # vLLM must never try to push PyTorch weights into this.
        return None


def register_out_of_tree_architectures() -> None:
    """Idempotent; safe to call from every process."""
    from vllm.model_executor.models.registry import ModelRegistry

    try:
        supported = set(ModelRegistry.get_supported_archs())
    except Exception:  # pragma: no cover - registry shape drift
        supported = set()

    for arch, target in OUT_OF_TREE_ARCHITECTURES.items():
        if arch in supported:
            continue
        try:
            ModelRegistry.register_model(arch, target)
            logger.info(
                "Registered out-of-tree architecture %s -> %s with vLLM's "
                "ModelRegistry. This target is instantiated only under "
                "MODEL_IMPL_TYPE=vllm; flax_nnx/auto resolve the JAX model "
                "from tpu-inference's own registry instead.", arch, target)
        except Exception:
            # vLLM swallows plugin exceptions and then fails later with an
            # unhelpful "architectures not supported", so log loudly here.
            logger.exception("Failed to register out-of-tree architecture %s",
                             arch)
