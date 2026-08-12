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
from tpu_inference.layers.vllm import backends as backends
from tpu_inference.layers.vllm import custom_ops as custom_ops
from tpu_inference.layers.vllm import ops as ops
from tpu_inference.layers.vllm import quantization as quantization


# NOTE: this function is the `vllm.general_plugins` entry_points target (see
# setup.py). vLLM re-runs `load_general_plugins()` in every process it spawns,
# which makes it the only reliable place to teach vLLM's OWN ModelRegistry
# about JAX-only architectures -- that lookup happens in
# `ModelConfig.__post_init__`, before any tpu-inference code would otherwise
# run. See models/common/oot_registration.py.
def register_layers():
    from tpu_inference.models.common.oot_registration import \
        register_out_of_tree_architectures
    register_out_of_tree_architectures()
