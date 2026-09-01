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

from dataclasses import dataclass
from typing import Any

import jax


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class FusedMoELoRAWeights:
    """Single-active-adapter LoRA factors for fused GPT-OSS experts.

    The MXFP4 base tensors stay immutable. These BF16 factors are consumed
    beside the base GMMs, using the already-sorted expert token groups. Gate
    and up ``A`` factors and the down ``B`` factor are shared across experts,
    matching MaxText/Qwix's GPT-OSS parameterization.

    Shapes (E=experts, H=padded hidden, I=padded intermediate, R=max rank):
      gate_a/up_a: (H, R)       gate_b/up_b: (E, R, I)
      down_a:      (E, I, R)    down_b:      (R, H)
      scale: scalar
    """

    gate_a: Any
    gate_b: Any
    up_a: Any
    up_b: Any
    down_a: Any
    down_b: Any
    scale: Any
