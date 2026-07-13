#!/usr/bin/env python3
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
"""Generates test pattern and exclude lists for JAX unit tests."""

import argparse
import os
import re
import shlex
import subprocess

# ==============================================================================
# BASE / DEFAULT TEST PATTERNS & EXCLUDES
# Easily modify or add new default patterns and excludes right here at the top!
# ==============================================================================

BASE_SINGLE_CHIP_PATTERNS = [
    "/workspace/tpu_inference/tests/**/{test_*.py,*_test.py}",
]

BASE_SINGLE_CHIP_EXCLUDES = [
    "/workspace/tpu_inference/tests/e2e/**/*.py",
    "/workspace/tpu_inference/tests/lora/test_lora.py",
    "/workspace/tpu_inference/tests/lora/test_lora_adapter.py",
    "/workspace/tpu_inference/tests/kernels/collectives/**/*.py",
    "/workspace/tpu_inference/tests/kernels/ragged_paged_attention_kernel_v2_test.py",
    "/workspace/tpu_inference/tests/kernels/ragged_kv_cache_update_v2_test.py",
    "/workspace/tpu_inference/tests/kernels/spmm_v1_test.py",
    "/workspace/tpu_inference/tests/models/jax/test_deepseek_v3.py",
    "/workspace/tpu_inference/tests/offload/tpu_offload_multi_request_accuracy_test.py",
    "/workspace/tpu_inference/tests/offload/tpu_offload_accuracy_test.py",
    "/workspace/tpu_inference/tests/offload/tpu_offload_performance_test.py",
    "/workspace/tpu_inference/tests/core/test_dp_group_sampling_prefix_cache.py",
    "/workspace/tpu_inference/tests/runner/test_async_logprobs.py",
    "/workspace/tpu_inference/tests/runner/test_processed_logprobs.py",
    "/workspace/tpu_inference/tests/runner/test_moe_expert_ids.py",
]

BASE_MULTI_CHIP_PATTERNS = [
    "/workspace/tpu_inference/tests/lora/{test_lora,test_layers}.py",
]

# ==============================================================================
# CONDITIONAL TEST LOGIC
# ==============================================================================


def get_changed_files():
    """Retrieves changed files from Buildkite metadata or environment."""
    try:
        res = subprocess.run(
            [
                "buildkite-agent", "meta-data", "get", "changed_files",
                "--default", ""
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0 and res.stdout.strip():
            return [
                f.strip() for f in res.stdout.strip().split(",") if f.strip()
            ]
    except Exception:
        pass

    # Fallback to git diff locally if in a git repository
    try:
        res = subprocess.run(
            ["git", "diff", "--name-only", "origin/main...HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0 and res.stdout.strip():
            return [
                f.strip() for f in res.stdout.strip().split("\n") if f.strip()
            ]
    except Exception:
        pass

    return []


def should_run_kernels(changed_files, is_nightly):
    """Checks if general kernel unit tests should be included."""
    if is_nightly:
        return True
    pattern = re.compile(
        r"^(tpu_inference/kernels|tests/kernels|tests/conftest\.py|requirements\.txt)"
    )
    return any(pattern.match(f) for f in changed_files)


def should_run_collectives(changed_files, is_nightly):
    """Checks if collective kernel unit tests should be included."""
    if is_nightly:
        return True
    pattern = re.compile(
        r"^(tpu_inference/kernels/collectives|tests/kernels/collectives|tests/conftest\.py|requirements\.txt)"
    )
    return any(pattern.match(f) for f in changed_files)


def main():
    parser = argparse.ArgumentParser(
        description="Generate JAX unit test lists")
    parser.add_argument(
        "--export-env",
        "--export_env",
        dest="export_env",
        action="store_true",
        help="Print shell export statements for evaluation by caller",
    )
    args = parser.parse_args()

    tpu_version = os.environ.get("TPU_VERSION", "tpu6e")
    is_nightly = os.environ.get("NIGHTLY", "0") == "1"

    changed_files = get_changed_files()
    run_kernels = should_run_kernels(changed_files, is_nightly)
    run_collectives = should_run_collectives(changed_files, is_nightly)

    single_chip_patterns = list(BASE_SINGLE_CHIP_PATTERNS)
    single_chip_excludes = list(BASE_SINGLE_CHIP_EXCLUDES)
    multi_chip_patterns = list(BASE_MULTI_CHIP_PATTERNS)

    if tpu_version == "tpu7x":
        single_chip_patterns.extend([
            "/workspace/tpu_inference/tools/kernel/tuner/v1/tests/test_kernel_tuner_runner.py",
            "/workspace/tpu_inference/tools/kernel/tuner/v1/tests/test_tuned_params_structure.py",
            "/workspace/tpu_inference/tools/kernel/tuner/v1/tests/test_inspect_result_cli.py",
        ])

    if not run_kernels:
        single_chip_excludes.append(
            "/workspace/tpu_inference/tests/kernels/**/*.py")

    if run_collectives:
        multi_chip_patterns.append(
            "/workspace/tpu_inference/tests/kernels/collectives/**/{test_*.py,*_test.py}"
        )

    def format_patterns(patterns):
        if not patterns:
            return ""
        if len(patterns) == 1:
            return patterns[0]
        prefix = "/workspace/tpu_inference/tests/"
        if all(p.startswith(prefix) for p in patterns):
            stripped = [p[len(prefix):] for p in patterns]
            return prefix + "{" + ",".join(stripped) + "}"
        return "{" + ",".join(patterns) + "}"

    single_patterns_str = format_patterns(single_chip_patterns)
    single_excludes_str = format_patterns(single_chip_excludes)
    multi_patterns_str = format_patterns(multi_chip_patterns)

    if args.export_env:
        print(
            f"export SINGLE_CHIP_PATTERNS={shlex.quote(single_patterns_str)}")
        print(
            f"export SINGLE_CHIP_EXCLUDES={shlex.quote(single_excludes_str)}")
        print(f"export MULTI_CHIP_PATTERNS={shlex.quote(multi_patterns_str)}")
    else:
        print(f"[INFO] SINGLE_CHIP_PATTERNS=\n{single_patterns_str}")
        print(f"[INFO] SINGLE_CHIP_EXCLUDES=\n{single_excludes_str}")
        print(f"[INFO] MULTI_CHIP_PATTERNS=\n{multi_patterns_str}")


if __name__ == "__main__":
    main()
