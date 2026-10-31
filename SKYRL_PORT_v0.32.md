# SkyRL fork on upstream v0.32.0: port record

Every commit of `skyrl/v0.23.0-lora` (+ the SWA KV commits, `6dab20e74`) and its fate on `skyrl/v0.32.0*`.
Production serves every model through the torch path (`MODEL_IMPL_TYPE=vllm`) on v5p, v6e and v4.

| v0.23 commit | what | on v0.32 |
|---|---|---|
| `a77620929` dynamic LoRA adapter management (#2707) | worker add/remove/list/pin_lora | upstream (in 0.32.0) |
| (new) async token substitution under preemption | recompute after preemption asserted "Expected 1 token ... got 900" (Gemma) | **fixed on v0.32** (`b2fec41fc`). Open: vLLM `num_output_placeholders >= 0` assert under heavy preemption |
| `0ae6bf7aa` NaN logprob clamp | `compute_logprobs` nan_to_num | ported (`1fdf87d69`) |
| runner.patch + thinking_budget*.py (SkyRL repo assets) | native thinking-token budget | ported (`4ecb7654b`); `allow_distributed_sampling` passed through. Fix (`498722a9f`): 0.32's `sample` carries compiler_options, illegal on a nested jit; `sample_with_budget` carries them and inlines `sample.__wrapped__` (crashed every capped request on the v6e test farm) |
| `9b1a12a34`, `73decaf4f` Muse JAX core + JAX serving model | Muse flax path | ported (clean) |
| `f02489100` Muse registry wiring | `_MODEL_TYPE_REGISTRY` fallback | ported; kept upstream DFlash registrations |
| `42e4d796f` register Muse with vLLM + KV replication | `register_layers` plugin hook | ported; hook now runs upstream `register_models()` then ours |
| `4f2f67942`, `b945d0054`, `afe0cb9e9` Muse torch model (LoRA), logging, qkv geometry | production Muse path | ported (clean after registration) |
| `848263fc4` qkv width contract under LoRA | `_restore_stock_qkv_width_under_lora` | ported; `create_lora_manager(model, vllm_config)` (0.32 signature) |
| (new) K/V LoRA on replicated-KV qkv | `_tile_replicated_kv_lora_b` in `set_lora` | **fixed on v0.32 only**: k/v lora_b tiled per replica like the base weights. On v0.23 this was a documented residual: Muse adapters (q,k,v,o,gate,up,down) at TP > 2 (2 KV heads) had misplaced K/V adapter rows. Adapters that cannot be tiled are refused before activation (`_check_replicated_kv_lora_before_activation`), never half-loaded |
| `cd2228190` Muse keeps per-layer sliding windows | SlidingWindowSpec for Muse | ported (clean) |
| `a4fb045bf` SWA KV for mixed head sizes (Gemma 4) | flag, page-size unification, HBM / shape count | ported; our per-shape tensor allocation dropped: 0.32's allocator keys slots by (heads, head_size, dtype) |
| `6dab20e74` SWA KV on by default | default on for sliding-window models | ported (clean) |
| `b7ff9dc7e` v4/gemma4 serving bundle | several pieces | see below |
| `24c767036`, `81eb319d6`, `22d9fcc6c`, `b9e4024b5`, `acfe1f056`, `34dd47337`, `f8a15f67f`, `8b20a6749`, `182bc4426` | GPT-OSS MXFP4 / MoE LoRA | not ported: no GPT-OSS / MoE serving planned on v0.32 |
| `f7fd39fcb` SKIP_JAX_PRECOMPILE / VLLM_XLA_CACHE_PATH to Ray workers | env passthrough | not ported: 0.32 has SKIP_JAX_PRECOMPILE; farms run engines as processes, not Ray workers |

`b7ff9dc7e` pieces:
- `USE_JAX_RAGGED_CONV1D` (v4 Qwen: the old Pallas conv1d failed on v4): not ported. 0.32 replaced the separate
  conv1d call with a fused conv1d + GDN kernel (`kernels.gdn.v3`, `GDNLinearAttentionBackend`); the knob has no
  call site. On v4 the fused kernel needs two settings, both v4-only: a 0.9 VMEM limit (its 64-token mixed tile needs
  11.43M against the default 11.2M) and decode tile 1 (the default tile of 4 halts a core under load:
  RuntimeUnexpectedCoreHalt, then SLICE_FAILURE). Four full benchmark rounds on the OD v4-64 ran clean, at v0.23's
  throughput. A pure-JAX replacement kernel (e55a5f4f0) worked but ran 11-14% slower and was reverted.
- RPA v3 default block sizes have no `tpu_version=4` case on 0.32: ported (`966498714`). Without it the v4 AOT
  lower failed, and the warmup fallback then died on LoRA metadata leaked by the failed trace; `step_fun_impl`
  now restores LoRA metadata in a `finally` (`f75c0eb7f`).
- batched RPA bkv_sz / VMEM-fit fixes: not ported. Do not use `USE_BATCHED_RPA_KERNEL` with prefix caching on v0.32: the
  second sample of an n>1 request reads the wrong context after a partial hit during chunked prefill (Muse on v4, 9/24
  requests; v0.23 0/24; batched off 0/24 and as fast). SkyRL's config refuses the combination.
- `CUSTOM_NUM_TOKENS_BUCKETS`, `SERIALIZE_MODEL_AND_SAMPLING`: not ported, no profile sets them.
- Gemma 4 `quantize_kv` layout workaround: not ported, it is in the JAX Gemma model; Gemma serves via the torch path.
- fp32 logits only with logprobs, host-side metadata unpack: not ported (precompile / performance only; 0.32
  throughput measured at or above v0.23 per engine).
