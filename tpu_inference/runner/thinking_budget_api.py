# SPDX-License-Identifier: Apache-2.0
"""Opt-in OpenAI completions extension for the TPU thinking-budget runner."""
import json

from .thinking_budget import ANSWERING, FORCING, KEY, BudgetSpec, audit_output, build_text_detector


ANSWER_CUE = "Here is the final complete program:\n\n```python\n"
FORMATS = {
    "qwen3.5-27b": ("<think>", "</think>", "\n</think>\n\n" + ANSWER_CUE),
    "gemma4-31b": ("<|channel>thought\n", "<channel|>", "<channel|>" + ANSWER_CUE),
    "muse-glimmer-30b": (" to=self<|message|>", " to=user<|message|>",
        "<|eom|><|start|>assistant to=user<|message|>" + ANSWER_CUE),
}
CLOSE_MARKERS = {"qwen3.5-27b": "</think>", "gemma4-31b": "<channel|>",
                 "muse-glimmer-30b": "to=user"}


def prepare_request(payload, tokenizer, max_model_len, vocab_size=None, thinking_format="qwen3.5-27b"):
    budget = payload.get("thinking_token_budget")
    if budget is not None and type(budget) is not int:
        raise ValueError("thinking_token_budget must be an integer")
    if budget is None or budget == -1:
        if "forced_thinking_transition" in payload:
            raise ValueError("forced_thinking_transition requires a thinking_token_budget")
        return None
    if payload.get("stream") or payload.get("echo") or payload.get("use_beam_search"):
        raise ValueError("thinking budgets currently require non-streaming, non-echo sampling")
    if payload.get("structured_outputs") or payload.get("response_format"):
        raise ValueError("thinking transitions cannot be combined with structured outputs")
    prompt = payload.get("prompt")
    if not isinstance(prompt, list) or not prompt or any(type(t) is not int for t in prompt):
        raise ValueError("thinking-budget completions require one token-ID prompt")
    if thinking_format not in FORMATS:
        raise ValueError("unsupported model thinking format")
    start_text, end_text, default_transition = FORMATS[thinking_format]
    transition = payload.pop("forced_thinking_transition", default_transition)
    if not isinstance(transition, str) or not transition or len(transition) > 4096:
        raise ValueError("forced_thinking_transition must be nonempty text of at most 4096 characters")
    start = tokenizer.encode(start_text, add_special_tokens=False)
    end = tokenizer.encode(end_text, add_special_tokens=False)
    if not start or not end:
        raise ValueError("empty thinking marker encoding")
    # Fast tokenizers may materialize the entire vocabulary for __len__.
    # Never do that once per prompt token. Serving caches this at startup.
    if vocab_size is None:
        vocab_size = len(tokenizer)
    cache = getattr(tokenizer, "_skyrl_completer_detectors", None)
    if cache is None:
        cache = tokenizer._skyrl_completer_detectors = {}
    marker = CLOSE_MARKERS[thinking_format]
    if marker not in cache:
        cache[marker] = build_text_detector(tokenizer, marker, vocab_size)
    spec = BudgetSpec.parse(dict(budget=budget, start_id=start[0], end_id=end[0],
        transition=tokenizer.encode(transition, add_special_tokens=False),
        start_sequence=start, end_sequence=end, completer_detector=cache[marker]), vocab_size)
    if any(t < 0 or t >= vocab_size for t in prompt):
        raise ValueError("prompt contains an invalid token ID")
    eos = tokenizer.eos_token_id
    stops = set(payload.get("stop_token_ids") or []) | ({eos} if isinstance(eos, int) else set(eos or []))
    if stops.intersection(spec.transition):
        raise ValueError("transition contains an EOS or request stop token")
    stop_text = payload.get("stop") or []
    if isinstance(stop_text, str):
        stop_text = [stop_text]
    if any(s and s in transition for s in stop_text):
        raise ValueError("request stop text would truncate the transition")
    limit = payload.get("max_tokens")
    if type(limit) is not int or limit < spec.budget + len(spec.transition) + 1:
        raise ValueError("max_tokens must reserve thinking budget, transition and at least one answer token")
    if len(prompt) + limit > max_model_len:
        raise ValueError("prompt plus max_tokens exceeds the model context")
    extra = dict(payload.get("vllm_xargs") or {})
    if KEY in extra:
        raise ValueError("internal thinking-budget specification cannot be overridden")
    extra[KEY] = spec.serialize()
    payload["vllm_xargs"] = extra
    # Our TPU runner implements this request. Do not activate a second native
    # GPU reasoning processor or require the upstream reasoning parser.
    payload["thinking_token_budget"] = None
    payload["return_token_ids"] = True
    return spec, prompt


def annotate_response(payload, spec, prompt):
    if "error" in payload:
        return payload
    for choice in payload["choices"]:
        positions, state = audit_output(spec, prompt, choice["token_ids"])
        if state[0] == FORCING:
            raise RuntimeError("generation stopped in the middle of the forced transition")
        choice["forced_token_positions"] = positions
        forced = set(positions)
        choice["loss_mask"] = [0.0 if i in forced else 1.0 for i in range(len(choice["token_ids"]))]
        choice["thinking_budget"] = dict(enforced=True, forced=bool(positions),
                                         thinking_tokens=state[1], answered=state[0] == ANSWERING)
        if spec.completer_detector:
            choice["thinking_budget"].update(budget_basis="phase1_generated_tokens",
                                              counted_phase1_tokens=state[1])
    return payload


class ThinkingBudgetMiddleware:
    def __init__(self, app, tokenizer, max_model_len, supported=True, thinking_format="qwen3.5-27b"):
        self.app, self.tokenizer, self.max_model_len = app, tokenizer, max_model_len
        self.supported = supported
        self.thinking_format = thinking_format
        self.vocab_size = len(tokenizer)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("path") != "/v1/completions" or scope.get("method") != "POST":
            return await self.app(scope, receive, send)
        data = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            data.extend(message.get("body", b""))
            if len(data) > 8 * 1024**2:
                return await self.error(send, 413, "request exceeds 8 MiB")
            if not message.get("more_body", False):
                break
        try:
            payload = json.loads(data)
            if not isinstance(payload, dict):
                raise ValueError("request must be a JSON object")
            if payload.get("vllm_xargs") is not None and not isinstance(payload["vllm_xargs"], dict):
                raise ValueError("vllm_xargs must be an object")
            if KEY in (payload.get("vllm_xargs") or {}):
                raise ValueError("internal thinking-budget specification is reserved")
            enabled = payload.get("thinking_token_budget") not in (None, -1)
            if enabled and not self.supported:
                raise ValueError("thinking budgets are unsupported with this engine's decoding configuration")
            prepared = prepare_request(payload, self.tokenizer, self.max_model_len, self.vocab_size, self.thinking_format)
        except (ValueError, TypeError) as exc:
            return await self.error(send, 400, str(exc))
        body = json.dumps(payload).encode()
        sent = False

        async def replay():
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        updated = dict(scope, headers=[(k, v) for k, v in scope.get("headers", []) if k.lower() != b"content-length"]
                       + [(b"content-length", str(len(body)).encode())])
        if prepared is None:
            return await self.app(updated, replay, send)
        start, chunks = None, []

        async def collect(message):
            nonlocal start
            if message["type"] == "http.response.start":
                start = message
            elif message["type"] == "http.response.body":
                chunks.append(message.get("body", b""))

        await self.app(updated, replay, collect)
        if start is None:
            return await self.error(send, 500, "inference returned no response")
        result = b"".join(chunks)
        if start["status"] < 400:
            try:
                result = json.dumps(annotate_response(json.loads(result), *prepared)).encode()
            except (ValueError, KeyError, RuntimeError, TypeError) as exc:
                return await self.error(send, 500, f"thinking-budget output audit failed: {exc}")
        start = dict(start, headers=[(k, v) for k, v in start.get("headers", []) if k.lower() != b"content-length"]
                     + [(b"content-length", str(len(result)).encode())])
        await send(start)
        await send({"type": "http.response.body", "body": result})

    @staticmethod
    async def error(send, status, message):
        body = json.dumps({"error": {"message": message, "type": "thinking_budget_error"}}).encode()
        await send({"type": "http.response.start", "status": status,
                    "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]})
        await send({"type": "http.response.body", "body": body})


class ThinkingBudgetApp(ThinkingBudgetMiddleware):
    """ASGI wrapper preserving FastAPI state/routes after vLLM builds its stack."""
    def __getattr__(self, name):
        return getattr(self.app, name)
