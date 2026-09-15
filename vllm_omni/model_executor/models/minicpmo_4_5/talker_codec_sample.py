"""Framework boundaries for Talker codec sampling.

The greedy PoC is kept intact. Production stochastic sampling uses a separate
logits-filter boundary: AscendC handles history counting, temperature,
repetition penalty, EOS mask, top-p and top-k, while native NPU operators keep
softmax and ``torch.multinomial`` semantics (including Generator state)
unchanged.

``CodecStepGraph`` then captures that whole per-frame chain -- head projection,
the A14 filter, softmax, multinomial and the state transition -- into a single
NPUGraph. The filter kernel collapses 41 dispatches; the graph collapses what
is left, which after the kernel is mostly the ~25 tiny state-transition ops.
"""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass

import torch
from vllm.logger import init_logger

VOCAB_SIZE = 6562
HISTORY_WINDOW = 16
MIN_TOKENS_TO_KEEP = 3
_A14_EXTENSION_ENV = "VLLM_OMNI_A14_EXTENSION"
_A14_OPAPI_ENV = "VLLM_OMNI_A14_OPAPI_LIBRARY"
_A14_MODE_ENV = "VLLM_OMNI_A14_MODE"
_A14_GRAPH_ENV = "VLLM_OMNI_A14_GRAPH"
_extension_load_attempted = False
_extension_available = False
_opapi_handle: ctypes.CDLL | None = None

logger = init_logger(__name__)


@dataclass(frozen=True)
class TalkerCodecDeviceState:
    history: torch.Tensor
    history_len: torch.Tensor
    step: torch.Tensor
    max_tokens: torch.Tensor
    finished: torch.Tensor


@dataclass(frozen=True)
class TalkerCodecSampleResult:
    sampled_token: torch.Tensor
    state: TalkerCodecDeviceState
    emit: torch.Tensor
    # Set only by ``CodecStepGraph``, which folds the codec delta and the
    # stop-logits row into the same replay instead of masking them afterwards.
    delta: torch.Tensor | None = None
    stop_row: torch.Tensor | None = None


def a14_mode() -> str:
    """Return the explicit production A14 mode.

    ``auto`` is the default: it uses the packaged operator when available and
    falls back to the eager chain without changing sampling semantics, so a
    host without the companion payload still serves. ``off`` preserves the
    legacy stochastic boundary for a package-level baseline. ``required``
    fails fast if the production boundary is unavailable -- use it in
    validation runs that must prove the operator is live. An explicit
    development bridge implies ``required`` for backwards-compatible PoC
    commands.

    The default is deliberately not ``off``: the official evaluation starts
    the server with no vLLM-Omni environment variables, so anything gated
    behind one is dead on arrival.
    """
    configured = os.environ.get(_A14_MODE_ENV)
    if configured is None:
        return "required" if os.environ.get(_A14_EXTENSION_ENV) else "auto"
    mode = configured.strip().lower()
    if mode not in {"auto", "off", "required"}:
        raise RuntimeError(f"{_A14_MODE_ENV} must be 'auto', 'off' or 'required', got {configured!r}")
    return mode


def a14_graph_enabled() -> bool:
    """Return whether the captured single-replay codec step is enabled.

    On by default once the paired A/B on the ranked configuration has run.
    The graph binds one request's Generator for the life of the process, and
    ``_codec_step_graph_for`` keeps a request that is already sampling eagerly
    on the eager path, so a second concurrent request simply does not get it.
    Set ``VLLM_OMNI_A14_GRAPH=0`` to force the eager chain.
    """
    configured = os.environ.get(_A14_GRAPH_ENV)
    if configured is None:
        return True
    value = configured.strip().lower()
    if value in {"0", "false", "off", ""}:
        return False
    if value in {"1", "true", "on"}:
        return True
    raise RuntimeError(f"{_A14_GRAPH_ENV} must be 0 or 1, got {configured!r}")


def make_device_state(
    codes: torch.Tensor,
    *,
    step: int,
    max_tokens: int,
    finished: bool,
) -> TalkerCodecDeviceState:
    """Create one persistent B=1 state at request/segment setup time."""
    recent = codes.reshape(-1)[-HISTORY_WINDOW:].to(dtype=torch.int32)
    history = torch.zeros((1, HISTORY_WINDOW), dtype=torch.int32, device=codes.device)
    if recent.numel():
        history[0, : recent.numel()].copy_(recent)
    return TalkerCodecDeviceState(
        history=history,
        history_len=torch.tensor([recent.numel()], dtype=torch.int32, device=codes.device),
        step=torch.tensor([step], dtype=torch.int32, device=codes.device),
        max_tokens=torch.tensor([max_tokens], dtype=torch.int32, device=codes.device),
        finished=torch.tensor([finished], dtype=torch.bool, device=codes.device),
    )


def _extension_loaded(source: str) -> bool:
    """Record that the fused boundary is live, and from where.

    Under the default ``auto`` there is no other way to tell from a log
    whether a run used the operator or the native sampler -- the fallback is
    silent by design on a host without the payload, and both produce audio.
    """
    global _extension_available
    _extension_available = True
    logger.info("A14 fused codec boundary active, loaded from %s", source)
    return True


def _auto_fallback(message: str, error: Exception | None = None) -> bool:
    if error is None:
        logger.warning("%s; using the native codec sampler", message)
    else:
        logger.warning("%s; using the native codec sampler: %s", message, error)
    return False


def _load_configured_extension() -> bool:
    """Load a packaged production bridge or an explicit development override."""
    global _extension_load_attempted, _extension_available, _opapi_handle
    mode = a14_mode()
    if mode == "off":
        return False
    if _extension_load_attempted:
        return _extension_available
    _extension_load_attempted = True
    extension = os.environ.get(_A14_EXTENSION_ENV)
    if not extension:
        from vllm_omni._a14_opp import npu_ops_module

        module = npu_ops_module()
        if module is None:
            if mode == "auto":
                return _auto_fallback("no A14 operator payload is installed")
            raise RuntimeError(f"{_A14_MODE_ENV}=required but no A14 operator payload is installed")
        try:
            module.load()
        except (OSError, RuntimeError) as error:
            # Includes UnsupportedSocError: a payload built for another
            # Ascend generation is a fallback, not a crash.
            if mode == "auto":
                return _auto_fallback(f"failed to load the A14 operator payload from {module.__name__}", error)
            raise RuntimeError(f"failed to load the A14 operator payload from {module.__name__}") from error
        return _extension_loaded(module.__name__)
    if not os.path.isfile(extension):
        if mode == "auto":
            return _auto_fallback(f"{_A14_EXTENSION_ENV} is not a file: {extension}")
        raise RuntimeError(f"{_A14_EXTENSION_ENV} is not a file: {extension}")
    opapi = os.environ.get(_A14_OPAPI_ENV)
    try:
        if opapi:
            if not os.path.isfile(opapi):
                raise RuntimeError(f"{_A14_OPAPI_ENV} is not a file: {opapi}")
            # vLLM-Ascend already loads a different libcust_opapi.so. Load the
            # A14 library by absolute path so its symbols enter global scope.
            _opapi_handle = ctypes.CDLL(opapi, mode=ctypes.RTLD_GLOBAL)
        torch.ops.load_library(extension)
    except (OSError, RuntimeError) as error:
        if mode == "auto":
            return _auto_fallback(f"failed to load {_A14_EXTENSION_ENV}={extension}", error)
        raise RuntimeError(f"failed to load {_A14_EXTENSION_ENV}={extension}") from error
    return _extension_loaded(extension)


def load_operator_bridge() -> bool:
    """Load the shared operator payload, and say whether it is live.

    The payload carries more than the codec sampler now -- the Talker's
    decode attention is in the same vendor package and the same bridge -- so
    this is the entry point for anything that needs the payload, not only this
    module. Loading is idempotent and attempted once per process.
    """
    return _load_configured_extension()


def _custom_op():
    if not _load_configured_extension():
        return None
    namespace = getattr(torch.ops, "vllm_omni_npu", None)
    return getattr(namespace, "talker_fused_codec_sample", None) if namespace is not None else None


def _prepare_custom_op():
    if not _load_configured_extension():
        return None
    namespace = getattr(torch.ops, "vllm_omni_npu", None)
    op = getattr(namespace, "talker_codec_logits_prepare", None) if namespace is not None else None
    if op is not None:
        return op
    namespace = getattr(torch.ops, "npu", None)
    op = getattr(namespace, "talker_codec_logits_prepare", None) if namespace is not None else None
    if op is None and a14_mode() == "required":
        raise RuntimeError(
            f"{_A14_MODE_ENV}=required but talker_codec_logits_prepare was not registered"
        )
    return op


def a14_accelerated() -> bool:
    """Whether the fused production boundary is live in this worker."""
    return _prepare_custom_op() is not None


_SAMPLE_V2_ENV = "VLLM_OMNI_A14_SAMPLE_V2"


def _sample_advance_op():
    """The fused Gumbel-max sampler + state transition, or None.

    One launch replaces softmax, `torch.multinomial`'s ~8-kernel decomposition
    and the ~15 elementwise kernels of `codec_sample_result` -- together
    ~90 us of the ~1.1 ms frame. Sampling stays multinomial-in-distribution
    (argmax(logits + Gumbel(u)) with u drawn from the same per-request
    generator); the draw *sequence* differs from `torch.multinomial`, so codec
    tokens differ the way any RNG change differs, and the gate is WER/SIM.
    Off with ``VLLM_OMNI_A14_SAMPLE_V2=0``.
    """
    if os.environ.get(_SAMPLE_V2_ENV, "1") != "1":
        return None
    if not _load_configured_extension():
        return None
    namespace = getattr(torch.ops, "vllm_omni_npu", None)
    return getattr(namespace, "talker_codec_sample_advance", None) if namespace is not None else None


def _torch_logits_prepare_reference(
    raw_logits: torch.Tensor,
    state: TalkerCodecDeviceState,
    min_tokens: torch.Tensor,
    temperature: torch.Tensor,
    repetition_penalty: torch.Tensor,
    *,
    eos_token_id: int,
    top_k: int,
    top_p: float,
    min_tokens_to_keep: int,
) -> torch.Tensor:
    """Graph-capturable reference for the production AscendC boundary."""
    logits = raw_logits.float() / temperature.reshape(1, 1)
    positions = torch.arange(HISTORY_WINDOW, dtype=torch.int32, device=raw_logits.device).reshape(1, -1)
    valid = positions < state.history_len.reshape(1, 1)
    safe_tokens = torch.where(valid, state.history, torch.zeros_like(state.history)).to(torch.long)
    counts = torch.zeros((1, VOCAB_SIZE), dtype=torch.float32, device=raw_logits.device)
    counts.scatter_add_(1, safe_tokens, valid.to(torch.float32))
    alpha = torch.pow(repetition_penalty.reshape(1, 1), counts)
    logits = torch.where(logits < 0, logits * alpha, logits / alpha)
    eos = logits[..., eos_token_id : eos_token_id + 1]
    eos = torch.where(
        (state.step < min_tokens).reshape(1, 1),
        torch.full_like(eos, float("-inf")),
        eos,
    )
    logits = torch.cat([logits[..., :eos_token_id], eos, logits[..., eos_token_id + 1 :]], dim=-1)
    if 0.0 < float(top_p) < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=False, dim=-1)
        cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove = cumulative_probs <= (1.0 - float(top_p))
        remove[..., -min_tokens_to_keep:] = False
        remove = remove.scatter(-1, sorted_indices, remove)
        logits.masked_fill_(remove, float("-inf"))
    if top_k > 0:
        keep = min(VOCAB_SIZE, max(int(top_k), int(min_tokens_to_keep)))
        threshold = torch.topk(logits, keep, dim=-1).values[..., -1, None]
        logits.masked_fill_(logits < threshold, float("-inf"))
    return logits


def prepare_codec_logits(
    raw_logits: torch.Tensor,
    state: TalkerCodecDeviceState,
    min_tokens: torch.Tensor,
    temperature: torch.Tensor,
    repetition_penalty: torch.Tensor,
    *,
    eos_token_id: int,
    top_k: int,
    top_p: float,
    min_tokens_to_keep: int = MIN_TOKENS_TO_KEEP,
) -> torch.Tensor:
    """Prepare and filter logits without changing native NPU RNG semantics."""
    op = _prepare_custom_op()
    if op is None:
        return _torch_logits_prepare_reference(
            raw_logits,
            state,
            min_tokens,
            temperature,
            repetition_penalty,
            eos_token_id=eos_token_id,
            top_k=top_k,
            top_p=top_p,
            min_tokens_to_keep=min_tokens_to_keep,
        )
    return op(
        raw_logits,
        state.history,
        state.history_len,
        state.step,
        min_tokens,
        temperature,
        repetition_penalty,
        int(VOCAB_SIZE),
        int(eos_token_id),
        int(HISTORY_WINDOW),
        int(top_k),
        float(top_p),
        int(min_tokens_to_keep),
    )


def advance_codec_device_state(
    state: TalkerCodecDeviceState,
    sampled_token: torch.Tensor,
    *,
    eos_token_id: int,
) -> TalkerCodecDeviceState:
    """Advance persistent stochastic state with graph-capturable tensor ops."""
    sampled = sampled_token.reshape(1).to(torch.int32)
    active = ~state.finished
    next_step = torch.where(active, state.step + 1, state.step)
    is_eos = sampled == int(eos_token_id)
    reached_limit = active & (next_step >= state.max_tokens)
    finished = state.finished | is_eos | reached_limit
    emit = active & (~is_eos) & (~reached_limit)
    shifted = torch.cat([state.history[:, 1:], sampled.reshape(1, 1)], dim=1)
    append_at = state.history_len.clamp(min=0, max=HISTORY_WINDOW - 1).to(torch.long).reshape(1, 1)
    appended = state.history.scatter(1, append_at, sampled.reshape(1, 1))
    candidate = torch.where((state.history_len >= HISTORY_WINDOW).reshape(1, 1), shifted, appended)
    history = torch.where(emit.reshape(1, 1), candidate, state.history)
    history_len = torch.clamp(state.history_len + emit.to(torch.int32), max=HISTORY_WINDOW)
    return TalkerCodecDeviceState(history, history_len, next_step, state.max_tokens, finished)


def codec_sample_result(
    state: TalkerCodecDeviceState,
    sampled_token: torch.Tensor,
    *,
    eos_token_id: int,
) -> TalkerCodecSampleResult:
    """Advance device state and retain the device-side emit decision."""
    sampled = sampled_token.reshape(1).to(torch.int32)
    active = ~state.finished
    next_step = torch.where(active, state.step + 1, state.step)
    emit = active & (sampled != int(eos_token_id)) & (next_step < state.max_tokens)
    next_state = advance_codec_device_state(state, sampled, eos_token_id=eos_token_id)
    return TalkerCodecSampleResult(sampled_token=sampled, state=next_state, emit=emit)


def _torch_greedy_reference(
    raw_logits: torch.Tensor,
    state: TalkerCodecDeviceState,
    min_tokens: torch.Tensor,
    repetition_penalty: torch.Tensor,
    *,
    top_k: int,
    eos_token_id: int,
) -> TalkerCodecSampleResult:
    positions = torch.arange(HISTORY_WINDOW, dtype=torch.int32, device=raw_logits.device).reshape(1, -1)
    valid = positions < state.history_len.reshape(1, 1)
    safe_tokens = torch.where(valid, state.history, torch.zeros_like(state.history)).to(torch.long)
    counts = torch.zeros((1, VOCAB_SIZE), dtype=torch.float32, device=raw_logits.device)
    counts.scatter_add_(1, safe_tokens, valid.to(torch.float32))
    alpha = torch.pow(repetition_penalty.reshape(1, 1), counts)
    penalized = torch.where(raw_logits < 0, raw_logits * alpha, raw_logits / alpha)

    eos = penalized[..., eos_token_id : eos_token_id + 1]
    eos = torch.where(
        (state.step < min_tokens).reshape(1, 1),
        torch.full_like(eos, float("-inf")),
        eos,
    )
    penalized = torch.cat([penalized[..., :eos_token_id], eos, penalized[..., eos_token_id + 1 :]], dim=-1)
    if top_k > 0:
        keep = min(VOCAB_SIZE, max(int(top_k), MIN_TOKENS_TO_KEEP))
        threshold = torch.topk(penalized, keep, dim=-1).values[..., -1, None]
        penalized = penalized.masked_fill(penalized < threshold, float("-inf"))
    sampled = torch.argmax(penalized, dim=-1).to(torch.int32)

    active = ~state.finished
    next_step = torch.where(active, state.step + 1, state.step)
    is_eos = sampled.to(torch.int64) == int(eos_token_id)
    reached_limit = active & (next_step >= state.max_tokens)
    finished = state.finished | is_eos | reached_limit
    emit = active & (~is_eos) & (~reached_limit)

    shifted = torch.cat([state.history[:, 1:], sampled.to(torch.int32).unsqueeze(1)], dim=1)
    append_at = state.history_len.clamp(min=0, max=HISTORY_WINDOW - 1).to(torch.long).unsqueeze(1)
    appended = state.history.scatter(1, append_at, sampled.to(torch.int32).unsqueeze(1))
    candidate = torch.where((state.history_len >= HISTORY_WINDOW).unsqueeze(1), shifted, appended)
    history = torch.where(emit.unsqueeze(1), candidate, state.history)
    history_len = torch.clamp(state.history_len + emit.to(torch.int32), max=HISTORY_WINDOW)
    return TalkerCodecSampleResult(
        sampled_token=sampled,
        state=TalkerCodecDeviceState(history, history_len, next_step, state.max_tokens, finished),
        emit=emit,
    )


def greedy_codec_sample(
    raw_logits: torch.Tensor,
    state: TalkerCodecDeviceState,
    min_tokens: torch.Tensor,
    repetition_penalty: torch.Tensor,
    *,
    top_k: int,
    eos_token_id: int,
) -> TalkerCodecSampleResult:
    """Call the AscendC op when installed, otherwise the exact torch reference."""
    op = _custom_op()
    if op is None:
        return _torch_greedy_reference(
            raw_logits,
            state,
            min_tokens,
            repetition_penalty,
            top_k=top_k,
            eos_token_id=eos_token_id,
        )
    sampled, history, history_len, step, finished, emit = op(
        raw_logits,
        state.history,
        state.history_len,
        state.step,
        min_tokens,
        state.max_tokens,
        state.finished,
        repetition_penalty,
        int(VOCAB_SIZE),
        int(eos_token_id),
        int(HISTORY_WINDOW),
        int(top_k),
        int(MIN_TOKENS_TO_KEEP),
    )
    return TalkerCodecSampleResult(
        sampled_token=sampled,
        state=TalkerCodecDeviceState(history, history_len, step, state.max_tokens, finished),
        emit=emit,
    )


class CodecStepGraph:
    """One captured Talker post-head codec step, bound to one request at a time.

    Everything between the Talker hidden state and the next codec token runs as
    a single graph replay: ``head_code``, the A14 filter, softmax,
    ``torch.multinomial`` and the state transition. Only the hidden slice is
    copied in, and only the four tensors a consumer outlives are copied out.

    The generator is registered with the graph, so replay advances the same
    request-local RNG stream that the eager path would have advanced. The
    sequence is bit-exact against eager -- see
    ``probes/probe_codec_step_graph.py``.
    """

    def __init__(
        self,
        head: torch.nn.Module,
        *,
        device: torch.device,
        hidden_dtype: torch.dtype,
        hidden_size: int,
        eos_token_id: int,
        top_k: int,
        top_p: float,
        min_tokens_to_keep: int,
        seed: int,
        row_continue: torch.Tensor,
        row_stop: torch.Tensor,
    ) -> None:
        self._head = head
        self._device = device
        self._eos_token_id = int(eos_token_id)
        self._top_k = int(top_k)
        self._top_p = float(top_p)
        self._min_tokens_to_keep = int(min_tokens_to_keep)
        self._seed = int(seed)
        self.owner: str | None = None

        self._hidden = torch.zeros((1, hidden_size), dtype=hidden_dtype, device=device)
        self._state = TalkerCodecDeviceState(
            history=torch.zeros((1, HISTORY_WINDOW), dtype=torch.int32, device=device),
            history_len=torch.zeros(1, dtype=torch.int32, device=device),
            step=torch.zeros(1, dtype=torch.int32, device=device),
            max_tokens=torch.zeros(1, dtype=torch.int32, device=device),
            finished=torch.zeros(1, dtype=torch.bool, device=device),
        )
        self._min_tokens = torch.zeros(1, dtype=torch.int32, device=device)
        self._temperature = torch.zeros(1, dtype=torch.float32, device=device)
        self._penalty = torch.zeros(1, dtype=torch.float32, device=device)

        self._row_continue = row_continue.reshape(-1).to(device=device).contiguous()
        self._row_stop = row_stop.reshape(-1).to(device=device).contiguous()
        self._invalid_delta = torch.full((1, 1), -1, dtype=torch.int64, device=device)

        # Fused-sampler scratch: the graph bakes these six buffers as the op's
        # out-parameters, and the state copies them back in-graph.
        self._noise = torch.zeros((1, VOCAB_SIZE), dtype=torch.float32, device=device)
        self._v2_sampled = torch.zeros(1, dtype=torch.int32, device=device)
        self._v2_history = torch.zeros((1, HISTORY_WINDOW), dtype=torch.int32, device=device)
        self._v2_history_len = torch.zeros(1, dtype=torch.int32, device=device)
        self._v2_step = torch.zeros(1, dtype=torch.int32, device=device)
        self._v2_finished = torch.zeros(1, dtype=torch.bool, device=device)
        self._v2_emit = torch.zeros(1, dtype=torch.bool, device=device)

        self._out_token = torch.zeros(1, dtype=torch.int64, device=device)
        self._out_emit = torch.zeros(1, dtype=torch.bool, device=device)
        self._out_finished = torch.zeros(1, dtype=torch.bool, device=device)
        self._out_delta = torch.zeros((1, 1), dtype=torch.int64, device=device)
        self._out_stop = torch.zeros_like(self._row_continue)

        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(self._seed)
        self._graph: torch.npu.NPUGraph | None = None

    # -- capture ---------------------------------------------------------
    def _body(self) -> None:
        logits = prepare_codec_logits(
            self._head(self._hidden).float(),
            self._state,
            self._min_tokens,
            self._temperature,
            self._penalty,
            eos_token_id=self._eos_token_id,
            top_k=self._top_k,
            top_p=self._top_p,
            min_tokens_to_keep=self._min_tokens_to_keep,
        )
        fused = _sample_advance_op()
        if fused is not None:
            torch.rand(
                (1, VOCAB_SIZE),
                generator=self.generator,
                out=self._noise,
                device=self._noise.device,
                dtype=self._noise.dtype,
            )
            fused(
                logits,
                self._noise,
                self._state.history,
                self._state.history_len,
                self._state.step,
                self._state.finished,
                self._state.max_tokens,
                self._eos_token_id,
                self._v2_sampled,
                self._v2_history,
                self._v2_history_len,
                self._v2_step,
                self._v2_finished,
                self._v2_emit,
            )
            token = self._v2_sampled.reshape(1, 1).to(torch.int64)
            self._out_token.copy_(token.reshape(1))
            self._out_emit.copy_(self._v2_emit)
            self._out_finished.copy_(self._v2_finished)
            self._out_delta.copy_(torch.where(self._v2_emit.reshape(1, 1), token, self._invalid_delta))
            self._out_stop.copy_(
                torch.where(self._v2_finished, self._row_stop, self._row_continue)
            )
            self._state.history.copy_(self._v2_history)
            self._state.history_len.copy_(self._v2_history_len)
            self._state.step.copy_(self._v2_step)
            self._state.finished.copy_(self._v2_finished)
            return

        probabilities = torch.softmax(logits, dim=-1)
        sampled = torch.multinomial(
            probabilities, num_samples=1, generator=self.generator
        ).reshape(())
        result = codec_sample_result(self._state, sampled, eos_token_id=self._eos_token_id)
        token = result.sampled_token.reshape(1, 1).to(torch.int64)

        self._out_token.copy_(token.reshape(1))
        self._out_emit.copy_(result.emit.reshape(1))
        self._out_finished.copy_(result.state.finished.reshape(1))
        self._out_delta.copy_(torch.where(result.emit.reshape(1, 1), token, self._invalid_delta))
        self._out_stop.copy_(
            torch.where(result.state.finished.reshape(1), self._row_stop, self._row_continue)
        )

        # Feed the transition back into the same buffers the graph reads, so a
        # replay needs no host round trip to advance the segment.
        self._state.history.copy_(result.state.history)
        self._state.history_len.copy_(result.state.history_len)
        self._state.step.copy_(result.state.step)
        self._state.finished.copy_(result.state.finished)

    def capture(self) -> None:
        """Warm up and capture. Costs ~15 ms once per process."""
        if self._graph is not None:
            return
        import torch_npu  # noqa: F401  (registers torch.npu)

        # The A14 bridge and its op-api must already be resident: dlopen during
        # capture would be recorded as host work the replay cannot repeat.
        _load_configured_extension()

        warmup_stream = torch.npu.Stream()
        warmup_stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(warmup_stream):
            for _ in range(3):
                self._body()
        torch.npu.current_stream().wait_stream(warmup_stream)
        torch.npu.synchronize()

        graph = torch.npu.NPUGraph()
        graph.register_generator_state(self.generator)
        with torch.npu.graph(graph):
            self._body()
        torch.npu.synchronize()
        self._graph = graph

    # -- binding ---------------------------------------------------------
    @property
    def hidden_dtype(self) -> torch.dtype:
        return self._hidden.dtype

    def matches(self, *, eos_token_id: int, top_k: int, top_p: float, min_tokens_to_keep: int) -> bool:
        """Whether the host-side filter attributes are the captured ones.

        ``temperature``, ``repetition_penalty``, ``min_tokens`` and
        ``max_tokens`` are device tensors and may change per segment; these four
        are baked into the capture and may not.
        """
        return (
            int(eos_token_id) == self._eos_token_id
            and int(top_k) == self._top_k
            and float(top_p) == self._top_p
            and int(min_tokens_to_keep) == self._min_tokens_to_keep
        )

    def bind(
        self,
        request_id: str,
        state: TalkerCodecDeviceState,
        min_tokens: torch.Tensor,
        temperature: torch.Tensor,
        penalty: torch.Tensor,
    ) -> None:
        """Adopt one segment's starting state. Does not touch the RNG stream."""
        self._state.history.copy_(state.history)
        self._state.history_len.copy_(state.history_len)
        self._state.step.copy_(state.step)
        self._state.max_tokens.copy_(state.max_tokens)
        self._state.finished.copy_(state.finished)
        self._min_tokens.copy_(min_tokens)
        self._temperature.copy_(temperature)
        self._penalty.copy_(penalty)
        self.owner = request_id

    def seed_generator(self) -> None:
        """Start a fresh request-local stream, exactly as production does."""
        self.generator.manual_seed(self._seed)

    def release(self, request_id: str) -> None:
        if self.owner == request_id:
            self.owner = None

    # -- steady state ----------------------------------------------------
    def step(self, hidden_state: torch.Tensor) -> TalkerCodecSampleResult:
        if self._graph is None:
            raise RuntimeError("CodecStepGraph.step() before capture()")
        self._hidden.copy_(hidden_state)
        self._graph.replay()
        # The graph writes fixed addresses that the next frame overwrites, so
        # hand consumers their own copies of anything that outlives this step.
        return TalkerCodecSampleResult(
            sampled_token=self._out_token.clone(),
            state=TalkerCodecDeviceState(
                history=self._state.history,
                history_len=self._state.history_len,
                step=self._state.step,
                max_tokens=self._state.max_tokens,
                finished=self._out_finished.clone(),
            ),
            emit=self._out_emit.clone(),
            delta=self._out_delta.clone(),
            stop_row=self._out_stop.clone(),
        )
