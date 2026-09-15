"""Early activation for the A14 Ascend operator resources."""

from __future__ import annotations

import os
import warnings
from typing import Any


def _staged(module: Any | None) -> bool:
    """Whether this module can resolve a complete operator payload."""
    if module is None:
        return False
    try:
        module.environment()
    except Exception:
        return False
    return True


def npu_ops_module() -> Any | None:
    """Return a module with a staged operator payload, in-tree one first.

    The payload now ships inside this distribution (``vllm_omni.npu_ops``),
    because the ranked evaluation installs exactly one distribution. The
    standalone ``vllm_omni_npu_ops`` wheel is still honoured so a host that
    already has it installed keeps working -- and it is what a source checkout
    with no staged payload falls back to.
    """
    try:
        from vllm_omni import npu_ops
    except ImportError:
        npu_ops = None
    if _staged(npu_ops):
        return npu_ops
    try:
        import vllm_omni_npu_ops
    except ModuleNotFoundError as error:
        if error.name == "vllm_omni_npu_ops":
            return None
        raise
    return vllm_omni_npu_ops if _staged(vllm_omni_npu_ops) else None


def activate_required_a14_opp() -> None:
    """Expose the A14 OPP before torch-npu initializes CANN.

    Loading the dispatcher bridge remains lazy, but CANN discovers custom OPP
    metadata only during process initialization. Activating on the first codec
    sample is therefore too late for long-lived Stage workers. ``auto`` keeps
    that early activation when the payload is present and otherwise leaves the
    native sampler untouched.
    """
    configured = os.environ.get("VLLM_OMNI_A14_MODE")
    # Exposing the optional OPP is harmless for other models; actually using
    # the operator remains MiniCPM-stage scoped. Doing this by default is what
    # makes the operator reachable under the official launch, which sets no
    # vLLM-Omni environment variables at all.
    mode = "auto" if configured is None else configured.strip().lower()
    if mode not in {"auto", "required"}:
        return
    module = npu_ops_module()
    if module is None:
        if mode == "auto":
            # A source checkout without a staged payload, or a non-Ascend
            # host. Both are ordinary, so stay silent.
            return
        raise RuntimeError("VLLM_OMNI_A14_MODE=required but no A14 operator payload is installed")
    try:
        module.activate()
    except Exception as error:
        if mode == "required":
            raise
        warnings.warn(
            f"A14 auto activation failed; using the native sampler: {error}",
            RuntimeWarning,
            stacklevel=2,
        )
