"""Locate and activate the custom OPP resources shipped inside this package.

The payload travels in the main ``vllm_omni`` distribution rather than in a
companion wheel, because the ranked evaluation installs exactly one
distribution (``pip install -e . --no-build-isolation``) and never sees a
second one.

One payload serves every supported Ascend SoC, but not by merging them. The
A2 and A3 builds turned out to differ in more than their ``kernel/<soc>``
directories: the kernel is retuned per chip (the vocabulary split is 3072/3520
UB chunks on 910C against 4096/2496 on 910B) and each board's CANN compiles
its own ``libcust_opapi.so``, ``liboptiling.so`` and dispatcher bridge -- A3 on
CANN 9.0.0, A2 on 9.1.0. Only one copy of each of those can win, so a merged
vendor tree would silently ship one board's host libraries to the other.

The payload is therefore partitioned at the top: ``_payload/socs/<soc>/``, each
holding a complete self-consistent ``vendors/`` tree and its own ``lib/``
bridge. Exactly one is selected for the running chip.
"""

from __future__ import annotations

import ctypes
import json
import os
import re
import subprocess
import sys
from importlib.resources import files
from pathlib import Path
from typing import Any


class OperatorPackageError(RuntimeError):
    """The shipped operator payload is incomplete or cannot be loaded."""


class UnsupportedSocError(OperatorPackageError):
    """The payload carries no kernel binary for the running Ascend SoC."""


class _Unset:
    """Distinguishes "not probed yet" from "probed, and unknowable"."""


_UNSET = _Unset()
# Names npu-smi's plain table reports that identify a generation but not a
# board -- A3 answers "Ascend910" for a 910_9382, which matches nothing staged.
_AMBIGUOUS_FAMILIES = frozenset({"ascend910"})
_opapi_handle: ctypes.CDLL | None = None
_loaded_bridge: Path | None = None


def _payload_root() -> Path:
    resource = files("vllm_omni.npu_ops").joinpath("_payload")
    # Deliberately not zip-safe: CANN and dlopen both require stable
    # filesystem paths for the OPP tree and the shared libraries.
    return Path(str(resource))


def _single_match(root: Path, pattern: str, description: str) -> Path:
    matches = sorted(path for path in root.glob(pattern) if path.is_file())
    if len(matches) != 1:
        raise OperatorPackageError(
            f"expected one {description} matching {root / pattern}, found {len(matches)}"
        )
    return matches[0]


def staged_socs() -> tuple[str, ...]:
    """Return the SoC families this payload carries a complete build for."""
    socs_root = _payload_root() / "socs"
    if not socs_root.is_dir():
        return ()
    return tuple(sorted(path.name for path in socs_root.iterdir() if path.is_dir()))


def _soc_root() -> Path:
    """Return the staged directory for the running chip.

    Selection happens here rather than at load time because CANN reads custom
    OPP metadata during process initialization, so the vendor path has to be
    on the environment before torch-npu brings CANN up -- which is also why
    this must not import torch.
    """
    staged = staged_socs()
    if not staged:
        raise OperatorPackageError(f"payload stages no SoC builds under {_payload_root() / 'socs'}")
    soc = probe_soc()
    if soc is not None:
        matched = [candidate for candidate in staged if soc.startswith(candidate)]
        if not matched:
            raise UnsupportedSocError(
                f"no A14 build for SoC {soc!r}; payload carries {', '.join(staged)}"
            )
        return _payload_root() / "socs" / max(matched, key=len)
    if len(staged) == 1:
        # A single-board payload -- the submitted one -- needs no probe at all.
        return _payload_root() / "socs" / staged[0]
    raise OperatorPackageError(
        f"cannot identify the running Ascend SoC to choose between {', '.join(staged)}; "
        "set VLLM_OMNI_A14_SOC to one of them"
    )


def _paths() -> tuple[Path, Path, Path, Path]:
    root = _soc_root()
    vendors_root = root / "vendors"
    vendor_dirs = sorted(path for path in vendors_root.iterdir() if path.is_dir()) if vendors_root.is_dir() else []
    if len(vendor_dirs) != 1:
        raise OperatorPackageError(
            f"expected one staged OPP vendor under {vendors_root}, found {len(vendor_dirs)}"
        )
    vendor = vendor_dirs[0]
    opapi = _single_match(vendor, "op_api/lib/libcust_opapi.so", "custom op-api library")
    bridge = _single_match(root, "lib/_vllm_omni_a14_prepare_npu*.so", "A14 dispatcher bridge")
    return root, vendor, opapi, bridge


def environment() -> dict[str, str]:
    """Return environment entries required by the packaged OPP."""
    _, vendor, opapi, bridge = _paths()
    current = [part for part in os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(os.pathsep) if part]
    vendor_text = str(vendor)
    opp_path = os.pathsep.join([vendor_text, *(part for part in current if part != vendor_text)])
    return {
        "ASCEND_CUSTOM_OPP_PATH": opp_path,
        "VLLM_OMNI_A14_OPAPI_LIBRARY": str(opapi),
        "VLLM_OMNI_A14_EXTENSION": str(bridge),
    }


def activate() -> dict[str, str]:
    """Activate this package's private OPP for the current process.

    Called at ``import vllm_omni`` time, before torch-npu initializes CANN,
    because CANN reads custom OPP metadata only during process init. It must
    therefore not touch torch or the device -- the SoC check belongs in
    ``load()``, which first runs once a worker is about to sample.
    """
    configured = environment()
    os.environ.update(configured)
    return configured


def supported_socs() -> tuple[str, ...]:
    """Kept as the public name for what the payload can serve."""
    return staged_socs()


_probed_soc: str | None | _Unset = _UNSET


def probe_soc() -> str | None:
    """Return the running Ascend SoC in lowercase, without importing torch.

    Called during ``import vllm_omni``, before torch-npu initializes CANN, so
    it cannot use ``torch_npu.npu.get_device_name`` -- importing torch-npu here
    would both cost seconds and reorder the very initialization this exists to
    get ahead of. Sources, cheapest first:

    1. ``VLLM_OMNI_A14_SOC``, then ``SOC_VERSION`` / ``ASCEND_SOC_VERSION``.
       CANN launch environments commonly export the exact chip revision --
       the A3 box has ``SOC_VERSION=ascend910_9391`` in its shell already.
    2. torch-npu, but only if some earlier import already brought it in, in
       which case asking is free.
    3. ``npu-smi``. Its plain table is enough on A2, whose Name column reads
       ``910B3``, but not on A3, where it reads a bare ``Ascend910`` that
       prefix-matches nothing we stage. There the per-chip query supplies the
       missing revision as a separate ``NPU Name: 9382`` field, and the two
       compose into ``ascend910_9382``.

    Chips report a narrower name than the kernel directories are named for
    (``Ascend910B3`` against ``ascend910b``, ``Ascend910_9382`` against
    ``ascend910_93``), so callers prefix-match rather than compare. A bare
    family name that matches no staged build is treated as unknown rather than
    forced, so an ambiguous answer falls back instead of picking a board.
    """
    global _probed_soc
    if _probed_soc is not _UNSET:
        return _probed_soc
    _probed_soc = _probe_soc_uncached()
    return _probed_soc


def _npu_smi(*arguments: str) -> str | None:
    try:
        completed = subprocess.run(
            ["npu-smi", *arguments], capture_output=True, text=True, timeout=20, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout


def _field(text: str, label: str) -> str | None:
    match = re.search(rf"^\s*{re.escape(label)}\s*:\s*(\S+)\s*$", text, re.MULTILINE)
    return match.group(1) if match else None


def _normalize(name: str) -> str:
    name = name.strip().lower()
    return name if name.startswith("ascend") else f"ascend{name}"


def _probe_soc_uncached() -> str | None:
    for variable in ("VLLM_OMNI_A14_SOC", "SOC_VERSION", "ASCEND_SOC_VERSION"):
        configured = os.environ.get(variable, "").strip().lower()
        if configured:
            return configured

    torch_npu = sys.modules.get("torch_npu")
    if torch_npu is not None:
        try:
            name = torch_npu.npu.get_device_name()
        except Exception:
            name = None
        if isinstance(name, str) and name:
            return name.strip().lower()

    table = _npu_smi("info")
    if table is None:
        return None
    # Rows look like "| 4     910B3               | OK  ...": the NPU id, then
    # the chip name.
    identifier = family = None
    for line in table.splitlines():
        match = re.match(r"\|\s*(\d+)\s+(\S+)\s*\|", line)
        if match:
            identifier, family = match.group(1), _normalize(match.group(2))
            break
    if family is None:
        return None
    if family not in _AMBIGUOUS_FAMILIES:
        return family

    # A3 reports the bare family here and keeps the revision in the per-chip
    # query, as "NPU Name: 9382" beside "Chip Name: Ascend910".
    detail = _npu_smi("info", "-t", "board", "-i", str(identifier), "-c", "0")
    if detail is None:
        return family
    revision = _field(detail, "NPU Name")
    chip = _field(detail, "Chip Name")
    if revision and revision.isdigit() and chip and _normalize(chip) == family:
        return f"{family}_{revision}"
    if chip and _normalize(chip) != family:
        return _normalize(chip)
    return family


def current_soc() -> str | None:
    """Backwards-compatible alias for :func:`probe_soc`."""
    return probe_soc()


def check_soc_supported() -> str | None:
    """Confirm the running chip has a staged build, before anything dispatches.

    ``_soc_root`` already raises when it cannot serve the running chip, so
    this is the named entry point callers use to turn that into an A14 ``auto``
    fallback rather than an opaque CANN dispatch error inside the first codec
    sample.
    """
    return _soc_root().name


def load() -> Path:
    """Load the packaged op-api and torch dispatcher bridge exactly once."""
    global _loaded_bridge, _opapi_handle
    if _loaded_bridge is not None:
        return _loaded_bridge
    configured = activate()
    check_soc_supported()
    opapi = Path(configured["VLLM_OMNI_A14_OPAPI_LIBRARY"])
    bridge = Path(configured["VLLM_OMNI_A14_EXTENSION"])
    try:
        _opapi_handle = ctypes.CDLL(str(opapi), mode=ctypes.RTLD_GLOBAL)
        import torch

        torch.ops.load_library(str(bridge))
    except (ImportError, OSError, RuntimeError) as error:
        raise OperatorPackageError(f"failed to load packaged A14 bridge {bridge}") from error
    _loaded_bridge = bridge
    return bridge


def package_info() -> dict[str, Any]:
    """Describe the installed payload using relocatable package metadata."""
    payload, vendor, opapi, bridge = _paths()
    metadata_path = payload / "build_info.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    return {
        "payload_root": str(payload),
        "vendor": vendor.name,
        "opapi": str(opapi),
        "bridge": str(bridge),
        "socs": list(staged_socs()),
        "selected_soc": _soc_root().name,
        "current_soc": probe_soc(),
        "build": metadata,
    }


def self_test(device: str = "npu:0") -> dict[str, Any]:
    """Validate the fixed ABI and torch-npu task-queue ordering."""
    load()
    import torch

    namespace = getattr(torch.ops, "vllm_omni_npu", None)
    op = getattr(namespace, "talker_codec_logits_prepare", None) if namespace is not None else None
    if op is None:
        raise OperatorPackageError("packaged bridge did not register vllm_omni_npu.talker_codec_logits_prepare")
    # One bridge registers every operator in the payload; a partial
    # registration would otherwise only surface as a silent fallback.
    if getattr(namespace, "talker_decode_attention_out", None) is None:
        raise OperatorPackageError("packaged bridge did not register vllm_omni_npu.talker_decode_attention_out")
    # Keep producers queued immediately before the custom boundary. A direct
    # ACLNN bridge can otherwise overtake torch-npu's task queue and still pass
    # a test that synchronizes or materializes every input first.
    raw_logits = torch.linspace(-3.0, 3.0, 6562, dtype=torch.float32, device=device).reshape(1, 6562)
    history = torch.arange(16, dtype=torch.int32, device=device).reshape(1, 16)

    def scalar_i32(value: int) -> Any:
        return torch.tensor([value], dtype=torch.int32, device=device)

    def scalar_f32(value: float) -> Any:
        return torch.tensor([value], dtype=torch.float32, device=device)

    output = op(
        raw_logits,
        history,
        scalar_i32(16),
        scalar_i32(17),
        scalar_i32(3),
        scalar_f32(0.8),
        scalar_f32(1.02),
        6562,
        6561,
        16,
        20,
        0.95,
        3,
    )
    if tuple(output.shape) != (1, 6562) or output.dtype != torch.float32 or output.device.type != "npu":
        raise OperatorPackageError(
            f"unexpected A14 output: shape={tuple(output.shape)}, dtype={output.dtype}, device={output.device}"
        )
    reference = raw_logits / 0.8
    counts = torch.zeros_like(reference)
    counts.scatter_add_(1, history.to(torch.long), torch.ones_like(history, dtype=torch.float32))
    alpha = torch.pow(torch.full((1,), 1.02, dtype=torch.float32, device=device).reshape(1, 1), counts)
    reference = torch.where(reference < 0, reference * alpha, reference / alpha)
    sorted_logits, sorted_indices = torch.sort(reference, descending=False, dim=-1)
    cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
    remove = cumulative_probs <= 0.05
    remove[..., -3:] = False
    remove = remove.scatter(-1, sorted_indices, remove)
    reference.masked_fill_(remove, float("-inf"))
    threshold = torch.topk(reference, 20, dim=-1).values[..., -1, None]
    reference.masked_fill_(reference < threshold, float("-inf"))
    torch.npu.synchronize()
    output_mask = torch.isneginf(output)
    reference_mask = torch.isneginf(reference)
    if not bool(torch.equal(output_mask, reference_mask)):
        raise OperatorPackageError("A14 filtering mask does not match the torch reference")
    finite = ~reference_mask
    max_abs = float((output[finite] - reference[finite]).abs().max().cpu())
    if max_abs > 1e-5:
        raise OperatorPackageError(
            f"A14 queued-producer ordering check failed: max_abs={max_abs}"
        )
    return {
        "device": str(output.device),
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "queued_producer_max_abs": max_abs,
        "status": "ok",
    }
