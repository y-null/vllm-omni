"""Runtime discovery and loading for the Ascend operators shipped in-tree."""

from .runtime import (
    OperatorPackageError,
    UnsupportedSocError,
    activate,
    check_soc_supported,
    current_soc,
    environment,
    load,
    package_info,
    probe_soc,
    self_test,
    staged_socs,
    supported_socs,
)

__all__ = [
    "OperatorPackageError",
    "UnsupportedSocError",
    "activate",
    "check_soc_supported",
    "current_soc",
    "environment",
    "load",
    "package_info",
    "probe_soc",
    "self_test",
    "staged_socs",
    "supported_socs",
]
