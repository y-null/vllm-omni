# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reusable exact-signature NPUGraph capture and replay helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import os
from typing import Protocol

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)


class _ReplayGraph(Protocol):
    def replay(self) -> None: ...


@dataclass
class CapturedDeviceGraph:
    graph: _ReplayGraph
    static_inputs: tuple[torch.Tensor, ...]
    static_outputs: tuple[torch.Tensor, ...]

    def replay(
        self,
        inputs: tuple[torch.Tensor, ...],
        output_into: tuple[torch.Tensor | None, ...] | None = None,
    ) -> tuple[torch.Tensor, ...]:
        with torch.inference_mode():
            for static, current in zip(self.static_inputs, inputs, strict=True):
                static.copy_(current)
            self.graph.replay()
            # Graph outputs are persistent and overwritten by the next replay.
            # Clone before they become request-owned streaming cache entries;
            # when ``output_into`` supplies a caller-owned buffer for an output
            # slot, copy straight into it instead (same stream, so the static
            # buffer stays valid until the copy is issued). The clone skips one
            # large allocation + copy per replayed step, which would otherwise
            # hit the allocator between every estimator step.
            if output_into is None:
                return tuple(output.detach().clone() for output in self.static_outputs)
            outputs: list[torch.Tensor] = []
            for destination, output in zip(output_into, self.static_outputs, strict=True):
                # Slots without a caller buffer must still clone: the static
                # buffer is overwritten by the next replay and must not leak
                # out as if it were request-owned.
                outputs.append(output.detach().clone() if destination is None else destination.copy_(output))
            return tuple(outputs)


def _tensor_signature(value: torch.Tensor) -> tuple[tuple[int, ...], str, str]:
    return tuple(value.shape), str(value.dtype), str(value.device)


class NPUExactGraphRunner:
    """Capture and replay tensor-only functions for exact NPU signatures."""

    def __init__(
        self,
        *,
        max_graphs: int = 32,
        component_name: str = "device graph",
        disable_config_hint: str = "disable graph capture",
    ) -> None:
        self.max_graphs = max(0, int(max_graphs))
        self.component_name = component_name
        self.disable_config_hint = disable_config_hint
        self._enabled = self.max_graphs > 0
        self._graphs: dict[tuple[object, ...], CapturedDeviceGraph] = {}
        self._failed_keys: set[tuple[object, ...]] = set()
        self._hits = 0
        self._graph_pool: object | None = None
        # 第 152 项：捕获准入 HBM 水位地板（字节）。默认 3GiB；0 = 关闭门控。
        _admission_gb = float(
            os.environ.get("VLLM_OMNI_GRAPH_CAPTURE_FREE_HBM_GB", "3")
        )
        self._capture_free_hbm_bytes = max(0.0, _admission_gb) * 2**30

    @staticmethod
    def is_supported() -> bool:
        npu = getattr(torch, "npu", None)
        return npu is not None and all(
            hasattr(npu, name)
            for name in (
                "NPUGraph",
                "graph",
                "is_current_stream_capturing",
                "synchronize",
            )
        )

    @staticmethod
    def _stream_is_capturing() -> bool:
        npu = getattr(torch, "npu", None)
        is_capturing = getattr(npu, "is_current_stream_capturing", None)
        if not callable(is_capturing):
            return False
        try:
            return bool(is_capturing())
        except (RuntimeError, TypeError):
            return False

    def _eligible(self, inputs: tuple[torch.Tensor, ...]) -> bool:
        return (
            self._enabled
            and bool(inputs)
            and all(value.device.type == "npu" for value in inputs)
            and self.is_supported()
            and not self._stream_is_capturing()
        )

    @property
    def stats(self) -> dict[str, int]:
        return {
            "captures": len(self._graphs),
            "failed": len(self._failed_keys),
            "hits": self._hits,
        }

    def capture(
        self,
        inputs: tuple[torch.Tensor, ...],
        compute: Callable[..., tuple[torch.Tensor, ...]],
    ) -> CapturedDeviceGraph:
        npu = torch.npu
        static_inputs = tuple(value.detach().clone() for value in inputs)
        npu.synchronize()
        graph = npu.NPUGraph()
        if self._graph_pool is None:
            from vllm.platforms import current_platform

            self._graph_pool = current_platform.get_global_graph_pool()
        with torch.inference_mode(), npu.graph(graph, pool=self._graph_pool):
            static_outputs = compute(*static_inputs)
        npu.synchronize()
        return CapturedDeviceGraph(
            graph=graph,
            static_inputs=static_inputs,
            static_outputs=static_outputs,
        )

    def run(
        self,
        operation: str,
        inputs: tuple[torch.Tensor, ...],
        constants: tuple[object, ...],
        compute: Callable[..., tuple[torch.Tensor, ...]],
        output_into: tuple[torch.Tensor | None, ...] | None = None,
    ) -> tuple[torch.Tensor, ...]:
        if self._failed_keys:
            raise RuntimeError(
                f"{self.component_name} cannot continue after a failed NPUGraph capture; "
                f"restart the stage process and {self.disable_config_hint} before retrying."
            )
        if not self._eligible(inputs):
            return compute(*inputs)

        key = (
            operation,
            constants,
            tuple(_tensor_signature(value) for value in inputs),
        )
        graph = self._graphs.get(key)
        if graph is not None:
            self._hits += 1
            if self._hits == 1:
                logger.info("%s started NPUGraph replay", self.component_name)
            # Pass the sink tuple only when present: graph objects with the
            # pre-sink replay() signature keep working on the common path.
            if output_into is None:
                return graph.replay(inputs)
            return graph.replay(inputs, output_into)

        # Prime lazy kernels and allocator state before capture. The next call
        # with the same exact tensor signature replays this graph.
        eager_outputs = compute(*inputs)
        if len(self._graphs) >= self.max_graphs:
            logger.warning_once(
                "%s reached the %d-entry NPUGraph limit; new tensor shapes will use eager execution.",
                self.component_name,
                self.max_graphs,
            )
            return eager_outputs
        # 第 152 项：捕获准入门控。HBM 空闲低于阈值时放弃捕获新图（本批走
        # eager），把 OOM 防线从"事后 empty_cache 清理"（第 112 项护栏，e151
        # 实验证明其参数放松即崩）前移为"事前准入"。e151 现场：捕获期与
        # 长音频 offset 组合下 ragged 图持续捕获（10 次/68s）+ 每图 static
        # buffer 常驻，free 掉到 403MiB 后 torch.cat 562MiB 直接 OOM。水位
        # 恢复后同形状自然重试捕获，零状态机。e152 四轮 c8 复验：门控触发
        # 40 次后仍于 free 830MiB 崩（12 张图 buffer 已常驻 + 多轮碎片化，
        # eager cat 1.10GiB 失败）——准入地板升到 3GiB，配合池上限 16->8
        # 把常驻图 buffer 减半。VLLM_OMNI_GRAPH_CAPTURE_FREE_HBM_GB=0 关闭
        # 门控（旧行为）。
        if self._capture_free_hbm_bytes > 0:
            try:
                free_bytes = torch.npu.mem_get_info()[0]
            except Exception:  # pragma: no cover - 非 NPU 环境兜底
                free_bytes = 1 << 62
            if free_bytes < self._capture_free_hbm_bytes:
                logger.warning_once(
                    "%s skipped NPUGraph capture: free HBM %.2f GiB < %.2f GiB "
                    "admission floor (eager fallback, retried when memory frees).",
                    self.component_name,
                    free_bytes / 2**30,
                    self._capture_free_hbm_bytes / 2**30,
                )
                return eager_outputs
        try:
            self._graphs[key] = self.capture(inputs, compute)
        except Exception as exc:
            self._failed_keys.add(key)
            self._enabled = False
            logger.exception(
                "%s failed to capture NPUGraph for %s; the torch-npu "
                "allocator/RNG capture state may be invalid and this stage "
                "process must be restarted.",
                self.component_name,
                operation,
            )
            raise RuntimeError(
                f"{self.component_name} NPUGraph capture failed for {operation}; "
                f"restart the stage process. To run eagerly, {self.disable_config_hint}."
            ) from exc
        logger.info(
            "%s captured NPUGraph %d/%d for %s",
            self.component_name,
            len(self._graphs),
            self.max_graphs,
            operation,
        )
        return eager_outputs
