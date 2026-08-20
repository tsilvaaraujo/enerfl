import time
from abc import ABC, abstractmethod
from typing import Any


class MetricsCollector(ABC):
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def on_start(self, phase: str, round_num: int, **context: Any) -> dict: ...

    @abstractmethod
    def on_end(
        self, phase: str, round_num: int, start_state: dict, **context: Any
    ) -> dict: ...


class TimingCollector(MetricsCollector):
    def name(self) -> str:
        return "timing"

    def on_start(self, phase: str, round_num: int, **context: Any) -> dict:
        # t0 is pre-captured by Profiler.measure() before the collector loop
        return {"t0": context.get("t0", time.perf_counter())}

    def on_end(
        self, phase: str, round_num: int, start_state: dict, **context: Any
    ) -> dict:
        return {f"{phase}_time": time.perf_counter() - start_state["t0"]}


class MessageCountCollector(MetricsCollector):
    def name(self) -> str:
        return "messages"

    def on_start(self, phase: str, round_num: int, **context: Any) -> dict:
        return {}

    def on_end(
        self, phase: str, round_num: int, start_state: dict, **context: Any
    ) -> dict:
        metrics = {}
        if "messages" in context:
            msgs = context["messages"]
            metrics[f"{phase}_sent_count"] = (
                len(msgs) if hasattr(msgs, "__len__") else len(list(msgs))
            )
        if "replies" in context:
            reps = context["replies"]
            metrics[f"{phase}_received_count"] = (
                len(reps) if hasattr(reps, "__len__") else len(list(reps))
            )
        return metrics


class UpdateCountCollector(MetricsCollector):
    def name(self) -> str:
        return "updates"

    def on_start(self, phase: str, round_num: int, **context: Any) -> dict:
        return {}

    def on_end(
        self, phase: str, round_num: int, start_state: dict, **context: Any
    ) -> dict:
        metrics = {}
        if "updates_sent" in context:
            metrics[f"{phase}_updates_sent"] = context["updates_sent"]
        if "updates_received" in context:
            metrics[f"{phase}_updates_received"] = context["updates_received"]
        return metrics


class WeightCountCollector(MetricsCollector):
    def name(self) -> str:
        return "weights"

    def on_start(self, phase: str, round_num: int, **context: Any) -> dict:
        return {}

    def on_end(
        self, phase: str, round_num: int, start_state: dict, **context: Any
    ) -> dict:
        metrics = {}
        if "weights_sent" in context:
            metrics[f"{phase}_weights_sent"] = context["weights_sent"]
        return metrics


class PayloadSizeCollector(MetricsCollector):
    def name(self) -> str:
        return "payload"

    def on_start(self, phase: str, round_num: int, **context: Any) -> dict:
        return {}

    def on_end(
        self, phase: str, round_num: int, start_state: dict, **context: Any
    ) -> dict:
        metrics = {}
        if "payload_bytes_sent" in context:
            metrics[f"{phase}_payload_bytes_sent"] = context["payload_bytes_sent"]
        if "payload_bytes_received" in context:
            metrics[f"{phase}_payload_bytes_received"] = context[
                "payload_bytes_received"
            ]
        return metrics


class CompressionCollector(MetricsCollector):
    def name(self) -> str:
        return "compression"

    def on_start(self, phase: str, round_num: int, **context: Any) -> dict:
        return {}

    def on_end(
        self, phase: str, round_num: int, start_state: dict, **context: Any
    ) -> dict:
        metrics = {}
        if "original_bytes" in context and "compressed_bytes" in context:
            orig = context["original_bytes"]
            comp = context["compressed_bytes"]
            metrics[f"{phase}_original_bytes"] = orig
            metrics[f"{phase}_compressed_bytes"] = comp
            metrics[f"{phase}_compression_ratio"] = (
                orig / comp if comp > 0 else float("inf")
            )
        if "total_params" in context:
            metrics[f"{phase}_total_params"] = context["total_params"]
        if "nonzero_params" in context:
            nz = context["nonzero_params"]
            metrics[f"{phase}_nonzero_params"] = nz
            if "total_params" in context:
                tp = context["total_params"]
                metrics[f"{phase}_sparsity_ratio"] = (
                    1.0 - (nz / tp) if tp > 0 else 0.0
                )
        if "bits_per_param" in context:
            metrics[f"{phase}_bits_per_param"] = context["bits_per_param"]
        return metrics
