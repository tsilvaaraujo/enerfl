from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProfilerMetrics:
    rounds: dict[int, dict[str, dict[str, Any]]] = field(default_factory=dict)
    global_metrics: dict[str, Any] = field(default_factory=dict)

    def record(self, round_num: int, phase: str, metrics: dict[str, Any]) -> None:
        self.rounds.setdefault(round_num, {}).setdefault(phase, {}).update(metrics)

    def record_global(self, name: str, value: Any) -> None:
        self.global_metrics[name] = value

    def get_flat_metrics(self, round_num: int | None = None) -> dict[str, Any]:
        result = {}
        for r in ([round_num] if round_num is not None else self.rounds.keys()):
            if r in self.rounds:
                for phase_metrics in self.rounds[r].values():
                    result.update(phase_metrics)
        return result

    def to_dict(self) -> dict:
        return {"global": self.global_metrics, "rounds": dict(self.rounds)}
