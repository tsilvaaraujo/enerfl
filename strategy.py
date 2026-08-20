import time
from collections.abc import Callable
from logging import INFO
from typing import Any, NamedTuple

from flwr.common import ArrayRecord, ConfigRecord, Message, MetricRecord, log
from flwr.serverapp import Grid
from flwr.serverapp.strategy import Strategy
from flwr.serverapp.strategy.strategy import Result, log_strategy_start_info

from .server import Profiler


class _SendReceiveResult(NamedTuple):
    replies: list[Message]
    elapsed: float
    tx_delta: int
    rx_delta: int
    observed: bool


class _ProfiledStrategy(Strategy):
    def __init__(self, strategy: Strategy, profiler: Profiler) -> None:
        self._strategy = strategy
        self._profiler = profiler

    def summary(self) -> None:
        self._strategy.summary()

    def configure_train(self, server_round, arrays, config, grid):
        return self._strategy.configure_train(server_round, arrays, config, grid)

    def aggregate_train(self, server_round, replies):
        return self._strategy.aggregate_train(server_round, replies)

    def configure_evaluate(self, server_round, arrays, config, grid):
        return self._strategy.configure_evaluate(server_round, arrays, config, grid)

    def aggregate_evaluate(self, server_round, replies):
        return self._strategy.aggregate_evaluate(server_round, replies)

    def start(
        self,
        grid: Grid,
        initial_arrays: ArrayRecord,
        num_rounds: int = 3,
        timeout: float = 3600,
        train_config: ConfigRecord | None = None,
        evaluate_config: ConfigRecord | None = None,
        evaluate_fn: (
            Callable[[int, ArrayRecord], MetricRecord | None] | None
        ) = None,
    ) -> Result:
        _t = time.perf_counter
        _rec = self._profiler.record
        p = self._profiler

        log(INFO, "Starting %s (profiled):", self._strategy.__class__.__name__)
        log_strategy_start_info(
            num_rounds, initial_arrays, train_config, evaluate_config
        )
        self.summary()
        log(INFO, "")

        train_config = ConfigRecord() if train_config is None else train_config
        evaluate_config = (
            ConfigRecord() if evaluate_config is None else evaluate_config
        )
        result = Result()
        p.start_carbon()
        t_total_start = _t()

        if evaluate_fn:
            p.set_round(0)
            t0 = _t()
            res = evaluate_fn(0, initial_arrays)
            _rec("initial_evaluation", {"initial_evaluation_time": _t() - t0})
            log(INFO, "Initial global evaluation: %s", res)
            if res is not None:
                result.evaluate_metrics_serverapp[0] = res

        arrays = initial_arrays

        for current_round in range(1, num_rounds + 1):
            p.set_round(current_round)
            log(INFO, "")
            log(INFO, "[ROUND %s/%s]", current_round, num_rounds)
            t_round_start = _t()

            train_ctx = self._profile_train_phase(
                grid, current_round, arrays, train_config, timeout, _t, _rec,
                result,
            )

            eval_ctx = self._profile_evaluate_phase(
                grid, current_round, arrays, evaluate_config, timeout, _t, _rec,
                result,
            )

            if evaluate_fn:
                t0 = _t()
                server_res = evaluate_fn(current_round, arrays)
                _rec(
                    "server_evaluation",
                    {"server_evaluation_time": _t() - t0},
                )
                if server_res is not None:
                    log(INFO, "\t\u2514\u2500\u2500> Server eval: %s", server_res)
                    result.evaluate_metrics_serverapp[current_round] = server_res

            round_total_time = _t() - t_round_start
            round_metrics: dict[str, Any] = {"round_total_time": round_total_time}
            efficiency = _round_efficiency(
                round_total_time,
                train_ctx.get("client_computation"),
                eval_ctx.get("client_computation"),
            )
            if efficiency is not None:
                round_metrics["round_efficiency"] = efficiency
            _rec("round_total", round_metrics)

            if train_ctx.get("agg_arrays") is not None:
                arrays = train_ctx["agg_arrays"]
            p.write_round_to_csv(current_round)

        self._record_session_totals(num_rounds, _t() - t_total_start)
        return result

    def _profile_train_phase(
        self, grid, current_round, arrays, train_config, timeout, _t, _rec,
        result,
    ) -> dict[str, Any]:
        p = self._profiler
        net_iface = getattr(p, "_network_interface", None)

        t0 = _t()
        train_msgs = list(
            self.configure_train(current_round, arrays, train_config, grid)
        )
        _rec("configure_train", {
            "configure_train_time": _t() - t0,
            "messages_configured": len(train_msgs),
        })

        sr = _measure_send_receive_with_network_counters(
            grid=grid,
            messages=train_msgs,
            timeout=timeout,
            iface=net_iface,
        )
        _rec(
            "send_receive_train",
            {
                "send_receive_train_time": sr.elapsed,
                "updates_sent": len(train_msgs),
                "updates_received": len(sr.replies),
            },
        )

        train_payload_sent = sr.tx_delta if sr.observed else 0
        train_payload_received = sr.rx_delta if sr.observed else 0
        train_replies = sr.replies

        p._cumulative_bytes_sent += train_payload_sent
        p._cumulative_bytes_received += train_payload_received

        comm: dict[str, Any] = {
            "send_receive_train_time": sr.elapsed,
            "train_payload_bytes_sent": train_payload_sent,
            "train_payload_bytes_received": train_payload_received,
            "updates_sent": len(train_msgs),
            "updates_received": len(train_replies),
            "cumulative_bytes_sent": p._cumulative_bytes_sent,
            "cumulative_bytes_received": p._cumulative_bytes_received,
        }

        client_times = _client_total_times(train_replies)
        client_computation = max(client_times) if client_times else None
        if client_times:
            network_overhead = max(0.0, sr.elapsed - client_computation)
            comm["client_computation_train"] = client_computation
            comm["network_overhead_train"] = network_overhead
            comm["straggler_index_train"] = client_computation - min(client_times)
            if client_computation > 0:
                comm["comm_to_compute_ratio_train"] = (
                    network_overhead / client_computation
                )

        if sr.observed and sr.elapsed > 0:
            comm["bandwidth_utilization_bytes_per_sec_train"] = (
                train_payload_sent + train_payload_received
            ) / sr.elapsed

        _rec("communication_train", comm)

        t0 = _t()
        agg_arrays, agg_metrics = self.aggregate_train(
            current_round, train_replies,
        )
        _rec("weight_aggregation_train", {
            "weight_aggregation_train_time": _t() - t0,
        })

        t0 = _t()
        if agg_arrays is not None:
            result.arrays = agg_arrays
        _rec("global_model_update_train", {
            "global_model_update_train_time": _t() - t0,
        })

        if agg_metrics is not None:
            log(INFO, "\t\u2514\u2500\u2500> Train metrics: %s", agg_metrics)
            result.train_metrics_clientapp[current_round] = agg_metrics
            client_scalars = _scalar_metrics(agg_metrics)
            if client_scalars:
                _rec("client_train", client_scalars)

        return {
            "payload_sent": train_payload_sent,
            "payload_received": train_payload_received,
            "agg_arrays": agg_arrays,
            "client_computation": client_computation,
        }

    def _profile_evaluate_phase(
        self, grid, current_round, arrays, evaluate_config, timeout, _t, _rec,
        result,
    ) -> dict[str, Any]:
        p = self._profiler
        net_iface = getattr(p, "_network_interface", None)

        t0 = _t()
        eval_msgs = list(
            self.configure_evaluate(
                current_round, arrays, evaluate_config, grid,
            )
        )
        _rec("configure_evaluate", {"configure_evaluate_time": _t() - t0})

        sr = _measure_send_receive_with_network_counters(
            grid=grid,
            messages=eval_msgs,
            timeout=timeout,
            iface=net_iface,
        )
        _rec(
            "send_receive_evaluate",
            {
                "send_receive_evaluate_time": sr.elapsed,
                "updates_sent": len(eval_msgs),
                "updates_received": len(sr.replies),
            },
        )

        eval_payload_sent = sr.tx_delta if sr.observed else 0
        eval_payload_received = sr.rx_delta if sr.observed else 0
        eval_replies = sr.replies

        p._cumulative_bytes_sent += eval_payload_sent
        p._cumulative_bytes_received += eval_payload_received

        comm: dict[str, Any] = {
            "send_receive_evaluate_time": sr.elapsed,
            "evaluate_payload_bytes_sent": eval_payload_sent,
            "evaluate_payload_bytes_received": eval_payload_received,
            "updates_sent": len(eval_msgs),
            "updates_received": len(eval_replies),
            "cumulative_bytes_sent": p._cumulative_bytes_sent,
            "cumulative_bytes_received": p._cumulative_bytes_received,
        }

        client_times = _client_total_times(eval_replies)
        client_computation = max(client_times) if client_times else None
        if client_times:
            network_overhead = max(0.0, sr.elapsed - client_computation)
            comm["client_computation_evaluate"] = client_computation
            comm["network_overhead_evaluate"] = network_overhead
            comm["straggler_index_evaluate"] = client_computation - min(client_times)
            if client_computation > 0:
                comm["comm_to_compute_ratio_evaluate"] = (
                    network_overhead / client_computation
                )

        if sr.observed and sr.elapsed > 0:
            comm["bandwidth_utilization_bytes_per_sec_evaluate"] = (
                eval_payload_sent + eval_payload_received
            ) / sr.elapsed

        _rec("communication_evaluate", comm)

        t0 = _t()
        agg_eval_metrics = (
            self.aggregate_evaluate(current_round, eval_replies)
            if eval_replies else None
        )
        _rec("metric_aggregation_evaluate", {
            "metric_aggregation_evaluate_time": _t() - t0,
        })

        if agg_eval_metrics is not None:
            log(INFO, "\t\u2514\u2500\u2500> Eval metrics: %s", agg_eval_metrics)
            result.evaluate_metrics_clientapp[current_round] = agg_eval_metrics
            client_scalars = _scalar_metrics(agg_eval_metrics)
            if client_scalars:
                _rec("client_evaluate", client_scalars)

        return {
            "payload_sent": eval_payload_sent,
            "payload_received": eval_payload_received,
            "client_computation": client_computation,
        }

    def _record_session_totals(self, num_rounds: int, total_time: float) -> None:
        p = self._profiler
        p.record("total", {"total_time": total_time})
        p.record_global("num_rounds", num_rounds)
        p.record_global("cumulative_bytes_sent", p._cumulative_bytes_sent)
        p.record_global("cumulative_bytes_received", p._cumulative_bytes_received)

        cumulative_total = (
            p._cumulative_bytes_sent + p._cumulative_bytes_received
        )
        p.record_global("cumulative_bytes_total", cumulative_total)

        p.stop_carbon()
        p.rewrite_csv_from_metrics()


def profiled(
    strategy: Strategy, profiler: Profiler | None = None
) -> Strategy:
    if profiler is not None and not profiler._enabled:
        return strategy
    if profiler is None:
        profiler = Profiler(enable_timing=True)
    if not profiler._algorithm_name:
        profiler._algorithm_name = strategy.__class__.__name__
        profiler._metrics.record_global(
            "algorithm_name", profiler._algorithm_name,
        )
    return _ProfiledStrategy(strategy, profiler)


def _round_efficiency(
    round_total_time: float, *client_computations: float | None
) -> float | None:
    compute = 0.0
    for value in client_computations:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            compute += float(value)
    if round_total_time > 0 and compute > 0:
        return compute / round_total_time
    return None


def _client_total_times(replies: list[Message]) -> list[float]:
    times: list[float] = []
    for reply in replies:
        try:
            if reply.has_error():
                continue
            value = reply.content["metrics"]["client_total_time"]
        except (AttributeError, KeyError, TypeError, ValueError):
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            times.append(float(value))
    return times


def _scalar_metrics(metrics: Any) -> dict[str, Any]:
    if metrics is None:
        return {}
    try:
        items = metrics.items()
    except AttributeError:
        return {}
    out: dict[str, Any] = {}
    for key, value in items:
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            out[str(key)] = value
    return out


def _read_network_counters(iface: str | None = None) -> tuple[int, int] | None:
    try:
        with open("/proc/net/dev", "r", encoding="utf-8") as f:
            lines = f.readlines()[2:]
    except OSError:
        return None

    tx_total = 0
    rx_total = 0

    for line in lines:
        if ":" not in line:
            continue
        name, rest = line.split(":", 1)
        nic = name.strip()
        if iface is not None and nic != iface:
            continue
        if iface is None and nic == "lo":
            continue

        parts = rest.split()
        if len(parts) < 16:
            continue
        rx_total += int(parts[0])
        tx_total += int(parts[8])

    if iface is not None and tx_total == 0 and rx_total == 0:
        return None
    return tx_total, rx_total


def _measure_send_receive_with_network_counters(
    *,
    grid,
    messages: list[Message],
    timeout: float,
    iface: str | None,
) -> _SendReceiveResult:
    before = _read_network_counters(iface)
    t0 = time.perf_counter()
    replies = list(grid.send_and_receive(messages=messages, timeout=timeout))
    elapsed = time.perf_counter() - t0
    after = _read_network_counters(iface)

    if before is None or after is None:
        return _SendReceiveResult(replies, elapsed, 0, 0, False)

    return _SendReceiveResult(
        replies=replies,
        elapsed=elapsed,
        tx_delta=max(0, after[0] - before[0]),
        rx_delta=max(0, after[1] - before[1]),
        observed=True,
    )
