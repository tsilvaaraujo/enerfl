# EnerFL

A lightweight, pluggable profiling library for [Flower](https://flower.ai/) federated learning experiments. It captures per-round and per-stage metrics on both **server** and **client** sides with minimal intrusion on FL algorithm execution.

## Features

- **Server-side profiling** (`Profiler`) — timing, payload sizes, message/update counts, compression ratios, and carbon emissions per round.
- **Client-side profiling** (`ClientProfiler`) — per-stage timing (data loading, training, serialization, etc.) with context-manager API.
- **Strategy wrapper** (`profiled()`) — wraps any Flower `Strategy` to automatically capture all per-round metrics without modifying the strategy code.
- **Pluggable collectors** — enable/disable individual metric collectors (timing, payload, compression, messages, weights) via constructor flags.
- **Carbon tracking** — optional CO₂ and energy (Joules) tracking via [CodeCarbon](https://github.com/mlco2/codecarbon). A single tracker is started once per process and reused across all rounds/stages (per-stage values are read as deltas), so CodeCarbon is not re-initialized each round.
- **Cross-algorithm comparison** — `compare_runs()` and `compare_runs_to_csv()` for side-by-side metric comparison across different FL algorithms.
- **Export** — CSV and pandas DataFrame export for analysis.
- **Low overhead** — payload estimation uses protobuf `ByteSize()` with sample-1 optimization; all other operations are sub-millisecond.

## Requirements

- Python >= 3.10
- See [requirements.txt](requirements.txt) for pinned dependency versions.

## Installation

```bash
pip install -r requirements.txt
```

## Testing

The unit tests stub `flwr`/`codecarbon` when they are absent, so they run
off-cluster with only `pytest`:

```bash
uv run --with pytest pytest
```

Or, in an existing environment:

```bash
pip install -e ".[dev]"
pytest
```

## Project Structure

```
enerfl/
├── __init__.py       # Public API — re-exports all components
├── carbon.py         # CarbonTracker — CodeCarbon wrapper for CO₂/energy
├── client.py         # ClientProfiler — client-side per-stage profiling
├── collectors.py     # Pluggable MetricsCollector base class and built-ins
├── comparison.py     # compare_runs() / compare_runs_to_csv()
├── metrics.py        # ProfilerMetrics — per-round metric storage dataclass
├── server.py         # Profiler — server-side profiler with collector system
├── strategy.py       # profiled() — Strategy wrapper for automatic profiling
└── requirements.txt  # Pinned dependency versions
```

## Usage

### Server side

Wrap any Flower strategy with `profiled()` to capture all per-round metrics automatically:

```python
from enerfl import Profiler, profiled, compare_runs_to_csv
from flwr.serverapp.strategy import FedAvg

profiler = Profiler(
    enable_timing=True,
    enable_carbon=True,
    carbon_tracker_kwargs={"country_iso_code": "BRA"},
    algorithm_name="FedAvg",
)

base_strategy = FedAvg(fraction_train=1.0, fraction_evaluate=1.0)
strategy = profiled(base_strategy, profiler)

# Run FL as usual — profiler captures everything
result = strategy.start(grid=grid, initial_arrays=arrays, num_rounds=5)

# Output
profiler.summary(show_phases=True)
profiler.to_csv("profiler_results.csv")
```

### Client side

Use `ClientProfiler` with context managers to time each stage:

```python
from enerfl import ClientProfiler

cp = ClientProfiler(enable_carbon=True, algorithm_name="FedAvg", country_iso_code="BRA")

with cp.stage("data_loading"):
    X_train, y_train = load_data()

with cp.stage("local_training"):
    model.fit(X_train, y_train)

with cp.stage("model_serialization"):
    params = get_model_params(model)

# Embed timing in Flower MetricRecord for server-side decomposition
metrics = cp.to_metric_record()
cp.summary()
```

### Cross-algorithm comparison

```python
from enerfl import compare_runs_to_csv

# After running experiments with different profilers
compare_runs_to_csv(profiler_fedavg, profiler_fedprox, profiler_fedadam, path="comparison.csv")
```

### Profiler constructor options

| Parameter | Default | Description |
|---|---|---|
| `enable_timing` | `True` | Per-phase wall-clock timing |
| `enable_payload_size` | `True` | Protobuf wire-size estimation |
| `enable_update_count` | `True` | Messages sent/received counts |
| `enable_message_count` | `False` | Detailed message counting |
| `enable_weights_count` | `False` | Weight tensor counts |
| `enable_compression` | `False` | Compression/sparsity ratios |
| `enable_carbon` | `False` | CO₂ emissions via CodeCarbon |
| `algorithm_name` | `""` | Tag for CSV export and comparison |

## Metrics Collected

### Server-side (per round)

- **Timing**: `configure_train_time`, `send_receive_train_time`, `weight_aggregation_train_time`, `round_total_time`, and evaluate equivalents. `send_receive_train_time` is the full dispatch→collect span (network + client compute + straggler wait), not pure network.
- **Payload**: `serialize_train_payload_bytes_sent`, `deserialize_train_payload_bytes_received`, cumulative byte totals.
- **Client decomposition**: `client_computation_train`, `network_overhead_train`, and evaluate equivalents — reconstructed from the per-client `client_total_time` embedded in each reply.
- **Derived** (recorded only when the inputs are available): `straggler_index_train`/`straggler_index_evaluate` (slowest − fastest client), `comm_to_compute_ratio_train`/`_evaluate` (network overhead ÷ client compute), `bandwidth_utilization_bytes_per_sec_train`/`_evaluate` (payload bytes ÷ dispatch→collect span, only when NIC counters are observable), and `round_efficiency` (client compute ÷ round wall-clock).
- **Convergence**: accuracy, loss, and other numeric metrics from aggregation results.
- **Carbon** (optional): `total_emissions_kg`, `total_energy_joules`, `total_carbon_duration`.

### Client-side (per stage)

- Per-stage timing (`data_loading_time`, `local_training_time`, etc.)
- `client_total_time`
- Optional payload bytes per stage
- Optional carbon emissions

## Extending with Custom Collectors

Subclass `MetricsCollector` to add custom metric collection:

```python
from enerfl import MetricsCollector, Profiler

class MyCustomCollector(MetricsCollector):
    def name(self) -> str:
        return "custom"

    def on_start(self, phase, round_num, **context):
        return {"start_value": get_something()}

    def on_end(self, phase, round_num, start_state, **context):
        return {f"{phase}_custom_metric": compute_metric(start_state["start_value"])}

profiler = Profiler()
profiler.add_collector(MyCustomCollector())
```

## Performance

The profiler is designed for minimal intrusion:

| Operation | Overhead per round |
|---|---|
| Payload estimation (protobuf ByteSize, sample-1) | ~0.3ms (sklearn-scale) |
| All timing / dict / metric operations | ~0.1ms |
| **Total profiler overhead** | **< 0.5ms per round** |

Payload size estimation uses Flower's `message_to_proto()` + protobuf `ByteSize()` on a single sampled message, multiplied by the message count. This gives exact gRPC wire-size results with ~10x less overhead than serializing every message.
