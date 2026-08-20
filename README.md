 # EnerFL

A lightweight, pluggable profiler for federated learning and distributed machine learning workloads. It captures timing, communication, resource, energy, and carbon metrics through explicit instrumentation.

The core `Profiler` and `ClientProfiler` APIs are framework-agnostic and can instrument any Python workload through context managers. An optional [Flower](https://flower.ai/) strategy wrapper provides automatic per-round profiling for Flower applications.

## Features

- **Run and round profiling** (`Profiler`) — timing, payload sizes, message/update counts, compression ratios, and carbon emissions.
- **Stage profiling** (`ClientProfiler`) — per-stage timing (data loading, training, serialization, etc.) through a context-manager API.
- **Framework-agnostic instrumentation** — profile arbitrary Python phases without coupling the workload to a specific federated learning framework.
- **Flower integration** (`profiled()`) — optionally wrap a Flower `Strategy` to capture per-round metrics without modifying strategy code.
- **Pluggable collectors** — enable/disable individual metric collectors (timing, payload, compression, messages, weights) via constructor flags.
- **Carbon tracking** — optional CO₂ and energy (Joules) tracking via [CodeCarbon](https://github.com/mlco2/codecarbon). A single tracker is started once per process and reused across all rounds/stages (per-stage values are read as deltas), so CodeCarbon is not re-initialized each round.
- **Cross-algorithm comparison** — `compare_runs()` and `compare_runs_to_csv()` for side-by-side metric comparison across different FL algorithms.
- **Export** — CSV and pandas DataFrame export for analysis.

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
├── strategy.py       # Optional Flower Strategy integration
└── requirements.txt  # Pinned dependency versions
```

## Usage

### Generic workload profiling

Use `Profiler.measure()` to collect metrics around phases in any Python workload:

```python
from enerfl import Profiler

profiler = Profiler(
    enable_timing=True,
    enable_carbon=True,
    carbon_tracker_kwargs={"country_iso_code": "BRA"},
    algorithm_name="FedAvg",
)

profiler.set_round(1)

with profiler.measure("data_loading"):
    train_data = load_data()

with profiler.measure("local_training"):
    model.fit(train_data)

profiler.summary(show_phases=True)
profiler.to_csv("profiler_results.csv")
```

Use `ClientProfiler` when metrics need to be returned as a flat record:

```python
from enerfl import ClientProfiler

client_profiler = ClientProfiler(
    enable_carbon=True,
    algorithm_name="FedAvg",
    country_iso_code="BRA",
)

with client_profiler.stage("data_loading"):
    train_data = load_data()

with client_profiler.stage("local_training"):
    model.fit(train_data)

metrics = client_profiler.to_metric_record()
client_profiler.summary()
```

### Flower integration

Wrap a Flower strategy with `profiled()` to capture per-round metrics automatically:

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

result = strategy.start(grid=grid, initial_arrays=arrays, num_rounds=5)

profiler.summary(show_phases=True)
profiler.to_csv("profiler_results.csv")
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

### Run/round metrics

- **Timing**: `configure_train_time`, `send_receive_train_time`, `weight_aggregation_train_time`, `round_total_time`, and evaluate equivalents. `send_receive_train_time` is the full dispatch→collect span (network + client compute + straggler wait), not pure network.
- **Payload**: `serialize_train_payload_bytes_sent`, `deserialize_train_payload_bytes_received`, cumulative byte totals.
- **Client decomposition with Flower**: `client_computation_train`, `network_overhead_train`, and evaluate equivalents — reconstructed from the per-client `client_total_time` embedded in each reply.
- **Derived** (recorded only when the inputs are available): `straggler_index_train`/`straggler_index_evaluate` (slowest − fastest client), `comm_to_compute_ratio_train`/`_evaluate` (network overhead ÷ client compute), `bandwidth_utilization_bytes_per_sec_train`/`_evaluate` (payload bytes ÷ dispatch→collect span, only when NIC counters are observable), and `round_efficiency` (client compute ÷ round wall-clock).
- **Convergence**: accuracy, loss, and other numeric metrics from aggregation results.
- **Carbon** (optional): `total_emissions_kg`, `total_energy_joules`, `total_carbon_duration`.

### Stage metrics

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
