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

- Python >= 3.11
- Dependencies and their supported version ranges are declared in [pyproject.toml](pyproject.toml). Flower is optional and only needed for `profiled()` (extra `flower`).

## Installation

```bash
pip install .
```

With the Flower integration (`profiled()`):

```bash
pip install ".[flower]"
```

With `to_dataframe()` support:

```bash
pip install ".[dataframe]"
```

## Testing

The tests run against the real `flwr` and `codecarbon`:

```bash
pip install -e ".[dev]"
pytest
```

## Project Structure

```
src/enerfl/
├── __init__.py       # Public API — re-exports all components
├── _safe.py          # guard()/failure counters — profiling-failure isolation
├── carbon.py         # CarbonTracker — CodeCarbon wrapper for CO₂/energy
├── client.py         # ClientProfiler — client-side per-stage profiling
├── collectors.py     # Pluggable MetricsCollector base class and built-ins
├── comparison.py     # compare_runs() / compare_runs_to_csv()
├── csv_writer.py     # MetricsCSVWriter — streaming CSV output
├── metrics.py        # ProfilerMetrics — per-round metric storage dataclass
├── metadata.py       # run_id and the JSON run-metadata file
├── network.py        # Host NIC byte counters (/proc/net/dev)
├── scheduler.py      # SLURM/OAR job and node-list detection
├── server.py         # Profiler — server-side profiler with collector system
└── strategy.py       # Optional Flower Strategy integration
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

Requires the `flower` extra (`pip install ".[flower]"`). Wrap a Flower strategy with `profiled()` to capture per-round metrics automatically:

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
| `run_id` | generated | Identifier of the run (see [Run metadata](#run-metadata)) |
| `metadata_path` | next to the CSV | Where to write the run's JSON metadata |

### Run metadata

Every `Profiler` has a `run_id` (`YYYYMMDDTHHMMSSZ-<8 hex>` unless you pass one), recorded in every CSV row. When `stream_csv_path` or `metadata_path` is set, a JSON file (`run.csv` -> `run.meta.json` by default) is written when the profiler is created, so an interrupted run still leaves its metadata, and rewritten at the end of the run with `finished_at_utc` and the final global metrics. It records the machine (hostname, OS, CPU, memory), Python and package versions (enerfl, flwr, codecarbon, protobuf), the git commit and dirty flag of the directory the process runs from, the batch-job context (`scheduler`: SLURM or OAR job id, name, partition/account/cluster, array ids, node count and the expanded node list; `null` outside a job) and the profiler configuration. The scheduler section holds only job and node identifiers, never user names or directories; SLURM node lists are expanded in Python (`node[01-03]` becomes three hosts) and, if one cannot be parsed, only the raw string is kept. Outside a Flower `profiled()` run, call `profiler.write_metadata(finished=True)` yourself. Metadata failures never abort a run.

## Metrics Collected

Every column, with its unit and where it comes from, is documented in [docs/csv-columns.md](docs/csv-columns.md). In short:

- **Timing** per round and phase: `configure_train_time`, `send_receive_train_time`, `weight_aggregation_train_time`, `round_total_time`, `total_time`, and evaluate equivalents.
- **Communication**: messages sent and received per phase, host payload bytes (`train_payload_bytes_sent`, ...) and run totals (`cumulative_bytes_*`).
- **Client decomposition**: `client_computation_train`, `network_overhead_train`, `straggler_index_train`, `comm_to_compute_ratio_train`, `round_efficiency`.
- **Client metrics**: every numeric metric aggregated from client replies, suffixed with its phase (`loss_train`, `accuracy_evaluate`). Flower aggregates them as weighted means.
- **Client energy totals**: `client_energy_joules_sum_train` and `_max`/`_min`, the same for `client_emissions_kg`, read from each reply.
- **Server carbon** (optional): `total_energy_joules`, `total_emissions_kg`, `total_carbon_duration`.
- **Data quality**: `profiler_failures`, `profiler_failure_sites`.

## Failure isolation

A failure in profiling code never aborts an FL run. Collectors, the carbon
tracker (init, start, stop, read), the CSV writer, derived metrics and the
`/proc/net/dev` reader are all isolated: the error is logged once per site as a
warning, counted, and the run continues without that metric.

Errors from the experiment itself (`configure_*`, `send_and_receive`,
`aggregate_*`, `evaluate_fn`) are **not** swallowed, and neither are
`KeyboardInterrupt`/`SystemExit`, so job cancellation still works.

Two global columns make degraded runs detectable in the CSV:

- `profiler_failures` — total swallowed failures (`0` means complete data);
- `profiler_failure_sites` — `site=count;...`, e.g. `carbon.init=1`.

Deliberately not degraded: a CSV path without a `.csv` suffix raises at
construction (a deterministic config error, caught before cluster time is
spent). I/O errors on the CSV (unwritable directory, full disk) do degrade.

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
