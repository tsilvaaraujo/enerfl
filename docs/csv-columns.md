# CSV columns and units

This page lists every column enerfl writes, with its unit and where its value comes from. It covers the CSV written by a Flower run wrapped with `profiled()`, the metrics a `ClientProfiler` adds to each client reply, and the columns produced by collectors when `Profiler.measure()` is used directly.

## Files written per run

| File | Content |
|---|---|
| `<name>.csv` (`stream_csv_path`) | One row per round. A round `0` row exists only when an `evaluate_fn` is given (initial evaluation). |
| `<name>.meta.json` | Run metadata: machine, versions, git commit, scheduler job, configuration and the final global metrics. See [Run metadata](../README.md#run-metadata). |

## Conventions

- **Times** are wall-clock seconds measured with `time.perf_counter()` on the process that records them (the server for run and round columns, each client for client columns).
- **Bytes** are bytes. **Energy** is joules (J). **Emissions** are kilograms of CO₂-equivalent (kg CO₂eq).
- **Phase suffix**: `_train` and `_evaluate` (or a `train_`/`evaluate_` prefix for payload bytes) name the phase a value belongs to.
- **Empty cell**: the value was not measured in that round (for example round `0` has no train phase) or its inputs were not available (for example no client reported `client_total_time`).
- **Global columns** hold one value for the whole run and are repeated in every row. Except for `run_id` and `algorithm_name`, they are only filled when the CSV is rewritten at the end of the run, so rows streamed during the run leave them empty.
- **Columns appear dynamically**: a column exists only if some round produced it, and new columns are appended to the header as they appear.

## Flower run (`profiled()`)

### Identification

| Column | Unit | Description |
|---|---|---|
| `algorithm` | — | Algorithm name (`algorithm_name` or the strategy class name). |
| `round` | — | Round number. `0` is the initial evaluation. |
| `run_id` | — | Run identifier, `YYYYMMDDTHHMMSSZ-<8 hex>` unless one was passed. Global. |
| `algorithm_name` | — | Same as `algorithm`. Global. |

### Run totals (global)

| Column | Unit | Description |
|---|---|---|
| `num_rounds` | rounds | Rounds actually run (fewer than requested when `should_stop` ends the run early). |
| `total_bytes_sent` | bytes | Bytes sent by the server host during all send/receive spans of the run. |
| `total_bytes_received` | bytes | Bytes received, same scope. |
| `total_bytes` | bytes | `total_bytes_sent + total_bytes_received`. |
| `total_energy_joules` | J | Server process energy over the whole run (CodeCarbon). Only with `enable_carbon=True`. |
| `total_emissions_kg` | kg CO₂eq | Server emissions over the whole run. Only with `enable_carbon=True`. |
| `total_carbon_duration` | s | Time the server carbon tracker ran. Only with `enable_carbon=True`. |
| `profiler_failures` | count | Profiling failures swallowed during the run. `0` means the data is complete. |
| `profiler_failure_sites` | — | `site=count;...` for each site that failed, e.g. `carbon.init=1`. Empty when none failed. |

### Per round

| Column | Unit | Description |
|---|---|---|
| `initial_evaluation_time` | s | `evaluate_fn` on the initial model. Round `0` only. |
| `server_evaluation_time` | s | `evaluate_fn` on the round's model. Only with an `evaluate_fn`. |
| `round_total_time` | s | Whole round on the server, from configure_train to the end of server evaluation. |
| `round_efficiency` | ratio | `(client_computation_train + client_computation_evaluate) / round_total_time`. Share of the round spent in client computation. |
| `updates_sent` | messages | Messages sent in the round, train + evaluate. |
| `updates_received` | messages | Replies received in the round, train + evaluate. |
| `cumulative_bytes_sent` | bytes | Bytes sent by the server host from the start of the run to the end of this round. |
| `cumulative_bytes_received` | bytes | Bytes received, same scope. |
| `cumulative_bytes_total` | bytes | `cumulative_bytes_sent + cumulative_bytes_received`. |
| `total_time` | s | Whole run on the server. Last round's row only. |

### Per phase

Every column below exists for both phases; the train name is shown, replace `train` with `evaluate` for the other one.

| Column | Unit | Description |
|---|---|---|
| `configure_train_time` | s | Strategy's `configure_train` / `configure_evaluate`. |
| `messages_configured` | messages | Messages built by `configure_train`. Train phase only. |
| `send_receive_train_time` | s | Full dispatch → collect span: network, client computation and straggler wait, not pure network. |
| `updates_sent_train` | messages | Messages sent to clients. |
| `updates_received_train` | messages | Replies received from clients. |
| `train_payload_bytes_sent` | bytes | Bytes sent by the server host during the send/receive span (`/proc/net/dev`). `0` when the counters are not observable. |
| `train_payload_bytes_received` | bytes | Bytes received, same scope. |
| `bandwidth_utilization_bytes_per_sec_train` | bytes/s | `(payload sent + received) / send_receive_train_time`. Only when the counters are observable. |
| `weight_aggregation_train_time` | s | Strategy's `aggregate_train`. Train phase only. |
| `global_model_update_train_time` | s | Replacing the global model with the aggregated one. Train phase only. |
| `metric_aggregation_evaluate_time` | s | Strategy's `aggregate_evaluate`. Evaluate phase only. |

### Client decomposition (per phase)

Derived from the `client_total_time` each client puts in its reply. Absent when no reply carries it.

| Column | Unit | Description |
|---|---|---|
| `client_computation_train` | s | Slowest client's `client_total_time`. |
| `network_overhead_train` | s | `max(0, send_receive_train_time - client_computation_train)`. |
| `straggler_index_train` | s | Slowest minus fastest client's `client_total_time`. |
| `comm_to_compute_ratio_train` | ratio | `network_overhead_train / client_computation_train`. |

### Client energy totals (per phase)

Read from every successful client reply. Absent when no reply carries the metric.

| Column | Unit | Description |
|---|---|---|
| `client_energy_joules_sum_train` | J | Energy of all clients in the phase: the federation's client energy. |
| `client_energy_joules_max_train` | J | Highest client energy in the phase. |
| `client_energy_joules_min_train` | J | Lowest client energy in the phase. |
| `client_emissions_kg_sum_train` | kg CO₂eq | Emissions of all clients in the phase. |
| `client_emissions_kg_max_train` | kg CO₂eq | Highest client emissions in the phase. |
| `client_emissions_kg_min_train` | kg CO₂eq | Lowest client emissions in the phase. |

### Aggregated client metrics (per phase)

Every numeric metric returned by the strategy's `aggregate_train` / `aggregate_evaluate`, suffixed with the phase: `loss_train`, `accuracy_evaluate`, `local_training_time_train`, `client_energy_joules_train`, and so on. Units are those of the client metric.

Flower's default aggregation is a **weighted average by `num-examples`**, so these are means per client, not totals. For energy and emissions totals use the [client energy totals](#client-energy-totals-per-phase).

## Client reply (`ClientProfiler.to_metric_record()`)

The metrics a client adds to its reply. The server turns them into the client decomposition, the client energy totals and the aggregated client metrics above.

| Metric | Unit | Description |
|---|---|---|
| `<stage>_time` | s | Each `stage()` block, e.g. `data_loading_time`, `local_training_time`. |
| `client_total_time` | s | Sum of the stage times. |
| `client_energy_joules` | J | Client energy since the `ClientProfiler` was created. Only with `enable_carbon=True`. |
| `client_emissions_kg` | kg CO₂eq | Client emissions over the same span. Only with `enable_carbon=True`. |
| `client_carbon_duration` | s | Length of that span. Only with `enable_carbon=True`. |

## Collectors (`Profiler.measure()`)

When a workload is instrumented with `with profiler.measure(phase, **context):`, each enabled collector writes columns named after the phase. A `profiled()` Flower run records its columns directly and does not go through the collectors, so the `enable_*` collector flags do not change the Flower columns above.

| Collector (flag) | Column | Unit | Description |
|---|---|---|---|
| timing (`enable_timing`) | `<phase>_time` | s | Duration of the `measure()` block. |
| messages (`enable_message_count`) | `<phase>_sent_count` | messages | `len(context["messages"])`. |
| | `<phase>_received_count` | messages | `len(context["replies"])`. |
| updates (`enable_update_count`) | `<phase>_updates_sent` | updates | `context["updates_sent"]`. |
| | `<phase>_updates_received` | updates | `context["updates_received"]`. |
| weights (`enable_weights_count`) | `<phase>_weights_sent` | weights | `context["weights_sent"]`. |
| payload (`enable_payload_size`) | `<phase>_payload_bytes_sent` | bytes | `context["payload_bytes_sent"]`. |
| | `<phase>_payload_bytes_received` | bytes | `context["payload_bytes_received"]`. |
| compression (`enable_compression`) | `<phase>_original_bytes` | bytes | `context["original_bytes"]`. |
| | `<phase>_compressed_bytes` | bytes | `context["compressed_bytes"]`. |
| | `<phase>_compression_ratio` | ratio | `original_bytes / compressed_bytes`. |
| | `<phase>_total_params` | parameters | `context["total_params"]`. |
| | `<phase>_nonzero_params` | parameters | `context["nonzero_params"]`. |
| | `<phase>_sparsity_ratio` | ratio | `1 - nonzero_params / total_params`. |
| | `<phase>_bits_per_param` | bits | `context["bits_per_param"]`. |

Collectors only report what is passed in `context`; they do not measure payloads or counts themselves (timing excepted).

## Known caveats

- **Payload bytes are host traffic.** They are the difference of the `/proc/net/dev` counters before and after the send/receive span, summed over all interfaces except `lo` unless `network_interface` is set. Any other traffic on the host in that span is included. Outside Linux the counters are not observable and the payload columns are `0`.
- **Energy covers the whole machine, not only FL work.** CodeCarbon measures the machine by default; when several clients share a node, their energy readings overlap.
- **Energy may be estimated.** When CodeCarbon cannot read hardware counters (for example RAPL is not readable), it falls back to a model based on the CPU's TDP and load. The values are then estimates, not measurements.
