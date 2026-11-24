"""Host network traffic counters, read from ``/proc/net/dev`` (Linux only)."""


def read_network_counters(iface: str | None = None) -> tuple[int, int] | None:
    try:
        return _parse_network_counters(iface)
    except Exception:
        return None


def _parse_network_counters(iface: str | None) -> tuple[int, int] | None:
    with open("/proc/net/dev", "r", encoding="utf-8") as f:
        lines = f.readlines()[2:]

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
