import logging
import threading
from collections import Counter
from contextlib import contextmanager
from typing import Any, Iterator


logger = logging.getLogger(__name__)

_lock = threading.Lock()
_counts: Counter[str] = Counter()


def report_failure(site: str, exc: BaseException) -> None:
    with _lock:
        _counts[site] += 1
        first = _counts[site] == 1
    if first:
        logger.warning(
            "enerfl: profiling failure in %s (%s: %s); continuing without it. "
            "Further failures at this site are counted, not logged.",
            site,
            type(exc).__name__,
            exc,
        )


@contextmanager
def guard(site: str) -> Iterator[None]:
    try:
        yield
    except Exception as exc:  # noqa: BLE001
        report_failure(site, exc)


def failure_count() -> int:
    with _lock:
        return sum(_counts.values())


def failure_sites() -> dict[str, int]:
    with _lock:
        return dict(_counts)


def failure_summary() -> dict[str, Any]:
    sites = failure_sites()
    return {
        "profiler_failures": sum(sites.values()),
        "profiler_failure_sites": ";".join(
            f"{site}={n}" for site, n in sorted(sites.items())
        ),
    }


def reset_failures() -> None:
    with _lock:
        _counts.clear()
