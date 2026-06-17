"""Lightweight timing and memory helpers for operational diagnostics."""

from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter
import tracemalloc


def start_memory_tracing() -> None:
    """Enable tracemalloc once for process-level Python memory snapshots."""
    if not tracemalloc.is_tracing():
        tracemalloc.start()


def current_memory_mb() -> float:
    """Return current traced Python memory usage in megabytes."""
    if not tracemalloc.is_tracing():
        return 0.0
    current, _peak = tracemalloc.get_traced_memory()
    return current / (1024 * 1024)


@contextmanager
def timed_stage() -> float:
    """Measure execution duration for one stage."""
    start = perf_counter()
    holder = [0.0]
    try:
        yield holder
    finally:
        holder[0] = perf_counter() - start
