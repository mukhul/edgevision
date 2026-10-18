"""Summary statistics for timing samples.

Pure Python (no NumPy) so the definitions are explicit and testable:

* **Percentiles** use linear interpolation between the two nearest ranks of
  the sorted samples (the "type 7" definition, which is also NumPy's default
  ``np.percentile(..., method="linear")``). p50 is the median.
* **Standard deviation** is the sample standard deviation (divides by
  ``n - 1``). It is ``None`` when there is only one sample, because spread
  cannot be estimated from a single value.

Empty input is an error: a statistic of nothing is never reported as 0.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

PERCENTILES = (50, 90, 95)


def percentile(samples: Sequence[float], q: float) -> float:
    """The ``q``-th percentile (0-100) of ``samples`` with linear interpolation."""
    if not samples:
        raise ValueError("cannot compute a percentile of no samples")
    if not 0 <= q <= 100:
        raise ValueError(f"percentile must be between 0 and 100, got {q}")
    ordered = sorted(samples)
    position = (len(ordered) - 1) * q / 100
    lower = math.floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def summarize(samples: Sequence[float]) -> dict[str, Any]:
    """Count, mean, median, p90, p95, min, max and sample standard deviation.

    Raises ValueError for an empty sequence or non-finite values (a broken
    timer must not silently produce a plausible-looking summary).
    """
    values = [float(v) for v in samples]
    if not values:
        raise ValueError("cannot summarise an empty list of samples")
    if not all(math.isfinite(v) for v in values):
        raise ValueError("timing samples must be finite numbers")
    n = len(values)
    mean = math.fsum(values) / n
    stdev = math.sqrt(math.fsum((v - mean) ** 2 for v in values) / (n - 1)) if n > 1 else None
    return {
        "count": n,
        "mean": mean,
        "median": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "min": min(values),
        "max": max(values),
        "stdev": stdev,
    }


def throughput(frames: int, seconds: float) -> float:
    """Frames per second over a measured wall-clock interval."""
    if frames < 1:
        raise ValueError("throughput needs at least one frame")
    if not seconds > 0 or not math.isfinite(seconds):
        raise ValueError(f"the measured interval must be a positive number of seconds, got {seconds!r}")
    return frames / seconds
