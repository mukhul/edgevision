"""Tests for edgevision.benchmarks.stats (pure Python, deterministic)."""

from __future__ import annotations

import math
import random
import statistics

import pytest

from edgevision.benchmarks.stats import percentile, summarize, throughput


@pytest.mark.parametrize(
    ("samples", "q", "expected"),
    [
        ([1, 2, 3, 4], 50, 2.5),
        ([1, 2, 3, 4], 90, 3.7),  # rank 0.9 * 3 = 2.7 -> 3 + 0.7 * (4 - 3)
        ([1, 2, 3, 4], 0, 1),
        ([1, 2, 3, 4], 100, 4),
        ([4, 1, 3, 2], 95, 3.85),  # order of the input does not matter
        ([7.5], 95, 7.5),  # a single sample is every percentile
        ([10, 10, 10], 90, 10),
    ],
)
def test_percentile_linear_interpolation(samples, q, expected) -> None:
    assert percentile(samples, q) == pytest.approx(expected)


def test_percentile_matches_numpy_default() -> None:
    np = pytest.importorskip("numpy")
    rng = random.Random(7)
    for n in (2, 3, 10, 101, 300):
        samples = [rng.lognormvariate(3, 0.4) for _ in range(n)]
        for q in (50, 90, 95, 99):
            assert percentile(samples, q) == pytest.approx(float(np.percentile(samples, q)), rel=1e-12)


@pytest.mark.parametrize("q", [-1, 101])
def test_percentile_out_of_range(q) -> None:
    with pytest.raises(ValueError, match="between 0 and 100"):
        percentile([1, 2], q)


def test_percentile_of_nothing_is_an_error() -> None:
    with pytest.raises(ValueError, match="no samples"):
        percentile([], 50)


def test_summarize() -> None:
    samples = [12.0, 10.0, 11.0, 30.0, 10.5]
    s = summarize(samples)
    assert s["count"] == 5
    assert s["mean"] == pytest.approx(14.7)
    assert s["median"] == 11.0
    assert s["min"] == 10.0 and s["max"] == 30.0
    assert s["p90"] == pytest.approx(percentile(samples, 90)) and s["p95"] == pytest.approx(percentile(samples, 95))
    assert s["stdev"] == pytest.approx(statistics.stdev(samples))  # sample standard deviation (n - 1)


def test_single_sample_has_no_standard_deviation() -> None:
    s = summarize([5.0])
    assert s["stdev"] is None  # unavailable, not 0
    assert s["mean"] == s["median"] == s["p95"] == 5.0


def test_summarize_keeps_full_precision() -> None:
    value = 1 / 3
    assert summarize([value, value])["mean"] == value


@pytest.mark.parametrize("bad", [[], [1.0, math.nan], [math.inf]])
def test_summarize_rejects_empty_or_non_finite(bad) -> None:
    with pytest.raises(ValueError):
        summarize(bad)


def test_throughput() -> None:
    assert throughput(300, 10.0) == 30.0
    with pytest.raises(ValueError, match="at least one frame"):
        throughput(0, 1.0)
    for seconds in (0, -1, math.inf, math.nan):
        with pytest.raises(ValueError, match="positive number of seconds"):
            throughput(10, seconds)
