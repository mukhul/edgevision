"""Reproducible timing of the detection backends.

``stats`` computes summary statistics, ``runner`` runs timed iterations of a
detector (or of the decode + detect + track pipeline), and ``report``
builds, writes, summarises and compares JSON result files. Importing this
package does not import PyTorch, ONNX Runtime or OpenCV.
"""

from edgevision.benchmarks.runner import BenchmarkError, BenchmarkPlan, RunResult, run_benchmark
from edgevision.benchmarks.stats import percentile, summarize, throughput

__all__ = ["BenchmarkError", "BenchmarkPlan", "RunResult", "percentile", "run_benchmark", "summarize", "throughput"]
