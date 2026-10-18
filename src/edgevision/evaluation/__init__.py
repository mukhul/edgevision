"""Evaluation and analysis.

* ``detection_metrics``: IoU, matching, precision / recall / F1 and COCO-style
  AP / mAP against ground truth, and agreement between two backends' outputs
  (which is not accuracy).
* ``annotations``: load ground truth in COCO JSON format.
* ``benchmark_plots``: validate benchmark result files and plot them
  (needs matplotlib, the ``plots`` extra).

Importing this package needs NumPy only.
"""
