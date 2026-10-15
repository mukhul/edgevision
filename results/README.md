# results/

Benchmark outputs, annotated videos, and plots. **Generated outputs are not committed.** Curated, verified results will be summarised in [`docs/experiment_log.md`](../docs/experiment_log.md) together with their configuration and hardware.

No results exist yet.

## Planned layout

Each run will write to its own uniquely named folder so earlier results are never silently overwritten:

```text
results/
└── <experiment_id>/
    ├── config.yaml        # exact configuration used
    ├── environment.json   # hardware, OS, library versions, execution providers, git commit
    ├── results.json       # per-run statistics and metadata
    ├── results.csv        # tabular view of the same results
    └── plots/             # figures generated from the saved results
```
