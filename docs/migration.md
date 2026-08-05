# Migration from Ag-GPT/MZGPT

GeoFarmAI contains analysis code only. LLM agents, API/UI code, conversational visual resolution, background job planning, sandbox execution, and web lookup stay in MZGPT/Ag-GPT.

## Commands

Old:

```bash
python -m jobs.flow_mzd
python -m jobs.run_experiments --cfg configs/project.yaml
```

New:

```bash
geofarmai run --config configs/example.yaml
geofarmai experiment --config configs/example.yaml
```

## Import mapping

| Old import | New import |
|---|---|
| `core.cluster` | `geofarmai.clustering` |
| `core.crs` | `geofarmai.crs` |
| `core.evaluate` | `geofarmai.evaluation` |
| `core.export` | `geofarmai.export` |
| `core.grid` | `geofarmai.grid` |
| `core.ingest` | `geofarmai.ingest` |
| `core.logging_utils` | `geofarmai.logging_utils` |
| `core.multispati` | `geofarmai.decomposition` |
| `core.raster_pipeline` | `geofarmai.raster` |
| `core.reconcile` | `geofarmai.reconcile` |
| `core.spatial` | `geofarmai.spatial` |
| `jobs.flow_mzd.mzd_flow_two_csvs` | `geofarmai.run_pipeline` or `geofarmai.GeoFarmPipeline` |
| `jobs.run_experiments` | `geofarmai.experiments` |

## Compatibility notes

- YAML keys and scientific metric/output columns are retained.
- Relative paths now resolve from the YAML file rather than the shell directory.
- Prefect decorators are removed; stage order and synchronous return behavior are retained with ordinary Python orchestration.
- The application-oriented `visual_manifest.json`/workspace registry dependency is replaced by a compact `artifact_manifest.json`. Scientific raster, table, GeoPackage, and PDF outputs keep their existing names.
- Explicit R MULTISPATI selection now fails clearly if unavailable instead of silently switching to PCA. Standard PCA remains the configured non-R vector path.
- Research CSVs were not migrated. The example uses newly generated synthetic data with compatible columns.
