# Configuration

Configuration sections and keys retain the existing pipeline names: `project`, `grid`, `weights`, `spatial_pca`, `clustering`, `raster`, `postprocess`, `export`, and optional `experiments`.

Relative soil, yield, and output paths are resolved against the YAML file's directory. Validation checks input existence and columns, supported algorithms, `k` values, deterministic seed types, component counts, neighborhood size, area thresholds, and CRS syntax.

Set `raster.enabled: true` for kriging/raster analysis or `false` for the adaptive vector/grid-cell workflow. `raster.parallel_gridsearch` and `raster.n_jobs` apply to raster candidate evaluation. CLI flags can override those two values for a run without renaming configuration keys.

Set `spatial_pca.use_r_multispati: true` only when the R pathway is installed. The raster pathway uses `spatial_pca.engine: multispaeti`; when Python `multispaeti` is absent it emits a warning and preserves the historical standard-PCA fallback.
