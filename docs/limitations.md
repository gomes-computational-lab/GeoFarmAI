# Limitations and preserved ambiguities

- The migrated algorithms have not yet been validated across multiple farms, crop systems, coordinate systems, or sampling densities.
- Cluster IDs are nominal and may be permuted across mathematically equivalent fits.
- Yield variance reduction fills undefined single-member zone variances with zero, preserving existing behavior; this can favor small zones and requires scientific review.
- The vector zone exporter buffers grid centroids by one metre before dissolving and filters by `min_area_m2`. This can produce small or empty polygons for large thresholds; behavior is preserved.
- `grid.target_points_per_cell`, `grid.kriging_variogram`, and `grid.kriging_nlags` are retained configuration keys but are not consumed by the current vector implementation.
- Raster automatic cell size remains `min(width, height) / 250`; no lower bound is imposed there.
- Python `multispaeti` absence preserves the historical PCA fallback, now with a warning. Explicit R MULTISPATI failure raises an error to prevent an unreported method substitution.
- Web basemaps are best-effort and skipped when unavailable. Reproducible publication figures should use controlled local basemap sources.
- The source QGIS style resource is empty and is packaged unchanged pending a reviewed style definition.
