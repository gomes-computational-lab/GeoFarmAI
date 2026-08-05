# Outputs

Raster runs write kriged GeoTIFFs and statistics, component or raw-feature GeoTIFFs, candidate statistics, two selected cluster rasters where available, management-zone polygons, leaderboard and selected-metric CSVs, previews, a PDF, `run_manifest.json`, and `artifact_manifest.json`.

Vector/grid-cell runs write a GeoPackage with `zones` and `samples` layers, a metrics CSV, leaderboard CSV, feature previews, a PDF, and an artifact manifest.

`GeoFarmResult` exposes output paths, metrics, the leaderboard, selected algorithm and `k`, labels when they are available in memory, the resolved configuration, and cache status. Cached raster runs cannot reconstruct the in-memory label vector and therefore return `best_labels=None`; the exported selected cluster raster remains available.
