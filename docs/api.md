# Python API

Public entry points are deliberately small:

- `geofarmai.run_pipeline(config, force=False, parallel_gridsearch=None, n_jobs=None)`
- `geofarmai.GeoFarmPipeline.from_yaml(path)` and `GeoFarmPipeline.run(...)`
- `geofarmai.GeoFarmResult`

Scientific functions remain importable from their focused modules, including `geofarmai.clustering`, `geofarmai.evaluation`, `geofarmai.decomposition`, `geofarmai.raster`, and the GIS support modules.
