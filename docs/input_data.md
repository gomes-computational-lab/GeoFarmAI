# Input data

GeoFarmAI reads one predictor CSV and one yield CSV. Configure the exact column names under `project.soil` and `project.yield`; names are case-sensitive and are not silently renamed. Coordinate values must be numeric or coercible to numeric. Rows with invalid coordinates are dropped, exact duplicate geometries are removed, and missing IDs are replaced with a deterministic row index, preserving the source workflow.

The yield measurement is internally renamed to `yield` for the vector/grid-cell workflow. The configured original yield column remains unchanged in the YAML and raster outputs.

The files in `examples/data/` are synthetic and distributable. The source archive's field data were not copied because redistribution permission is unknown.
