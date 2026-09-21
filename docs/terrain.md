# Terrain processing

GeoFarmAI can construct a metric elevation surface from point observations and
derive two terrain predictors without treating unordered point rows as a
surface:

```python
from geofarmai import terrain_from_elevation_points

terrain = terrain_from_elevation_points(
    "elevation_points.csv",
    x_column="longitude",
    y_column="latitude",
    elevation_column="elevation_ft",
    source_crs="EPSG:4326",
    elevation_units="feet",
    target_crs="EPSG:32612",
    cell_size_m=5.0,
    output_directory="outputs/terrain",
    interpolation_config={
        "variogram_model": "spherical",
        "nlags": 12,
        "n_closest_points": 24,
    },
)

print(terrain.predictor_rasters)
print(terrain.provenance)
```

The raster resolution is required explicitly. It should be selected from the
spatial sampling density and measurement accuracy, not merely from the field
extent.

## Scientific definitions

Elevation is converted to metres before interpolation and differentiation.
Horizontal coordinates must use a projected CRS with metre units. Ordinary
kriging delegates to GeoFarmAI's existing raster implementation.

Slope uses the Horn 3x3 finite-difference operator. If `z1` through `z9` are
the row-major cells in a 3x3 window, then:

```text
dz/dx = ((z3 + 2 z6 + z9) - (z1 + 2 z4 + z7)) / (8 dx)
dz/dy = ((z7 + 2 z8 + z9) - (z1 + 2 z2 + z3)) / (8 dy)
slope = degrees(atan(sqrt((dz/dx)^2 + (dz/dy)^2)))
```

Slope is reported in degrees.

General curvature is the elevation Laplacian:

```text
d2z/dx2 = (east  - 2 center + west)  / dx^2
d2z/dy2 = (south - 2 center + north) / dy^2
curvature = d2z/dx2 + d2z/dy2
```

For elevation and grid spacing in metres, curvature units are inverse metres.
Positive values denote a locally concave-up surface; negative values denote a
locally concave-down surface. This quantity is **general curvature (elevation
Laplacian)**, not profile curvature or plan curvature.

The outer raster cell is nodata because a complete finite-difference
neighborhood is unavailable. Interior slope is nodata if any cell in the Horn
3x3 neighborhood is nodata. Interior curvature is nodata if its centre,
north, south, east, or west value is nodata. GeoFarmAI does not fill these
derivatives silently.
