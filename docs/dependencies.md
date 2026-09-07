# Dependency boundaries

`pyproject.toml` is the authoritative dependency declaration. The retained
`requirements.txt` mirrors only the canonical base installation, while
`requirements-dev.txt` contains developer tooling.

## Base scientific installation

```bash
python -m pip install .
```

This command installs from a source checkout. GeoFarmAI is not currently
claimed to be published on PyPI.

The base dependencies are NumPy, pandas, SciPy, GeoPandas, pyproj, Shapely,
scikit-learn, scikit-fuzzy, statsmodels, and PyYAML. Together they support the
canonical data model, spatial harmonization, PCA, all standard clustering
methods, internal and outcome validation, YAML configuration, GeoFarmModel,
GeoFarmResult, and GeoPackage/CSV result export.

No LLM, conversational, UI, or server package is a base dependency.

## Optional scientific features

| Extra | Purpose |
|---|---|
| `multispati` | Existing Python `multispaeti` engine |
| `spatial` | libpysal/esda spatial-weight and Moran utilities |
| `r` | rpy2 plus Python-side spatial weights for existing R MULTISPATI |
| `raster` | rasterio, PyKrige, matplotlib, and contextily raster/report workflows |

Install an optional extra from a source checkout with, for example,
`python -m pip install ".[raster]"`. Extras are not part of the base
installation.

Explicit optional-engine requests fail clearly when their dependencies are
missing. Canonical `GeoFarmModel` decomposition does not silently replace an
explicit MULTISPATI request with PCA. The historical raster pipeline retains
its already-documented automatic Python-MULTISPATI fallback and provenance.

## Development

```bash
python -m pip install -e ".[dev]"
```

The `dev` extra contains pytest and the Python build frontend. The transitional
application extra was removed with the old Prefect jobs and application
services. GeoFarmAI has no application dependency group.

The following former declarations were removed because there are no current
production imports: scikit-gstat, mapclassify, tifffile, Great Expectations,
Jinja2, and tabulate. Conversational/server packages were also removed; that
functionality belongs to the separate ag-gpt repository.

The built distribution intentionally contains only the canonical `geofarmai`
package and the temporarily retained scientific `core` compatibility package.
It excludes `services` and `jobs` packages.
