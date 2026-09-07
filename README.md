# GeoFarmAI

GeoFarmAI is a reusable scientific Python library for geospatial agricultural
management-zone delineation. It accepts one or many explicitly role-labelled
data sources, harmonizes their spatial support, applies established
decomposition and clustering methods, evaluates every candidate solution, and
returns structured scientific results with provenance.

GeoFarmAI is the reusable geospatial analysis framework. The separate ag-gpt
repository is the conversational and web application that uses GeoFarmAI to
run analyses and interpret their outputs. The dependency direction is always
`ag-gpt -> GeoFarmAI`.

## Capabilities

- CSV, pandas DataFrame, and GeoPandas GeoDataFrame inputs
- one or many spatial data sources
- explicit predictor, outcome, coordinate, identifier, and metadata roles
- optional outcomes for fully unsupervised analysis
- CRS transformation, grid generation, and spatial harmonization
- IDW, nearest-neighbor, and buffer-mean reconciliation
- PCA and optional Python- or R-backed MULTISPATI
- KMeans, Gaussian mixture, fuzzy C-means, and agglomerative clustering
- silhouette and Calinski-Harabasz internal metrics
- generic outcome variance reduction and ANOVA validation
- explicit candidate-selection policies, including `selection=None`
- vector/raster scientific export and provenance

Outcome columns never enter the clustering predictor matrix. Variable names do
not imply scientific roles: `nitrate`, `yield`, or any other numeric variable
can be a predictor or outcome only when explicitly declared.

## Installation

From a source checkout:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Optional scientific features are installed separately:

```bash
python -m pip install -e ".[multispati]"  # Python MULTISPATI
python -m pip install -e ".[spatial]"     # libpysal/esda utilities
python -m pip install -e ".[raster]"      # raster, kriging, and PDF support
python -m pip install -e ".[r]"           # Python bridge for R MULTISPATI
```

The `r` extra does not install R or the required R packages. A compatible R
installation with `ade4`, `spdep`, and `adespatial` must be configured
separately. Explicit MULTISPATI requests fail clearly if the selected engine is
unavailable; canonical analysis does not silently substitute PCA.

## Five-minute Python workflow

```python
from geofarmai import FieldDataset, GeoFarmModel

data = FieldDataset.from_csv(
    "field.csv",
    coordinates=("longitude", "latitude"),
    crs="EPSG:4326",
    predictors=["EC", "moisture", "elevation"],
    outcome="yield",  # optional; any explicit numeric outcome is supported
)

model = GeoFarmModel(
    decomposition="pca",
    clustering=["kmeans", "gmm", "fcm"],
    k=range(2, 6),
    selection="silhouette",
    random_state=42,
)
result = model.fit(data)

print(result.summary())
table = result.to_dataframe()
artifacts = result.export("outputs/field_zones.gpkg")
```

Use `decomposition="none"`, `"pca"`, or `"multispati"`. Selection supports
`"silhouette"`, `"calinski_harabasz"`, `"variance_reduction"`, and `None`.
Variance-reduction selection requires an explicit outcome and, when several
outcomes exist, an explicit `selection_outcome`.

`delineate_zones(data, **model_options)` is a thin convenience wrapper around
`GeoFarmModel(...).fit(data)`. The deprecated `run_pipeline` name remains a
pure scientific wrapper for `FieldDataset` inputs; mapping-based application
orchestration has been removed.

## YAML workflow

The canonical YAML schema makes sources, coordinates, CRS, and variable roles
explicit:

```yaml
sources:
  - id: field
    path: ../data/example_field.csv
    coordinates: {x: x, y: y, crs: EPSG:32615}
    variables:
      EC: {role: predictor, domain: soil}
      moisture: {role: predictor, domain: soil}
      yield: {role: outcome, domain: crop}

analysis:
  harmonization: {strategy: auto, resolution: auto, method: idw}
  decomposition: {method: pca, n_components: 2}
  clustering:
    methods: [kmeans, gmm]
    k: {min: 2, max: 4}
  selection: {metric: variance_reduction, outcome: yield}
  random_state: 42
```

```python
from geofarmai import load_config, run_from_config

configuration = load_config("configs/example.yaml")
data, model = configuration.build()
result = model.fit(data)

# Equivalent one-call form
result = run_from_config("configs/example.yaml")
```

Relative paths resolve from the YAML file. `load_legacy_config()` and
`run_from_config(path, legacy=True)` retain translation for historical
project-style YAML without restoring the removed workflow jobs.

## Scientific CLI

```bash
geofarmai --help
geofarmai validate --config configs/example.yaml
geofarmai run --config configs/example.yaml
```

See [docs/scientific_cli.md](docs/scientific_cli.md) for the legacy-YAML flag
and Python equivalents.

## Results and outputs

`GeoFarmResult` exposes:

- selected zone labels, when a selection policy is used;
- every candidate solution and its deterministic identity;
- predictor and outcome identities;
- component scores and loadings when applicable;
- internal and outcome-validation metrics;
- CRS, geometry, harmonization, and decomposition provenance;
- configuration and exported artifact paths.

`result.export(...)` writes reusable GeoPackage and CSV products. The optional
raster module retains scientific kriging, raster clustering, GeoTIFF, preview,
and PDF generation without writing application workspace or visualization
manifests.

## Repository architecture

- `geofarmai/` — canonical data, configuration, model, result, and CLI APIs
- `core/` — temporarily retained scientific algorithms used by the canonical API
- `configs/` — portable examples and historical configuration fixtures
- `data/` — small example and test inputs
- `tests/` — deterministic scientific and package-boundary tests
- `docs/` — dependency, CLI, and migration documentation

Application services and Prefect workflow jobs are intentionally absent from
the source tree and wheel. See
[docs/ag_gpt_migration.md](docs/ag_gpt_migration.md) for the repository
boundary.

## Development

```bash
python -m pytest -q
python -m build
```

The default suite requires no network, R runtime, LLM provider, web server, or
application service. R-dependent tests are optional and separately marked.

## Limitations

- Vector reconciliation does not implement kriging; it fails explicitly.
  Ordinary kriging is available through the optional raster workflow.
- R-backed MULTISPATI depends on a compatible external native R environment.
- The top-level `core` namespace is retained temporarily to avoid moving or
  rewriting established scientific algorithms during repository separation.

## Citation and license

Citation metadata and final author information require human review before a
public release. Review the repository license status before redistribution.

Development status: pre-release (`0.1.0`).
