# GeoFarmAI

GeoFarmAI is a scientific Python library for delineating candidate agricultural
management zones from one or more heterogeneous geospatial datasets. It keeps
variable roles explicit, harmonizes point observations onto a compatible
spatial support, applies established feature-representation and clustering
methods, and returns candidate partitions with validation metrics and
provenance.

## Overview

Management-zone delineation partitions a field into spatial units that are
more internally similar than the field as a whole. GeoFarmAI provides the
analytical workflow for constructing and evaluating those partitions. It can
combine sources such as soil measurements, terrain attributes, crop
observations, or nutrient samples even when they were collected at different
locations.

GeoFarmAI produces analytical zone delineations. A selected partition is not,
by itself, proof that the zones are agronomically correct or suitable for an
operational decision; interpretation and field validation remain necessary.

## Key Capabilities

- CSV, pandas `DataFrame`, and point GeoPandas `GeoDataFrame` inputs
- one or many explicitly identified data sources
- predictor, outcome, coordinate, identifier, and metadata variable roles
- projected/geographic CRS handling and deterministic target-CRS selection
- direct spatial alignment or grid-based reconciliation of heterogeneous data
- IDW, nearest-neighbor, and buffer-mean vector reconciliation
- standardized predictors without reduction, PCA, and optional MULTISPATI
- KMeans, Gaussian mixture, fuzzy C-means, and connected agglomerative clustering
- fixed K or multiple candidate K values
- silhouette and Calinski–Harabasz internal validation
- optional outcome variance-reduction and ANOVA validation
- explicit model-selection policy or retention of all candidates without a winner
- structured results, provenance, GeoPackage output, and CSV metrics tables
- optional metric DEM interpolation, Horn slope, and elevation-Laplacian
  general curvature rasters

## Scientific Workflow

```text
agricultural point observations
              ↓
explicit source, coordinate, and variable roles
              ↓
spatial harmonization
              ↓
predictor matrix (outcomes excluded)
              ↓
optional dimensionality reduction
              ↓
candidate management-zone clustering
              ↓
internal validation
              ↓
optional outcome-based validation
              ↓
GeoFarmResult and scientific exports
```

The same implementation is used by the Python API and canonical YAML
configuration interface.

## Installation

GeoFarmAI currently supports Python 3.10 or later. From a source checkout,
create an isolated environment and install the base scientific package:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install .
```

For development and tests:

```bash
python -m pip install -e ".[dev]"
```

The following optional extras are defined in `pyproject.toml`:

```bash
python -m pip install ".[multispati]"  # Python multispaeti engine
python -m pip install ".[spatial]"     # libpysal/esda utilities
python -m pip install ".[raster]"      # raster, kriging, map, and PDF support
python -m pip install ".[r]"           # Python dependencies for R MULTISPATI
```

The `r` extra does not install R itself or the R packages `ade4`, `spdep`, and
`adespatial`. Those must be installed and configured separately. The base
installation does not require an R runtime, a MULTISPATI engine, a raster
stack, or an application/web stack.

GeoFarmAI is not claimed to be available from PyPI at this development stage.

## Quick Start

This example uses the tested synthetic data distributed in
[`examples/data/single_csv_yield.csv`](examples/data/single_csv_yield.csv):

```python
from geofarmai import FieldDataset, GeoFarmModel

data = FieldDataset.from_csv(
    "examples/data/single_csv_yield.csv",
    source_id="field_observations",
    coordinates=("x", "y"),
    crs="EPSG:32615",
    predictors=["EC", "moisture", "elevation"],
    outcome="yield",
)

model = GeoFarmModel(
    decomposition="pca",
    n_components=2,
    clustering="kmeans",
    k=range(2, 4),
    selection="variance_reduction",
    selection_outcome="yield",
    random_state=42,
)

result = model.fit(data)
print(result.summary())
artifacts = result.export("outputs/field_zones.gpkg")
print(artifacts)
```

`yield` is explicitly declared as an outcome here. It is used to validate and
select candidate partitions, but it is never included in the predictor matrix
or PCA inputs.

## Data Model

A `DataSource` represents one input table and includes:

- a stable source identifier;
- a CSV path, pandas `DataFrame`, or point `GeoDataFrame`;
- an explicit `CoordinateSpec` or x/y columns and CRS;
- one or more `VariableSpec` declarations;
- optional descriptive source metadata.

A `FieldDataset` contains one or many `DataSource` objects. For a single table,
`FieldDataset.from_csv`, `from_dataframe`, and `from_geodataframe` provide
concise constructors. Multiple inputs are assembled with
`FieldDataset.from_sources`.

Each variable has one explicit role:

- `predictor`: enters the feature matrix used to delineate zones;
- `outcome`: evaluates candidate zones and can drive selection;
- `coordinate`: locates observations;
- `identifier`: identifies records without becoming a feature;
- `metadata`: retained descriptive information that is not a feature.

Source and variable identities are retained together. Thus, variables named
`moisture` from two different sources remain distinct rather than silently
overwriting one another.

GeoFarmAI also provides non-LLM schema inspection and coordinate-name candidate
detection. These utilities do not assign scientific predictor or outcome roles.

## Predictors and Outcomes

Predictors are the variables used to construct management zones. Outcomes are
external measurements used to evaluate or select zone solutions when desired.

Outcomes are optional. Yield is one possible outcome, not a required input.
Nitrate, biomass, protein, profitability, or another numeric measurement can
also be an outcome when explicitly declared. A variable named `nitrate` can
instead be a predictor in a different analysis if the user explicitly assigns
that role.

GeoFarmAI does not infer a scientific role from names such as `yield`,
`nitrate`, or `EC`. Outcome variables are kept out of harmonized predictors and
decomposition inputs.

## Spatial Harmonization

Current harmonization operates on point observations. Geometry is authoritative
for a point `GeoDataFrame`; otherwise numeric x/y columns are converted to point
geometry.

The `auto` strategy behaves deterministically:

- one scientific source is reused directly;
- colocated sources are aligned directly within a configurable metric tolerance;
- when predictor locations match but outcome locations differ, predictors stay
  on their direct support and outcomes are reconciled to it;
- when predictor sources differ spatially, variables are reconciled to a common
  analysis grid.

Users can also request `strategy="direct"` or `strategy="grid"`. Direct
alignment fails if locations do not correspond within the configured tolerance.
Supported vector reconciliation methods are `idw`, `nearest`, and
`buffer_mean`.

Distance operations use a projected CRS with metre units. A shared metric CRS
is preserved where possible; otherwise GeoFarmAI uses its existing automatic
UTM selection, or an explicitly supplied metric target CRS. Original and target
CRS, actual strategy, reconciliation method, grid resolution, coverage, and
per-variable construction details are retained in provenance.

GeoFarmAI does not introduce general mean imputation, zero filling, or silent
row dropping. Missing/non-finite predictors remaining after harmonization must
be resolved by the user before model fitting.

## Feature Representation

`GeoFarmModel` supports:

- `decomposition="none"`: standardize predictors without reducing dimensions;
- `decomposition="pca"`: standardized principal component analysis;
- `decomposition="multispati"`: MULTISPATI using an explicitly selected
  `multispati_engine="python"` or `"r"`.

Python MULTISPATI requires the `multispati` extra. R MULTISPATI requires the
`r` extra plus a compatible external R installation and its spatial packages.
When MULTISPATI is explicitly requested and unavailable, GeoFarmAI raises a
clear error; the canonical model does not silently substitute PCA.

## Management-Zone Clustering

The `clustering` argument accepts one method or several:

- `"kmeans"` — KMeans;
- `"gmm"` — Gaussian mixture model;
- `"fcm"` — fuzzy C-means, returned as the current hard zone assignment;
- `"agglomerative"` — agglomerative clustering with spatial connectivity.

`k` may be a single integer or an iterable such as `range(2, 6)`. Every
algorithm/K combination is evaluated and retained as a `CandidateSolution`.
KMeans, GMM, and FCM use the model's explicit `random_state`; agglomerative
clustering has no stochastic seed in the current implementation.

## Model Selection and Validation

### Internal validation

Every candidate records:

- average silhouette (`asc`);
- Calinski–Harabasz score (`ch_score`).

Either may select a candidate through `selection="silhouette"` or
`selection="calinski_harabasz"`. Both selection policies maximize the requested
metric.

### Outcome validation

For every explicitly declared outcome with sufficient finite observations,
each candidate records:

- variance reduction;
- ANOVA p-value;
- observation count;
- coverage fraction.

`selection="variance_reduction"` maximizes variance reduction for the selected
outcome. With multiple outcomes, `selection_outcome` must identify the outcome
used for selection.

Outcome metrics are not required for clustering. With no outcomes, internal
metrics and candidate partitions remain available. With `selection=None`, all
candidates are retained but `selected_solution`, selected `zone_labels`, and
top-level selected metrics remain empty rather than choosing a winner
implicitly.

## Four Tested End-to-End Examples

All example CSV files are small, deterministic, synthetic datasets. Each
Python example has an equivalent canonical YAML configuration and is exercised
by the release acceptance suite.

1. **One CSV with yield** — EC, moisture, and elevation are predictors; yield
   is an outcome used for variance-reduction validation and selection.
   [Python](examples/single_csv_yield.py) ·
   [YAML](examples/configs/single_csv_yield.yaml)

2. **Heterogeneous soil, terrain, and yield sources** — sampling locations
   differ, so GeoFarmAI uses a reconciliation grid while preserving source and
   predictor provenance. Yield remains outcome-only.
   [Python](examples/heterogeneous_yield.py) ·
   [YAML](examples/configs/heterogeneous_yield.yaml)

3. **Nitrate as an outcome** — nitrate is an outcome because the configuration
   explicitly says so, not because GeoFarmAI interprets its name. It is
   reconciled to the soil support and excluded from PCA and clustering inputs.
   [Python](examples/nitrate_outcome.py) ·
   [YAML](examples/configs/nitrate_outcome.yaml)

4. **Predictors only** — spatially different soil and terrain predictors are
   harmonized with no outcome. Internal metrics are produced, outcome metrics
   are absent, and `selection=None` retains both K candidates without choosing
   a winner.
   [Python](examples/predictors_only.py) ·
   [YAML](examples/configs/predictors_only.yaml)

See [the acceptance-example guide](examples/README.md) for expected result and
export structures.

## Canonical YAML Configuration

The following is the tested single-CSV configuration. Because it lives under
`examples/configs/`, its input path is relative to that directory:

```yaml
sources:
  - id: field_observations
    path: ../data/single_csv_yield.csv
    coordinates: {x: x, y: y, crs: EPSG:32615}
    variables:
      EC: {role: predictor, domain: soil, units: dS/m}
      moisture: {role: predictor, domain: soil, units: percent}
      elevation: {role: predictor, domain: terrain, units: m}
      yield: {role: outcome, domain: crop, units: Mg/ha}

analysis:
  harmonization: {strategy: auto, resolution: auto, method: idw}
  decomposition: {method: pca, n_components: 2}
  clustering:
    methods: [kmeans]
    k: {min: 2, max: 3}
  selection: {metric: variance_reduction, outcome: yield}
  random_state: 42
```

Load and run canonical YAML through the public API:

```python
from geofarmai import load_config, run_from_config

configuration = load_config("examples/configs/single_csv_yield.yaml")
dataset, model = configuration.build()
result = model.fit(dataset)

# Equivalent one-call execution
result = run_from_config("examples/configs/single_csv_yield.yaml")
```

Relative source paths resolve from the YAML file, not the current working
directory. Historical project-style YAML can still be translated with
`load_legacy_config` or `run_from_config(path, legacy=True)`, but new analyses
should use the canonical source-based schema.

## Command-Line Interface

The installed scientific CLI exposes only validation and execution:

```bash
geofarmai --help
geofarmai validate --config examples/configs/single_csv_yield.yaml
geofarmai run --config examples/configs/single_csv_yield.yaml
```

`validate` parses the configuration and constructs its canonical dataset and
model. `run` executes the same `GeoFarmModel` workflow used by the Python API.
See [the CLI guide](docs/scientific_cli.md) for isolated legacy-configuration
support.

## Results and Outputs

`GeoFarmModel.fit` returns a `GeoFarmResult` containing:

- `candidate_solutions`: every algorithm/K candidate and its labels and metrics;
- `selected_solution` and `zone_labels`: the selected candidate, when selection
  is enabled;
- `predictor_names` and `outcome_names`;
- `component_scores` and `component_loadings`, when applicable;
- `internal_metrics` and `outcome_validation_metrics` for the selected solution;
- `spatial_metadata`, `harmonization_provenance`, and
  `decomposition_provenance`;
- the effective model `configuration` and exported `artifacts`.

Useful result interfaces are `result.summary()`, `result.leaderboard`,
`result.to_dataframe()`, and `result.export("outputs/field_zones.gpkg")`.

The base export writes:

- a GeoPackage with aligned samples and, when a solution is selected, dissolved
  zones;
- a candidate leaderboard CSV;
- a selected metrics/summary CSV.

For `selection=None`, `export()` writes the unlabelled analysis support and full
candidate table. Pass a retained candidate or its `solution_id` through the
`solution` argument to export that candidate's zones without changing the
result's selection state.

Raster, GeoTIFF, preview-map, and PDF functionality belongs to the optional
`raster` workflow and requires the `raster` extra.

## Reproducibility

GeoFarmAI records the choices needed to understand how an analysis was built:

- explicit source, coordinate, predictor, and outcome declarations;
- original and target CRS;
- requested and actual harmonization strategy;
- reconciliation method, parameters, coverage, and grid resolution;
- requested and actual decomposition method;
- clustering method, K, and applicable random seed;
- selection metric and outcome;
- every retained candidate and its metrics.

The tested YAML configurations and synthetic acceptance datasets provide
reproducible examples. Fixed seeds make supported stochastic methods
deterministic within the tested environment, but exact floating-point values
can still vary across operating systems, numerical libraries, or dependency
versions.

## Limitations

- Canonical vector harmonization currently requires point observations; it does
  not provide general polygon or arbitrary-geometry reconciliation.
- Vector reconciliation does not support kriging. Ordinary kriging is retained
  in the optional raster workflow.
- Python MULTISPATI depends on a compatible `multispaeti` and scikit-learn
  combination.
- R MULTISPATI requires a compatible external R/rpy2 environment and the
  required R spatial packages.
- Raster and additional spatial-weight functionality require optional extras.
- GeoFarmAI does not silently impute incomplete predictors. Outcome metrics
  reflect the finite observations and spatial coverage supplied by the user.
- Outcome-based validation depends on the outcome's measurement quality and
  spatial representativeness.
- Candidate zones require agronomic interpretation and field validation before
  operational use.

## External Applications

External applications can call GeoFarmAI through its public scientific API.
The separate `ag-gpt` repository is one such application consumer; the
dependency direction is `ag-gpt → GeoFarmAI`. Application and conversational
functionality are not GeoFarmAI runtime requirements.

## Research and Software Provenance

GeoFarmAI developed from an experimental scientific pipeline into a
generalized, reusable scientific and geospatial analysis library:

```text
experimental scientific pipeline
    -> generalized GeoFarmAI library
```

Results used in the historical poster or manuscript were produced with the
earlier experimental pipeline. This repository does not claim that GeoFarmAI
v0.1.0 generated those results.

## Development Status, Citation, and License

GeoFarmAI is currently a pre-release library at version `0.1.0`. The default
test suite requires no network, R runtime, or external service:

```bash
python -m pytest -q
python -m build
```

Researchers using GeoFarmAI should cite the software using the metadata in
[`CITATION.cff`](CITATION.cff). Software attribution is distinct from
manuscript authorship and funding. An associated research publication is in
preparation; no manuscript DOI or publication citation is currently claimed.

ORCID identifiers, remaining grant metadata, and the eventual manuscript
citation still require human confirmation.

```yaml
Proposed license: BSD-3-Clause
Status: pending confirmation of licensing/IP requirements.
```

No license has yet been applied or granted, so do not redistribute the package
until its license status has been resolved.
