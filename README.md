# GeoFarmAI

**A Geospatial AI Framework for Farmer-Centered Sustainable Agricultural Zone Management**

GeoFarmAI is a reusable Python framework for transforming soil and yield observations into candidate agricultural management zones. It packages the existing raster-first and vector/grid-cell scientific workflows, spatial dimensionality reduction, four clustering methods, agronomic validation, experiments, and GIS/report exports behind stable Python and command-line interfaces.

Development status: **0.1.0 alpha**. Scientific review, broader field validation, and release metadata decisions remain necessary before publication.

## Core capabilities

- CSV ingestion, CRS validation, automatic local UTM selection, adaptive grids, and IDW/nearest/buffer reconciliation
- Raster-first ordinary kriging and aligned raster stacks
- PCA, Python MULTISPATI-PCA when available, and optional R MULTISPATI for the vector workflow
- K-means, agglomerative clustering, Gaussian mixture models, and fuzzy C-means
- Silhouette, Calinski–Harabasz, yield variance reduction, and yield ANOVA evaluation
- Serial or multiprocessing raster grid search with deterministic configured seeds
- GeoTIFF, GeoPackage, CSV, PNG, PDF, cache-manifest, and artifact-manifest outputs
- Configured parameter experiments

## Installation

Python 3.10 or newer is required.

```bash
python -m pip install -e ".[dev]"
```

Install Python MULTISPATI support with `python -m pip install -e ".[multispati]"`. R-backed MULTISPATI additionally needs R, the R packages `ade4`, `spdep`, and `adespatial`, and `python -m pip install -e ".[r]"`.

## Five-minute quick start

The bundled example uses small synthetic data and the vector/grid-cell workflow so it does not require private research data.

```bash
geofarmai validate --config configs/example.yaml
geofarmai run --config configs/example.yaml
```

Force raster recomputation or override raster grid-search execution:

```bash
geofarmai run --config configs/example.yaml --force
geofarmai run --config configs/example.yaml --parallel-gridsearch --n-jobs 4
geofarmai run --config configs/example.yaml --serial-gridsearch
```

Run configured experiment permutations:

```bash
geofarmai experiment --config configs/example.yaml
```

## Python API

```python
from geofarmai import run_pipeline

result = run_pipeline(config="configs/example.yaml", force=False)
print(result.best_model, result.best_k)
print(result.leaderboard)
```

Or retain a pipeline instance:

```python
from geofarmai import GeoFarmPipeline

pipeline = GeoFarmPipeline.from_yaml("configs/example.yaml")
result = pipeline.run()
```

## Inputs and outputs

The primary inputs are a soil/predictor CSV and a yield CSV. Each requires configured coordinate columns; the soil file requires every configured predictor and the yield file requires the configured yield measurement. Relative paths are resolved from the YAML file, not the current shell directory. See `docs/input_data.md` and `docs/configuration.md`.

Depending on the selected workflow, outputs include interpolated and component rasters, candidate cluster rasters, management-zone GeoPackages, metrics and leaderboard CSVs, PNG previews, PDF reports, logs, and JSON manifests. See `docs/outputs.md`.

## Relationship to MZGPT

GeoFarmAI is the reusable geospatial analysis framework. MZGPT is a separate conversational application that uses GeoFarmAI to run analyses and interpret their outputs.

GeoFarmAI contains no LLM agents, chatbot state, FastAPI endpoints, Streamlit UI, live web lookups, or conversation management.

## Citation and license

`CITATION.cff` contains placeholders pending review of authors, repository metadata, and release date. A software license has **not** been selected; `LICENSE` is a non-license placeholder and `LICENSE-TODO.md` records the required decision. Do not publicly release the project until these items are resolved.
