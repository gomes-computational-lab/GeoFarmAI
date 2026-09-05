# GeoFarmAI

GeoFarmAI is a reusable scientific Python library for geospatial agricultural
management-zone delineation. Its canonical API accepts one or many explicitly
role-labelled data sources, harmonizes their spatial support, runs existing
decomposition and clustering implementations, and returns every candidate
solution with internal and optional outcome-validation metrics. The scientific
package does not require an LLM or application server.

## Primary Python API

```python
from geofarmai import FieldDataset, GeoFarmModel

data = FieldDataset.from_csv(
    "field.csv",
    coordinates=("longitude", "latitude"),
    crs="EPSG:4326",
    predictors=["EC", "moisture", "elevation"],
    outcome="yield",  # optional; any explicitly selected numeric outcome
)

model = GeoFarmModel(
    decomposition="multispati",
    clustering=["kmeans", "gmm", "fcm"],
    k=range(2, 6),
    random_state=42,
)
result = model.fit(data)

print(result.summary())
table = result.to_dataframe()
artifacts = result.export("outputs/field_zones.gpkg")
```

Use `decomposition="none"`, `"pca"`, or `"multispati"`. The `none` pathway
retains the existing standardized-but-unreduced feature behavior. Python MULTISPATI
uses the existing `multispaeti` engine; set `multispati_engine="r"` to request
the optional existing R implementation. Explicit MULTISPATI requests fail
clearly if their selected engine is unavailable and never silently become PCA.

With one outcome, candidate selection preserves the established priority of
variance reduction followed by silhouette. Without an outcome, selection uses
silhouette followed by Calinski-Harabasz. For multiple outcomes, specify
`selection_outcome`; otherwise selection remains internal while validation
metrics are reported for every outcome. Outcome columns never enter the
predictor matrix.

`run_pipeline(...)` remains available as a deprecated compatibility wrapper.
New code should use `GeoFarmModel.fit(...)`.

## Repository layout

- `configs/project.yaml` - Main project configuration. Defines input data paths, coordinate columns, soil/yield variables, grid settings, clustering settings, experiment sweeps, and output directory.
- `data/` - Input soil and yield CSV files referenced by the config.
- `core/` - Core spatial and modeling utilities.
- `jobs/flow_mzd.py` - Baseline management zone design pipeline.
- `jobs/run_experiments.py` - Batch experiment runner for parameter sweeps.
- `agents/interactive.py` - Terminal-based LangChain agent for asking questions and running tools.
- `api/main.py` - FastAPI backend used by the Streamlit app.
- `ui/app.py` - Streamlit user interface.
- `outputs/` - Generated GeoPackages, PDFs, preview images, and metrics CSVs.
- `environment.yml` - Recommended Conda environment.
- `requirements.txt` - Python dependency list for pip-based installs.

## Environment setup

Install the base scientific package without R support:

```bash
python -m pip install -e .
```

R-backed MULTISPATI is optional and must be requested explicitly:

```bash
python -m pip install -e ".[r]"
```

The `r` extra installs the Python bridge only. A compatible R installation
and the R packages `ade4`, `spdep`, and `adespatial` must also be installed
and discoverable through a consistent `R_HOME`. If R MULTISPATI is explicitly
requested but cannot initialize or execute, GeoFarmAI raises an error and
does not silently substitute PCA.

The raster Python-MULTISPATI pathway may fall back to PCA when its engine is
unavailable. Results and metrics record the requested method, actual method,
whether R was used, and whether fallback occurred.

Structured results use a `decomposition` record with `requested_method`,
`actual_method`, `used_r`, and `fallback_occurred`. Metrics CSVs contain the
corresponding `requested_decomposition_method`, `actual_decomposition_method`,
`used_r`, and `decomposition_fallback_occurred` columns.

The recommended setup is Conda because the project uses geospatial libraries and optional R integration for MULTISPATI.

```bash
cd /Users/gomesr/Library/CloudStorage/OneDrive-UW-EauClaire/Code/ag-gpt
conda env create -f environment.yml
conda activate ag-gpt
```

If the environment already exists, update it with:

```bash
conda env update -f environment.yml --prune
conda activate ag-gpt
```

The environment uses Python 3.11 and includes the main geospatial, modeling, orchestration, API, and LLM libraries:

- Data/geospatial: `pandas`, `numpy`, `geopandas`, `pyproj`, `shapely`, `gdal`, `rasterio`, `contextily`, `mapclassify`
- Spatial statistics: `libpysal`, `esda`, `statsmodels`
- Raster/modeling: `pykrige`, `multispaeti`, `tifffile`, `scikit-learn`, `scikit-fuzzy`
- Workflow/API/UI: `prefect`, `fastapi`, `uvicorn`, `streamlit`
- Agent/LLM: `langchain`, `langchain-community`, `tabulate`
- Optional R integration: `r-base`, `r-ade4`, `r-spdep`, `r-adespatial`, `rpy2`

The Streamlit app is not listed in the current `environment.yml`. If needed, install it after activating the environment:

```bash
pip install streamlit requests
```

## Optional LLM setup

The interactive assistant defaults to Ollama. Install Ollama and pull a local model, for example:

```bash
ollama pull mistral
ollama serve
```

The default model is `mistral`. You can override it:

```bash
export OLLAMA_MODEL=mistral
export OLLAMA_BASE_URL=http://localhost:11434
```

OpenAI-compatible models can also be used by setting:

```bash
export LLM_PROVIDER=openai
export OPENAI_API_KEY=your_api_key
export OPENAI_MODEL=gpt-4o-mini
```

## Optional live lookups

The assistant can answer project questions from local files, logs, and outputs without internet access. It also behaves like a normal chatbot for stable general questions, such as definitions, explanations, writing help, brainstorming, and conceptual questions. Generic questions that clearly depend on current external information are routed to an optional live-lookup layer instead of being answered from model memory. This includes weather, news, prices, schedules, scores, current office holders, and current company roles.

Enable live weather lookup with:

```bash
export AGGPT_ENABLE_WEATHER=true
```

Weather has a specialized lookup path using `wttr.in` and requires internet access. Example:

```text
What is the weather in Chicago?
```

Enable generic live web lookup with:

```bash
export AGGPT_ENABLE_WEB_LOOKUP=true
```

By default, web lookup uses the DuckDuckGo instant-answer API. To route searches through another service, set a custom endpoint that accepts a `q` query parameter:

```bash
export AGGPT_WEB_LOOKUP_URL=https://your-lookup-service.example/search
```

If these settings are not enabled, the assistant returns a clear fallback such as `That question appears to need current external information, but live web lookup is not enabled in this app` instead of fabricating an answer. Stable general questions still go to the regular LLM. Local date/time questions are handled deterministically and do not require live lookup.

## Configuration

The main configuration is `configs/project.yaml`. Key sections are:

- `project`: Project name, CRS, soil CSV, yield CSV, coordinate fields, and variable names.
- `grid`: Cell-size rules and interpolation method. Current method is `idw`.
- `weights`: Number of KNN neighbors for spatial weights.
- `spatial_pca`: Number of spatial components and whether to use R MULTISPATI.
- `clustering`: Algorithms, k values, and random seeds.
- `raster`: Raster-first workflow settings. `target_crs: auto_utm` automatically projects longitude/latitude inputs to the local UTM zone so kriging distances, cell size, and area filtering use meters.
- `postprocess`: Minimum polygon area for final zones.
- `export`: Output directory and basemap setting.
- `experiments`: Parameter sweeps for batch runs.

Before running, confirm that the input files in `data/` match the paths and column names in `configs/project.yaml`.

Outcome-based validation is optional. New configurations declare a column in
the predictor source explicitly:

```yaml
project:
  outcome: nitrate
```

or declare a separate source:

```yaml
project:
  outcome:
    name: nitrate
    column: nitrate_mg_kg
    path: data/nitrate.csv
    x: Longitude
    y: Latitude
```

Use `outcome: null`, or omit both the generic outcome and legacy yield
sections, for an unsupervised analysis. In that mode PCA/MULTISPATI,
clustering, internal metrics, and map export continue normally; variance
reduction and ANOVA are omitted. The historical `project.yield` section is
still accepted through a compatibility adapter.

## Run from the terminal

### Run the baseline pipeline

```bash
python -m jobs.flow_mzd
```

This uses `configs/project.yaml` by default. It writes outputs to the configured `outputs/` directory. The raster pipeline writes a `run_manifest.json` file and reuses existing outputs when the soil CSV, yield CSV, and relevant config settings have not changed.

To force a full rerun:

```bash
python -m jobs.flow_mzd --force
```

The clustering grid search runs serially by default for easier testing and reproducible timing. To use multiprocessing for the independent clustering candidates without editing `configs/project.yaml`, run:

```bash
python -m jobs.flow_mzd --force --parallel-gridsearch --n-jobs 4
```

To explicitly run serially:

```bash
python -m jobs.flow_mzd --force --serial-gridsearch
```

Use a conservative `--n-jobs` value. Larger values can be faster, but each worker needs memory for the raster feature matrix and clustering model.

With `raster.enabled: true`, the baseline uses the raster-first pipeline and writes outputs under:

```text
outputs/sacaton_field_raster/
```

Expected outputs include:

- `rasters/soil/` - PyKrige ordinary kriging GeoTIFFs and stats files for soil variables.
- `rasters/yield/` - PyKrige ordinary kriging GeoTIFFs and stats files for the configured yield variable.
- `pca/PC*.tif` - MULTISPATI-PCA component rasters.
- `pca/pca_summary_stats.txt` - PCA eigenvalue/loadings summary.
- `clusters/best_clusters.tif` - Best cluster raster selected by yield variance reduction.
- `clusters/best_ch_score_clusters.tif` - Best cluster raster selected by Calinski-Harabasz pseudo-F score.
- `sacaton_field_zones.gpkg` - Polygonized best management zones.
- `sacaton_field.pdf` - PDF report assembled from raster previews.
- `sacaton_field_metrics.csv` - Best-run summary metrics.
- `sacaton_field_gridsearch.csv` - Clustering leaderboard.
- `preview/` - PNG previews for Streamlit display, including the yield-selected cluster map, the Calinski-Harabasz-selected cluster map, and method-comparison plots for Calinski-Harabasz pseudo-F score, variance reduction, silhouette score, and ANOVA significance.
- `outputs/logs/baseline_*.log` - Terminal output captured during the baseline run.

To use the previous vector/grid-cell pipeline instead, set `raster.enabled: false` in `configs/project.yaml`.
That workflow supports `idw`, `nearest`, and `buffer_mean` reconciliation.
Vector `kriging` is not implemented and is rejected explicitly; ordinary
kriging is available only through the raster workflow.

The default baseline uses:

```text
soil + terrain variables
MULTISPATI-PCA features
kmeans, agglomerative, gmm, and fcm clustering
```

### Run configured experiments

```bash
python -m jobs.run_experiments --cfg configs/project.yaml
```

This expands the `experiments` section in `configs/project.yaml`, runs each parameter combination, and writes experiment outputs to `outputs/`. With the raster pipeline enabled, each experiment writes its own `outputs/sacaton_field_raster_<experiment>/` folder. Summary files include:

- `outputs/sacaton_field_experiments_summary.csv`
- `outputs/sacaton_field_experiments_gridsearch.csv`
- `outputs/logs/experiments_*.log`

The configured experiment named `feature_representation_comparison` compares:

- soil-only variables with PCA
- soil-only variables without PCA
- soil + terrain variables with PCA
- soil + terrain variables without PCA

Each run uses the four clustering algorithms:

```text
kmeans
agglomerative
gmm
fcm
```

and evaluates the configured `k` values. The key non-yield metric for comparing inter-cluster versus intra-cluster separation is:

```text
ch_score
```

This is the Calinski-Harabasz pseudo-F score. Higher values indicate stronger separation among clusters relative to within-cluster dispersion.

### Use the interactive agent

```bash
python -m agents.interactive --cfg configs/project.yaml
```

Example prompts:

```text
Which experiments achieved the highest variance reduction?
Compare baseline metrics to the latest experiment runs.
Suggest new parameter combinations to explore novel clusters.
List available artifacts.
```

The agent can inspect existing metrics, summarize the configuration, list artifacts, run the baseline pipeline, run experiments when explicitly requested, and answer questions about the results.

## Run through Streamlit

The Streamlit UI talks to the FastAPI backend, so start the API first.

### 1. Start the API backend

From the project root:

```bash
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Useful API endpoints:

- `GET /artifacts` - List generated files.
- `GET /artifacts/{path}` - Download a generated artifact.
- `POST /chat` - Ask the LLM agent a question.
- `POST /pipeline/run` - Run the baseline pipeline.

### 2. Start the Streamlit app

Open a second terminal, activate the same environment, and run:

```bash
conda activate ag-gpt
streamlit run ui/app.py
```

By default the UI expects the API at `http://localhost:8000`. To use another API URL:

```bash
export AGGPT_API_URL=http://127.0.0.1:8000
streamlit run ui/app.py
```

Long raster runs can take several minutes. The Streamlit app uses a longer pipeline timeout by default. To override it, set:

```bash
export AGGPT_PIPELINE_TIMEOUT=1800
streamlit run ui/app.py
```

The Streamlit interface provides:

- A chat panel for asking questions about the project and results.
- A `Run Pipeline` button to execute the baseline workflow.
- A `Refresh Outputs` button to reload generated artifacts.
- A `Clear Outputs` button to delete generated output files while preserving the output directory placeholder.
- Preview images from generated PDF/map outputs.
- Download links for CSV, PDF, image, and GeoPackage artifacts.

## Run logs

Baseline and experiment runs automatically record terminal output to timestamped log files while still printing progress to the terminal.

Log files are written under:

```text
outputs/logs/
```

Examples:

```text
outputs/logs/baseline_20260630_231500.log
outputs/logs/experiments_20260630_231800.log
```

These logs include progress messages, R/MULTISPATI warnings, pipeline stages, experiment numbers, parameter combinations, export messages, and final output paths. They are useful for checking whether a long run is still progressing and for documenting run provenance.

To watch the newest experiment log in another terminal:

```bash
tail -f outputs/logs/experiments_*.log
```

## How the pipeline works

1. **Input ingestion**
   `jobs/flow_mzd.py` reads the soil and yield CSV paths from `configs/project.yaml`. The raster pipeline keeps the same user-facing inputs as before: one soil CSV and one yield CSV.

2. **PyKrige raster generation**
   Soil variables and the configured yield column are kriged with ordinary kriging. The output is one GeoTIFF per variable plus a stats text file containing prediction range, value intervals, and fitted variogram parameters.

3. **Raster stack alignment**
   Soil rasters are intersected spatially and resampled onto a shared reference grid. Only pixels with complete values across the selected PCA variables are used downstream.

4. **Feature representation**
   If `raster.use_pca: true`, the raster stack is standardized and passed to `multispaeti.MultispatiPCA` using a KNN pixel-connectivity graph. If `multispaeti` is unavailable, the code falls back to standard `sklearn` PCA. If `raster.use_pca: false`, clustering uses the standardized kriged raster variables directly without PCA.

5. **Clustering**
   The component matrix is clustered using the algorithms listed in `clustering.algorithms`: `kmeans`, `agglomerative`, `gmm`, and `fcm`. The pipeline evaluates multiple values of `k`; seed-based algorithms also evaluate configured seeds.

6. **Evaluation**
   Candidate zone solutions are evaluated using:

   - `vr`: variance reduction in yield within zones relative to total field variance.
   - `asc`: average silhouette coefficient for cluster separation.
   - `ch_score`: Calinski-Harabasz pseudo-F score comparing between-cluster versus within-cluster dispersion in PCA/component space.
   - `anova_p`: ANOVA p-value for yield differences among zones.

   The best solution is selected primarily by variance reduction, then silhouette score.

7. **Post-processing and export**
   The best cluster raster is polygonized into management zones, small polygons are filtered using `postprocess.min_area_m2`, and outputs are written as GeoTIFFs, GeoPackage, metrics CSV, gridsearch CSV, PDF report, stats text files, and preview images.

## Main commands

```bash
# Activate environment
conda activate ag-gpt

# Run baseline pipeline
python -m jobs.flow_mzd

# Run experiment sweep
python -m jobs.run_experiments --cfg configs/project.yaml

# Run terminal agent
python -m agents.interactive --cfg configs/project.yaml

# Start API backend
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000

# Start Streamlit UI
streamlit run ui/app.py
```

## Notes and troubleshooting

- If R MULTISPATI fails, the code falls back to PCA and continues running.
- If basemap tiles cannot be fetched, PDF/map export skips the basemap rather than failing the run.
- Run logs are written to `outputs/logs/` for baseline and experiment runs.
- Generated outputs can become large. Clear the `outputs/` directory before a fresh experiment run if you want only the newest artifacts.
- The LLM agent should not run experiments unless explicitly asked. It is designed to inspect existing outputs first.
- If Streamlit cannot connect, confirm that `uvicorn` is running on port `8000` and that `AGGPT_API_URL` matches the API address.

## Citation-oriented method summary

Ag-GPT implements a reproducible geospatial workflow for agricultural management zone delineation. Soil and yield point observations are projected to a metric coordinate system and reconciled to an adaptive grid using spatial interpolation. Selected soil covariates are transformed into spatial components using MULTISPATI when available, with PCA as a fallback. Grid cells are clustered using Gaussian mixture models and fuzzy c-means across user-defined values of cluster number and random seed. Candidate zone solutions are evaluated using yield variance reduction, silhouette score, and ANOVA tests of yield separation among zones. Final zones are exported as geospatial layers, summary metrics, leaderboards, and map reports. A LangChain-based assistant, FastAPI backend, and Streamlit interface provide interactive access to the workflow and generated artifacts.
