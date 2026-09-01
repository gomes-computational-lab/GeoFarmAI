# Ag-GPT Terminal GIS Pipeline

This guide is for the GIS group running the Python management-zone pipeline
without the Streamlit web interface.

## Environment

From the repository root:

```bash
conda activate ag-gpt
pip install -r requirements.txt
```

The pipeline expects two CSV files:

- a soil/terrain/spectral CSV;
- a yield CSV.

The default project uses:

```text
data/soil.csv
data/yield.csv
```

Both files must include coordinate columns. The default coordinate columns are:

```text
Longitude
Latitude
```

The default yield column is:

```text
Yld_Mass_Dry_lb_ac
```

## Simple Run

Run the default configured pipeline:

```bash
python -m jobs.flow_mzd --cfg configs/project.yaml
```

Force a full rerun instead of using cached outputs:

```bash
python -m jobs.flow_mzd --cfg configs/project.yaml --force
```

Run the grid search with multiprocessing:

```bash
python -m jobs.flow_mzd --cfg configs/project.yaml --force --parallel-gridsearch --n-jobs 4
```

The worker count is capped at 4 for student-facing runs.

## Run Directly From Two CSV Files

The GIS-specific runner lets you pass CSV files and analysis settings without
editing the YAML file.

```bash
python scripts/run_gis_pipeline.py \
  --soil-csv data/soil.csv \
  --yield-csv data/yield.csv \
  --project-name sacaton_field \
  --soil-variables Slope Curve EC_DP Red IR Moisture Soil_Temp_C \
  --required-soil-variables Slope Moisture EC_DP \
  --yield-column Yld_Mass_Dry_lb_ac \
  --algorithms kmeans agglomerative gmm fcm \
  --k-values 2 3 4 5 6 \
  --seeds 42 1337 \
  --use-pca \
  --force
```

Parallel version:

```bash
python scripts/run_gis_pipeline.py \
  --soil-csv data/soil.csv \
  --yield-csv data/yield.csv \
  --project-name sacaton_field \
  --soil-variables Slope Curve EC_DP Red IR Moisture Soil_Temp_C \
  --required-soil-variables Slope Moisture EC_DP \
  --yield-column Yld_Mass_Dry_lb_ac \
  --algorithms kmeans agglomerative gmm fcm \
  --k-values 2 3 4 5 6 \
  --seeds 42 1337 \
  --use-pca \
  --parallel-gridsearch \
  --n-jobs 4 \
  --force
```

Preview the resolved configuration without running the pipeline:

```bash
python scripts/run_gis_pipeline.py \
  --soil-csv data/soil.csv \
  --yield-csv data/yield.csv \
  --dry-run
```

## Common Arguments

```text
--soil-csv                  soil/terrain/spectral CSV
--yield-csv                 yield CSV
--project-name              output project name
--soil-x / --soil-y          soil coordinate columns
--yield-x / --yield-y        yield coordinate columns
--yield-column              yield measurement column
--soil-variables            predictor columns
--required-soil-variables   columns that must be present
--pca-variables             variables used for PCA/raw clustering
--algorithms                kmeans agglomerative gmm fcm
--k-values                  candidate zone counts
--seeds                     random seeds for seeded algorithms
--use-pca                   use MULTISPATI-PCA/PCA features
--raw-features              cluster raw standardized raster features
--parallel-gridsearch       run clustering candidates in parallel
--serial-gridsearch         run clustering candidates serially
--n-jobs                    worker count, capped at 4
--force                     recompute instead of using cached outputs
--dry-run                   print resolved config and exit
```

## What The Pipeline Does

The terminal pipeline:

1. Reads the soil and yield CSV files.
2. Reprojects coordinates to a field-scale projected CRS.
3. Kriges soil/terrain/spectral variables into raster surfaces.
4. Kriges the yield variable into a raster surface.
5. Aligns the rasters into a common stack.
6. Uses MULTISPATI-PCA when PCA is enabled.
7. Runs the clustering grid search.
8. Scores each candidate using yield variance reduction, average silhouette
   coefficient, Calinski-Harabasz score, and ANOVA p-value.
9. Exports maps, rasters, preview images, tables, logs, and manifests.

## Main Outputs

Outputs are written under:

```text
outputs/<project>_raster/
```

Important files include:

```text
<project>_gridsearch.csv       all clustering candidates and metrics
<project>_metrics.csv          selected best run metrics
<project>_zones.gpkg           management-zone vector output
<project>.pdf                  report
preview/                       PNG maps and comparison figures
clusters/                      cluster GeoTIFFs
rasters/                       kriged input rasters
run_manifest.json              cache and run metadata
visual_manifest.json           generated visual catalog
```

Logs are written under:

```text
outputs/logs/
```

## Baseline Versus Experiments

The baseline command:

```bash
python -m jobs.flow_mzd --cfg configs/project.yaml
```

runs one configured feature representation with an internal clustering grid
search.

The experiment command:

```bash
python -m jobs.run_experiments --cfg configs/project.yaml
```

runs the configured outer experiment grid, such as comparing soil-only versus
soil-plus-terrain variables and PCA versus raw features.
