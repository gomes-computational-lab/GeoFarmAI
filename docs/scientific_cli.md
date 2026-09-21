# GeoFarmAI scientific CLI

The GeoFarmAI command-line interface validates and runs source-based scientific
YAML configurations through the same canonical objects used by the Python API.
It has no conversational, web, or application-server dependencies.

## Commands

From an installed development checkout:

```bash
geofarmai --help
geofarmai validate --config examples/configs/single_csv_yield.yaml
geofarmai run --config examples/configs/single_csv_yield.yaml
```

Historical project-style YAML files can be translated through the isolated
legacy configuration adapter:

```bash
geofarmai validate --legacy --config /path/to/private-project.yaml
geofarmai run --legacy --config /path/to/private-project.yaml
```

The legacy flag translates configuration into `FieldDataset` and
`GeoFarmModel`; it does not invoke the removed Prefect jobs. Research datasets
and their project-specific configuration are intentionally not distributed
with GeoFarmAI.

## Python equivalent

```python
from geofarmai import load_config, run_from_config

configuration = load_config("examples/configs/single_csv_yield.yaml")
dataset, model = configuration.build()
result = model.fit(dataset)

# Equivalent convenience call
result = run_from_config("examples/configs/single_csv_yield.yaml")
```

Use explicit source and variable-role declarations for new projects. Outcome
variables are optional and never enter the predictor matrix.
