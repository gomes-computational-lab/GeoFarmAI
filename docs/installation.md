# Installation

Use Python 3.10 or newer, preferably in an isolated environment outside the repository.

```bash
python -m pip install -e ".[dev]"
```

The base install includes the Python analytical/GIS workflow. `.[multispati]` enables Python `MultispatiPCA`; `.[r]` installs the Python R bridge but users must separately install R and the R packages `ade4`, `spdep`, and `adespatial`. `.[docs]` installs documentation tooling.

GeoFarmAI does not require Prefect. Normal Python orchestration replaces the former task decorators without changing stage order.
