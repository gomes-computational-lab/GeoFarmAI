# GeoFarmAI release acceptance examples

These four examples are deterministic, end-to-end release criteria for the
public GeoFarmAI scientific API. All CSV files in `data/` are synthetic and
contain no research or private field data.

Run a Python example from an editable installation at the repository root:

```bash
python examples/single_csv_yield.py
python examples/heterogeneous_yield.py
python examples/nitrate_outcome.py
python examples/predictors_only.py
```

Pass an output GeoPackage path to exercise export, for example:

```bash
python examples/single_csv_yield.py /tmp/single_csv_yield.gpkg
```

Each Python example has an equivalent canonical YAML document in `configs/`.
The YAML paths are resolved relative to the YAML file and can be executed with
`run_from_config(...)` or the GeoFarmAI CLI.

## Expected result structures

| Scenario | Spatial support | Predictors | Outcomes | Selection | Expected candidates |
| --- | --- | --- | --- | --- | --- |
| Single CSV + yield | Direct source rows | EC, moisture, elevation | yield | maximize yield variance reduction | KMeans at K=2 and K=3 |
| Heterogeneous sources + yield | 10 m reconciliation grid | soil EC/moisture and terrain elevation/slope | yield | maximize yield variance reduction | KMeans at K=2 and K=3 |
| Nitrate outcome | Soil observation support with reconciled nitrate | EC, CEC, organic_matter | nitrate | maximize nitrate variance reduction | KMeans at K=2 and K=3 |
| Predictors only | 10 m reconciliation grid | soil EC/moisture and terrain elevation | none | none | KMeans at K=2 and K=3, all retained |

Every result contains candidate labels, internal silhouette and
Calinski-Harabasz metrics, spatial metadata, decomposition metadata, source and
variable provenance, and the model configuration. Outcome scenarios also
contain variance reduction and ANOVA metrics. With `selection: null`, no
candidate is promoted to a winner; callers can still export the unlabelled
analysis support or explicitly choose one retained candidate for export.

`GeoFarmResult.export(...)` creates a GeoPackage plus metrics and candidate
CSV files. The automated release tests use temporary directories, so no
generated outputs are committed.

The examples are source-distribution/repository resources, not import-package
data. They are included in the source archive so the acceptance scenarios can
be reproduced from a release source tree, and intentionally omitted from the
wheel's `geofarmai` package namespace.
