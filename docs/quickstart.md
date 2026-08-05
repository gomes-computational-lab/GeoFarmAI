# Quickstart

```bash
geofarmai validate --config configs/example.yaml
geofarmai run --config configs/example.yaml
```

The example configuration resolves its synthetic inputs relative to `configs/example.yaml` and writes ignored artifacts to `outputs/`.

```python
from geofarmai import GeoFarmPipeline

result = GeoFarmPipeline.from_yaml("configs/example.yaml").run()
print(result.metrics)
print(result.artifacts)
```
