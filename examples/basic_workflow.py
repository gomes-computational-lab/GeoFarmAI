from pathlib import Path

from geofarmai import GeoFarmPipeline


project_root = Path(__file__).resolve().parents[1]
pipeline = GeoFarmPipeline.from_yaml(project_root / "configs" / "example.yaml")
result = pipeline.run()
print(result.as_dict())
