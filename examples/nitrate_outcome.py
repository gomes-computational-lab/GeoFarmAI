"""Acceptance scenario 3: nitrate explicitly configured as an outcome."""

from __future__ import annotations

import sys
from pathlib import Path

from geofarmai import CoordinateSpec, DataSource, FieldDataset, GeoFarmModel, VariableSpec


DATA_DIR = Path(__file__).resolve().parent / "data"
CRS = "EPSG:32615"


def run():
    coordinates = CoordinateSpec(x="x", y="y", crs=CRS)
    soil = DataSource.from_csv(
        DATA_DIR / "nitrate_soil.csv",
        source_id="soil",
        coordinates=coordinates,
        variables=[
            VariableSpec("EC", "predictor", domain="soil"),
            VariableSpec("CEC", "predictor", domain="soil", units="cmol(+)/kg"),
            VariableSpec("organic_matter", "predictor", domain="soil"),
        ],
    )
    nitrate = DataSource.from_csv(
        DATA_DIR / "nitrate_outcome.csv",
        source_id="nitrate_samples",
        coordinates=coordinates,
        variables=[VariableSpec("nitrate", "outcome", domain="nutrient", units="mg/kg")],
    )
    data = FieldDataset.from_sources([soil, nitrate])
    return GeoFarmModel(
        decomposition="pca",
        n_components=2,
        clustering="kmeans",
        k=[2, 3],
        selection="variance_reduction",
        selection_outcome="nitrate",
        random_state=42,
        harmonization_strategy="auto",
        reconciliation_method="idw",
    ).fit(data)


def main() -> None:
    result = run()
    if len(sys.argv) > 1:
        result.export(Path(sys.argv[1]))
    print(result.summary())


if __name__ == "__main__":
    main()
