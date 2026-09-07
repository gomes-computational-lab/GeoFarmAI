"""Acceptance scenario 4: heterogeneous predictors with no outcome."""

from __future__ import annotations

import sys
from pathlib import Path

from geofarmai import CoordinateSpec, DataSource, FieldDataset, GeoFarmModel, VariableSpec


DATA_DIR = Path(__file__).resolve().parent / "data"
CRS = "EPSG:32615"


def run():
    coordinates = CoordinateSpec(x="x", y="y", crs=CRS)
    soil = DataSource.from_csv(
        DATA_DIR / "predictors_only_soil.csv",
        source_id="soil",
        coordinates=coordinates,
        variables=[
            VariableSpec("EC", "predictor", domain="soil"),
            VariableSpec("moisture", "predictor", domain="soil"),
        ],
    )
    terrain = DataSource.from_csv(
        DATA_DIR / "predictors_only_terrain.csv",
        source_id="terrain",
        coordinates=coordinates,
        variables=[VariableSpec("elevation", "predictor", domain="terrain")],
    )
    data = FieldDataset.from_sources([soil, terrain])
    return GeoFarmModel(
        decomposition="pca",
        n_components=2,
        clustering="kmeans",
        k=[2, 3],
        selection=None,
        random_state=42,
        harmonization_strategy="auto",
        reconciliation_method="idw",
        harmonization_params={"cell_size": 10.0},
    ).fit(data)


def main() -> None:
    result = run()
    if len(sys.argv) > 1:
        result.export(Path(sys.argv[1]))
    print(result.summary())


if __name__ == "__main__":
    main()
