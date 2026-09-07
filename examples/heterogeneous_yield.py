"""Acceptance scenario 2: heterogeneous soil, terrain, and yield sources."""

from __future__ import annotations

import sys
from pathlib import Path

from geofarmai import CoordinateSpec, DataSource, FieldDataset, GeoFarmModel, VariableSpec


DATA_DIR = Path(__file__).resolve().parent / "data"
CRS = "EPSG:32615"


def run():
    coordinates = CoordinateSpec(x="x", y="y", crs=CRS)
    soil = DataSource.from_csv(
        DATA_DIR / "heterogeneous_soil.csv",
        source_id="soil",
        coordinates=coordinates,
        variables=[
            VariableSpec("EC", "predictor", domain="soil", units="dS/m"),
            VariableSpec("moisture", "predictor", domain="soil", units="percent"),
        ],
    )
    terrain = DataSource.from_csv(
        DATA_DIR / "heterogeneous_terrain.csv",
        source_id="terrain",
        coordinates=coordinates,
        variables=[
            VariableSpec("elevation", "predictor", domain="terrain", units="m"),
            VariableSpec("slope", "predictor", domain="terrain", units="percent"),
        ],
    )
    harvest = DataSource.from_csv(
        DATA_DIR / "heterogeneous_yield.csv",
        source_id="harvest",
        coordinates=coordinates,
        variables=[VariableSpec("yield", "outcome", domain="crop", units="Mg/ha")],
    )
    data = FieldDataset.from_sources([soil, terrain, harvest])
    return GeoFarmModel(
        decomposition="pca",
        n_components=2,
        clustering="kmeans",
        k=[2, 3],
        selection="variance_reduction",
        selection_outcome="yield",
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
