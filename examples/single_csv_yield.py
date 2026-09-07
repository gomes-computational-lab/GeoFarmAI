"""Acceptance scenario 1: one CSV with an explicit yield outcome."""

from __future__ import annotations

import sys
from pathlib import Path

from geofarmai import FieldDataset, GeoFarmModel


DATA_PATH = Path(__file__).resolve().parent / "data" / "single_csv_yield.csv"


def run():
    data = FieldDataset.from_csv(
        DATA_PATH,
        source_id="field_observations",
        coordinates=("x", "y"),
        crs="EPSG:32615",
        predictors=["EC", "moisture", "elevation"],
        outcome="yield",
    )
    return GeoFarmModel(
        decomposition="pca",
        n_components=2,
        clustering="kmeans",
        k=[2, 3],
        selection="variance_reduction",
        selection_outcome="yield",
        random_state=42,
    ).fit(data)


def main() -> None:
    result = run()
    if len(sys.argv) > 1:
        result.export(Path(sys.argv[1]))
    print(result.summary())


if __name__ == "__main__":
    main()
