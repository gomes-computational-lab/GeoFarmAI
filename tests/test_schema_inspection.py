from __future__ import annotations

import pandas as pd
import pytest

from geofarmai import coordinate_candidates, inspect_schema


def test_schema_inspection_reports_columns_dtypes_numeric_and_missingness():
    frame = pd.DataFrame(
        {
            "Longitude": [-91.5, -91.4],
            "Latitude": [44.8, 44.9],
            "nitrate": [3.0, None],
            "yield": [100.0, 110.0],
            "EC": ["low", "high"],
        }
    )

    inspection = inspect_schema(frame)

    assert inspection.columns == ("Longitude", "Latitude", "nitrate", "yield", "EC")
    assert set(inspection.numeric_columns) == {"Longitude", "Latitude", "nitrate", "yield"}
    assert inspection.missing_counts["nitrate"] == 1
    assert inspection.missing_fractions["nitrate"] == pytest.approx(0.5)
    assert inspection.coordinate_candidates.x == ("Longitude",)
    assert inspection.coordinate_candidates.y == ("Latitude",)
    assert inspection.coordinate_candidates.is_unambiguous
    assert inspection.coordinate_candidates.unambiguous_pair == ("Longitude", "Latitude")

    # Inspection deliberately contains no predictor/outcome inference, even
    # for scientifically suggestive names.
    assert not hasattr(inspection, "roles")
    assert "nitrate" not in inspection.coordinate_candidates.x
    assert "yield" not in inspection.coordinate_candidates.y
    assert "EC" not in inspection.coordinate_candidates.x


def test_coordinate_detection_returns_ambiguous_candidates_without_selecting_one():
    frame = pd.DataFrame(
        {"lon": [1.0], "lng": [1.0], "lat": [2.0], "y": [2.0]}
    )

    candidates = coordinate_candidates(frame)

    assert candidates.x == ("lon", "lng")
    assert candidates.y == ("lat", "y")
    assert not candidates.is_unambiguous
    assert candidates.unambiguous_pair is None


def test_coordinate_detection_supports_easting_and_northing():
    candidates = coordinate_candidates(
        pd.DataFrame({"Easting": [500000.0], "Northing": [4900000.0]})
    )

    assert candidates.unambiguous_pair == ("Easting", "Northing")


def test_schema_inspection_accepts_a_csv_path(tmp_path):
    path = tmp_path / "observations.csv"
    pd.DataFrame({"x": [1.0], "y": [2.0], "protein": [11.5]}).to_csv(path, index=False)

    inspection = inspect_schema(path)

    assert inspection.row_count == 1
    assert inspection.coordinate_candidates.unambiguous_pair == ("x", "y")
    assert "protein" in inspection.numeric_columns
