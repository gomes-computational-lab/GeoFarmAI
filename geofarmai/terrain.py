"""Metric DEM interpolation and reproducible terrain derivatives.

The point workflow delegates ordinary kriging, grid construction, and raster
writing to GeoFarmAI's retained raster implementation. Terrain derivatives are
calculated only after horizontal and vertical units have both been normalized
to metres.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import CRS

from .exceptions import OptionalDependencyError, TerrainProcessingError


_METRE_UNITS = {"m", "meter", "meters", "metre", "metres"}
_FOOT_UNITS = {"ft", "foot", "feet", "international_foot"}
_US_SURVEY_FOOT_UNITS = {"us_survey_foot", "us survey foot", "survey_foot"}


@dataclass(frozen=True, slots=True)
class TerrainDerivatives:
    """Slope and general-curvature arrays derived from a metric DEM."""

    slope_degrees: np.ndarray
    general_curvature: np.ndarray


@dataclass(frozen=True, slots=True)
class TerrainResult:
    """Georeferenced terrain rasters and their scientific provenance."""

    elevation: Path
    slope: Path
    curvature: Path
    provenance: Mapping[str, Any]
    provenance_path: Path | None = None

    @property
    def predictor_rasters(self) -> dict[str, Path]:
        """Return stable predictor names mapped to their raster paths."""

        return {
            "elevation": self.elevation,
            "slope_degrees": self.slope,
            "general_curvature": self.curvature,
        }


def terrain_derivatives(
    elevation_m: np.ndarray,
    *,
    x_resolution_m: float,
    y_resolution_m: float | None = None,
    nodata: float | None = None,
) -> TerrainDerivatives:
    """Derive Horn slope and general curvature from a metric DEM.

    Slope uses the Horn 3x3 finite-difference operator and is returned in
    degrees. General curvature is the elevation Laplacian
    ``d2z/dx2 + d2z/dy2`` using second-order central differences. With metres
    for both elevation and grid spacing, curvature units are inverse metres.
    Positive curvature denotes a locally concave-up surface; negative
    curvature denotes a locally concave-down surface.

    The outer raster cell is always nodata. Horn slope additionally requires
    all nine cells in its 3x3 window to be finite. Curvature requires the
    centre and its north, south, east, and west neighbors to be finite. No
    derivative values are filled across nodata.
    """

    z = np.asarray(elevation_m, dtype="float64")
    if z.ndim != 2:
        raise TerrainProcessingError("Elevation must be a two-dimensional raster array.")
    if z.shape[0] < 3 or z.shape[1] < 3:
        raise TerrainProcessingError(
            "Terrain derivatives require an elevation raster of at least 3 by 3 cells."
        )
    dx = _positive_resolution(x_resolution_m, "x_resolution_m")
    dy = dx if y_resolution_m is None else _positive_resolution(
        y_resolution_m, "y_resolution_m"
    )
    if nodata is not None and np.isfinite(nodata):
        z = z.copy()
        z[z == float(nodata)] = np.nan

    slope = np.full(z.shape, np.nan, dtype="float64")
    curvature = np.full(z.shape, np.nan, dtype="float64")

    nw, north, ne = z[:-2, :-2], z[:-2, 1:-1], z[:-2, 2:]
    west, center, east = z[1:-1, :-2], z[1:-1, 1:-1], z[1:-1, 2:]
    sw, south, se = z[2:, :-2], z[2:, 1:-1], z[2:, 2:]

    horn_valid = np.logical_and.reduce(
        tuple(np.isfinite(item) for item in (nw, north, ne, west, center, east, sw, south, se))
    )
    dz_dx = ((ne + 2.0 * east + se) - (nw + 2.0 * west + sw)) / (8.0 * dx)
    dz_dy = ((sw + 2.0 * south + se) - (nw + 2.0 * north + ne)) / (8.0 * dy)
    slope_inner = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))
    slope[1:-1, 1:-1] = np.where(horn_valid, slope_inner, np.nan)

    curvature_valid = np.logical_and.reduce(
        tuple(np.isfinite(item) for item in (north, west, center, east, south))
    )
    d2z_dx2 = (east - 2.0 * center + west) / (dx * dx)
    d2z_dy2 = (south - 2.0 * center + north) / (dy * dy)
    curvature_inner = d2z_dx2 + d2z_dy2
    curvature[1:-1, 1:-1] = np.where(
        curvature_valid, curvature_inner, np.nan
    )

    return TerrainDerivatives(slope, curvature)


def derive_terrain(
    elevation_raster: str | Path,
    *,
    output_directory: str | Path | None = None,
    provenance: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> TerrainResult:
    """Write slope and general-curvature rasters from a metric elevation raster.

    The input must use a projected CRS whose horizontal axes are measured in
    metres, and its elevation values must already be in metres. Raster rotation
    or shear is rejected because the finite-difference operators assume a
    north-up rectilinear grid.
    """

    rasterio = _rasterio()
    elevation_path = Path(elevation_raster).expanduser().resolve()
    if not elevation_path.is_file():
        raise TerrainProcessingError(f"Elevation raster does not exist: {elevation_path}")

    destination = (
        elevation_path.parent / "terrain"
        if output_directory is None
        else Path(output_directory).expanduser().resolve()
    )
    destination.mkdir(parents=True, exist_ok=True)
    slope_path = destination / "slope_degrees.tif"
    curvature_path = destination / "general_curvature.tif"
    provenance_path = destination / "terrain_provenance.json"
    _ensure_outputs_available(
        (slope_path, curvature_path, provenance_path), overwrite=overwrite
    )

    with rasterio.open(elevation_path) as dataset:
        if dataset.count != 1:
            raise TerrainProcessingError("Elevation raster must contain exactly one band.")
        if dataset.crs is None:
            raise TerrainProcessingError("Elevation raster must declare a projected metric CRS.")
        analysis_crs = CRS.from_user_input(dataset.crs)
        _validate_metric_crs(analysis_crs)
        transform = dataset.transform
        if not np.isclose(transform.b, 0.0) or not np.isclose(transform.d, 0.0):
            raise TerrainProcessingError(
                "Terrain derivatives require a north-up raster without rotation or shear."
            )
        elevation = dataset.read(1).astype("float64")
        nodata = dataset.nodata
        profile = dataset.profile.copy()

    derivatives = terrain_derivatives(
        elevation,
        x_resolution_m=abs(float(transform.a)),
        y_resolution_m=abs(float(transform.e)),
        nodata=nodata,
    )
    output_profile = profile.copy()
    output_profile.update(dtype="float64", count=1, nodata=np.nan)
    with rasterio.open(slope_path, "w", **output_profile) as dataset:
        dataset.write(derivatives.slope_degrees, 1)
    with rasterio.open(curvature_path, "w", **output_profile) as dataset:
        dataset.write(derivatives.general_curvature, 1)

    details: dict[str, Any] = dict(provenance or {})
    details.update(
        {
            "analysis_crs": analysis_crs.to_string(),
            "converted_elevation_units": "metres",
            "dem_resolution_m": {
                "x": abs(float(transform.a)),
                "y": abs(float(transform.e)),
            },
            "slope_method": "Horn 3x3 finite-difference operator",
            "slope_units": "degrees",
            "curvature_definition": "general curvature (elevation Laplacian: d2z/dx2 + d2z/dy2)",
            "curvature_units": "m^-1",
            "curvature_sign_convention": (
                "positive is locally concave-up; negative is locally concave-down"
            ),
            "nodata_policy": (
                "outer cells are nodata; derivatives are nodata when any value "
                "required by that finite-difference neighborhood is nodata"
            ),
        }
    )
    provenance_path.write_text(
        json.dumps(details, indent=2, sort_keys=True), encoding="utf-8"
    )
    return TerrainResult(
        elevation=elevation_path,
        slope=slope_path,
        curvature=curvature_path,
        provenance=details,
        provenance_path=provenance_path,
    )


def terrain_from_elevation_points(
    data: str | Path | pd.DataFrame | gpd.GeoDataFrame,
    *,
    elevation_column: str,
    elevation_units: str,
    cell_size_m: float,
    output_directory: str | Path,
    x_column: str | None = None,
    y_column: str | None = None,
    source_crs: str | int | CRS | None = None,
    target_crs: str | int | CRS | None = None,
    interpolation_config: Mapping[str, Any] | None = None,
    bounds: tuple[float, float, float, float] | None = None,
    overwrite: bool = False,
) -> TerrainResult:
    """Interpolate metric elevation and derive reusable terrain rasters.

    ``data`` may be a CSV path, a pandas DataFrame, or a GeoDataFrame. Geometry
    is authoritative for a GeoDataFrame; otherwise explicit x/y columns and a
    source CRS are required. ``cell_size_m`` is deliberately mandatory so a
    scientifically inappropriate sub-metre default cannot be selected merely
    from the field extent.

    Ordinary kriging delegates to :mod:`core.raster_pipeline`; no independent
    interpolation implementation is introduced here. Bounds, when supplied,
    use ``(x_min, x_max, y_min, y_max)`` in the target CRS.
    """

    cell_size = _positive_resolution(cell_size_m, "cell_size_m")
    output_dir = Path(output_directory).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    elevation_path = output_dir / "elevation_dem_m.tif"
    _ensure_outputs_available((elevation_path,), overwrite=overwrite)

    points, source_label = _point_frame(
        data,
        x_column=x_column,
        y_column=y_column,
        source_crs=source_crs,
    )
    if elevation_column not in points.columns:
        raise TerrainProcessingError(
            f"Elevation column {elevation_column!r} is missing from the point data."
        )
    original_crs = CRS.from_user_input(points.crs)
    analysis_crs = _target_metric_crs(points, target_crs)
    projected = points.to_crs(analysis_crs).reset_index(drop=True)
    elevation = pd.to_numeric(projected[elevation_column], errors="coerce").to_numpy(
        dtype="float64"
    )
    coordinates = np.column_stack((projected.geometry.x, projected.geometry.y))
    valid = np.isfinite(coordinates).all(axis=1) & np.isfinite(elevation)
    if int(valid.sum()) < 3:
        raise TerrainProcessingError(
            "Ordinary kriging requires at least three observations with finite "
            "coordinates and elevation."
        )
    factor, normalized_source_units = _elevation_conversion(elevation_units)
    elevation_m = elevation[valid] * factor
    x = coordinates[valid, 0]
    y = coordinates[valid, 1]

    try:
        from core.raster_pipeline import build_arcgis_grid, ordinary_krige, write_raster
    except OptionalDependencyError:
        raise
    except ImportError as exc:  # pragma: no cover - optional dependency environment
        raise OptionalDependencyError(
            "Terrain interpolation requires GeoFarmAI's raster dependencies. "
            "Install them with `pip install \"geofarmai[raster]\"`."
        ) from exc

    grid_x, grid_y, resolved_cell_size = build_arcgis_grid(
        x,
        y,
        cell_size=cell_size,
        bounds=bounds,
    )
    if len(grid_x) < 3 or len(grid_y) < 3:
        raise TerrainProcessingError(
            "The requested terrain grid must contain at least 3 rows and 3 columns."
        )
    config = dict(interpolation_config or {})
    dem, variogram_parameters = ordinary_krige(
        x,
        y,
        elevation_m,
        grid_x,
        grid_y,
        config,
        "euclidean",
    )
    write_raster(
        dem,
        elevation_path,
        analysis_crs.to_string(),
        grid_x,
        grid_y,
        resolved_cell_size,
    )

    resolved_interpolation = _resolved_kriging_configuration(config)
    provenance = {
        "source": source_label,
        "source_crs": original_crs.to_string(),
        "analysis_crs": analysis_crs.to_string(),
        "source_elevation_column": elevation_column,
        "source_elevation_units": normalized_source_units,
        "converted_elevation_units": "metres",
        "elevation_conversion_factor_to_metres": factor,
        "source_observations": int(len(points)),
        "interpolated_observations": int(valid.sum()),
        "interpolation_method": "ordinary kriging",
        "interpolation_configuration": resolved_interpolation,
        "fitted_variogram_parameters": [
            float(value) for value in np.asarray(variogram_parameters).ravel()
        ],
        "dem_resolution_m": {"x": resolved_cell_size, "y": resolved_cell_size},
        "raster_bounds_target_crs": {
            "x_min": float(grid_x.min() - resolved_cell_size / 2.0),
            "x_max": float(grid_x.max() + resolved_cell_size / 2.0),
            "y_min": float(grid_y.min() - resolved_cell_size / 2.0),
            "y_max": float(grid_y.max() + resolved_cell_size / 2.0),
        },
    }
    return derive_terrain(
        elevation_path,
        output_directory=output_dir,
        provenance=provenance,
        overwrite=overwrite,
    )


def _point_frame(
    data: str | Path | pd.DataFrame | gpd.GeoDataFrame,
    *,
    x_column: str | None,
    y_column: str | None,
    source_crs: str | int | CRS | None,
) -> tuple[gpd.GeoDataFrame, str]:
    if isinstance(data, gpd.GeoDataFrame):
        if data.empty:
            raise TerrainProcessingError("Elevation point data cannot be empty.")
        if data.crs is None:
            if source_crs is None:
                raise TerrainProcessingError("Elevation point geometry must declare a CRS.")
            frame = data.set_crs(source_crs, allow_override=True).copy()
        else:
            frame = data.copy()
            if source_crs is not None and CRS.from_user_input(source_crs) != CRS.from_user_input(
                frame.crs
            ):
                raise TerrainProcessingError(
                    "source_crs conflicts with the GeoDataFrame CRS."
                )
        if not (frame.geometry.geom_type == "Point").all():
            raise TerrainProcessingError("Elevation geometry must contain only points.")
        return frame, "GeoDataFrame"

    if isinstance(data, (str, Path)):
        path = Path(data).expanduser().resolve()
        if not path.is_file():
            raise TerrainProcessingError(f"Elevation point file does not exist: {path}")
        frame_data = pd.read_csv(path)
        source_label = str(path)
    elif isinstance(data, pd.DataFrame):
        frame_data = data.copy()
        source_label = "DataFrame"
    else:
        raise TerrainProcessingError(
            "Elevation points must be supplied as a CSV path, DataFrame, or GeoDataFrame."
        )
    if frame_data.empty:
        raise TerrainProcessingError("Elevation point data cannot be empty.")
    if source_crs is None:
        raise TerrainProcessingError("source_crs is required for tabular elevation points.")
    if not x_column or not y_column:
        raise TerrainProcessingError(
            "x_column and y_column are required for tabular elevation points."
        )
    missing = [name for name in (x_column, y_column) if name not in frame_data.columns]
    if missing:
        raise TerrainProcessingError(f"Missing coordinate columns: {missing}.")
    x = pd.to_numeric(frame_data[x_column], errors="coerce")
    y = pd.to_numeric(frame_data[y_column], errors="coerce")
    geometry = gpd.points_from_xy(x, y)
    return (
        gpd.GeoDataFrame(frame_data, geometry=geometry, crs=source_crs),
        source_label,
    )


def _target_metric_crs(
    points: gpd.GeoDataFrame,
    target_crs: str | int | CRS | None,
) -> CRS:
    if target_crs is not None:
        selected = CRS.from_user_input(target_crs)
    else:
        source = CRS.from_user_input(points.crs)
        if source.is_projected and _crs_uses_metres(source):
            selected = source
        else:
            from core.crs import to_utm_auto

            _, selected_name = to_utm_auto(points.to_crs("EPSG:4326"))
            selected = CRS.from_user_input(selected_name)
    _validate_metric_crs(selected)
    return selected


def _validate_metric_crs(crs: CRS) -> None:
    if not crs.is_projected:
        raise TerrainProcessingError(
            "Terrain processing requires a projected CRS; geographic degrees are invalid."
        )
    if not _crs_uses_metres(crs):
        raise TerrainProcessingError(
            "Terrain processing requires projected horizontal coordinates in metres."
        )


def _crs_uses_metres(crs: CRS) -> bool:
    axes = crs.axis_info
    return bool(axes) and all(
        axis.unit_conversion_factor is not None
        and np.isclose(float(axis.unit_conversion_factor), 1.0)
        for axis in axes[:2]
    )


def _elevation_conversion(units: str) -> tuple[float, str]:
    normalized = str(units).strip().lower().replace("-", "_")
    if normalized in _METRE_UNITS:
        return 1.0, "metres"
    if normalized in _FOOT_UNITS:
        return 0.3048, "feet"
    if normalized in _US_SURVEY_FOOT_UNITS:
        return 1200.0 / 3937.0, "US survey feet"
    raise TerrainProcessingError(
        f"Unsupported elevation units {units!r}. Use metres, feet, or US survey feet."
    )


def _resolved_kriging_configuration(config: Mapping[str, Any]) -> dict[str, Any]:
    nested = config.get("kriging", {})
    if not isinstance(nested, Mapping):
        nested = {}
    return {
        "variogram_model": config.get(
            "variogram_model", nested.get("variogram_model", "spherical")
        ),
        "nlags": int(config.get("nlags", nested.get("nlags", 12))),
        "weight": bool(config.get("weight", nested.get("weight", False))),
        "backend": config.get("backend", "loop"),
        "n_closest_points": int(
            config.get("n_closest_points", nested.get("n_closest_points", 24))
        ),
        "coordinates_type": "euclidean",
    }


def _positive_resolution(value: float, name: str) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise TerrainProcessingError(f"{name} must be a finite positive number.") from exc
    if not np.isfinite(resolved) or resolved <= 0:
        raise TerrainProcessingError(f"{name} must be a finite positive number.")
    return resolved


def _ensure_outputs_available(paths: tuple[Path, ...], *, overwrite: bool) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        raise TerrainProcessingError(
            "Terrain output already exists; choose a new output directory or set "
            f"overwrite=True: {existing}"
        )


def _rasterio():
    try:
        import rasterio
    except ImportError as exc:  # pragma: no cover - clean base environment
        raise OptionalDependencyError(
            "Terrain raster processing requires optional raster dependencies. "
            "Install them with `pip install \"geofarmai[raster]\"`."
        ) from exc
    return rasterio
