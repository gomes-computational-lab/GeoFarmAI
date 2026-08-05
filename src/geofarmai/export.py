import os
from pathlib import Path
from typing import Iterable, List

import geopandas as gpd
import pandas as pd

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import contextily as ctx


def zones_from_points(gdf: gpd.GeoDataFrame, labels, min_area=5000):
    # Dissolve Voronoi-like by buffering tiny amount then dissolving by label
    df = gdf.copy()
    df['zone'] = labels
    # Create small buffers to ensure contiguity, then dissolve
    polys = df.set_geometry(df.geometry.buffer(1.0)).dissolve(by='zone')
    # polys = df.buffer(1.0).dissolve(by='zone')
    # Remove tiny parts
    cleaned = polys[polys.area >= min_area]
    cleaned = gpd.GeoDataFrame(cleaned, geometry='geometry', crs=gdf.crs)
    cleaned['zone_id'] = cleaned.index
    return cleaned.reset_index(drop=True)


def _maybe_add_basemap(ax, layer: gpd.GeoDataFrame, basemap_mode: str):
    if basemap_mode != "web":
        return
    try:
        ctx.add_basemap(ax, source=ctx.providers.Stamen.TonerLite)
    except Exception:
        # Tile fetch failure should not break export; skip quietly.
        pass


def _plot_geodataframe(ax, gdf: gpd.GeoDataFrame, column: str | None, title: str,
                       cmap: str = "tab20", basemap: str = "web", categorical: bool = False):
    plot_gdf = gdf
    basemap_mode = basemap
    if basemap == "web":
        try:
            plot_gdf = gdf.to_crs(3857)
        except Exception:
            basemap_mode = "none"
    ax.set_title(title, fontsize=14)
    ax.set_axis_off()
    plot_kwargs = {"ax": ax}
    if column:
        plot_kwargs.update({"column": column, "legend": True, "cmap": cmap})
    if categorical:
        plot_kwargs["categorical"] = True
    plot_gdf.plot(**plot_kwargs)
    _maybe_add_basemap(ax, plot_gdf, basemap_mode)
    bounds = plot_gdf.total_bounds
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])


def export_pdf_report(zones: gpd.GeoDataFrame,
                      table: gpd.GeoDataFrame,
                      feature_columns: Iterable[str],
                      pdf_path: str,
                      basemap: str = "web") -> List[str]:
    pdf_path_obj = Path(pdf_path)
    os.makedirs(str(pdf_path_obj.parent or Path(".")), exist_ok=True)
    feature_columns = [c for c in feature_columns if c in table.columns]

    preview_dir = pdf_path_obj.parent / f"{pdf_path_obj.stem}_preview"
    if preview_dir.exists():
        for stale in preview_dir.glob("*.png"):
            try:
                stale.unlink()
            except OSError:
                pass
    preview_dir.mkdir(parents=True, exist_ok=True)

    image_paths: List[str] = []
    page_index = 1

    def _capture(fig):
        nonlocal page_index
        png_path = preview_dir / f"{pdf_path_obj.stem}_page{page_index:02d}.png"
        fig.savefig(png_path, dpi=200, bbox_inches="tight")
        image_paths.append(str(png_path))
        page_index += 1

    with PdfPages(str(pdf_path_obj)) as pdf:
        # Page 1: final zones
        fig, ax = plt.subplots(figsize=(11, 8.5))
        _plot_geodataframe(ax, zones, column="zone_id", title="Management Zones",
                           basemap=basemap, categorical=True)
        _capture(fig)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Additional pages for each feature
        for feature in feature_columns:
            series = table[feature]
            if series.dtype.kind not in "biufc":
                continue
            fig, ax = plt.subplots(figsize=(11, 8.5))
            _plot_geodataframe(ax, table, column=feature, title=feature,
                               basemap=basemap, cmap="viridis")
            _capture(fig)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    return image_paths


def save_package(zones, points, metrics: dict, out_path: str):
    zones.to_file(out_path, layer='zones', driver='GPKG')
    points.to_file(out_path, layer='samples', driver='GPKG')
    # Save diagnostics as a CSV next to gpkg for simplicity
    pd.DataFrame([metrics]).to_csv(os.path.splitext(out_path)[0] + '_metrics.csv', index=False)
