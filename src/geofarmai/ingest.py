from dataclasses import dataclass
import pandas as pd
import geopandas as gpd

@dataclass
class Schema:
    id: str
    x: str
    y: str
    variables: list[str]
    crs: str = "EPSG:3857"

class Ingestor:
    def __init__(self, cfg):
        pj = cfg["project"]
        self.schema = Schema(id=pj["id_column"], x=pj["x"], y=pj["y"], variables=pj["variables"], crs=pj["crs"])  # type: ignore

    def read_csv(self, path: str) -> gpd.GeoDataFrame:
        df = pd.read_csv(path)
        for col in [self.schema.id, self.schema.x, self.schema.y] + self.schema.variables:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[self.schema.x], df[self.schema.y]), crs=self.schema.crs)
        return gdf
