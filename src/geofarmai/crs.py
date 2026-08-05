import math
import geopandas as gpd

def utm_zone_from_lon(lon):
    return int(math.floor((lon + 180) / 6) + 1)

def to_utm_auto(gdf: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, str]:
    lon = gdf.geometry.x.mean()
    lat = gdf.geometry.y.mean()
    zone = utm_zone_from_lon(lon)
    north = lat >= 0
    epsg = f"EPSG:{326 if north else 327}{zone:02d}"  # 326xx north, 327xx south
    return gdf.to_crs(epsg), epsg
