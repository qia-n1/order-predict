from __future__ import annotations

import h3
import pandas as pd


SPATIAL_COL = "h3_index"


def resolve_spatial_column(columns: pd.Index) -> str:
    for candidate in [SPATIAL_COL, "geohash"]:
        if candidate in columns:
            return candidate
    raise KeyError("Input file must contain 'h3_index' or legacy 'geohash'")


def normalize_h3_column(df: pd.DataFrame, spatial_col: str | None = None) -> pd.DataFrame:
    spatial_col = spatial_col or resolve_spatial_column(df.columns)
    normalized = df.copy()
    if spatial_col != SPATIAL_COL:
        normalized[SPATIAL_COL] = normalized[spatial_col]
    normalized[SPATIAL_COL] = normalized[SPATIAL_COL].astype(str)
    return normalized


def coarsen_h3_index(h3_index: str, target_resolution: int) -> str:
    if not h3.is_valid_cell(h3_index):
        return h3_index
    current_resolution = h3.get_resolution(h3_index)
    if target_resolution >= current_resolution:
        return h3_index
    return h3.cell_to_parent(h3_index, target_resolution)


def coarsen_h3_series(series: pd.Series, target_resolution: int) -> pd.Series:
    return series.map(lambda value: coarsen_h3_index(str(value), target_resolution))


def h3_center_latlng(h3_index: str) -> tuple[float, float]:
    lat, lng = h3.cell_to_latlng(h3_index)
    return float(lat), float(lng)


def h3_ring(h3_index: str, k: int) -> list[str]:
    return sorted(h3.grid_disk(h3_index, k))


def h3_to_geo_boundary(h3_index: str) -> list[list[float]]:
    boundary = h3.cell_to_boundary(h3_index)
    return [[float(lat), float(lng)] for lat, lng in boundary]
