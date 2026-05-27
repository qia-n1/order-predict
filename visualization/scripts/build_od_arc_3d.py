r"""
Build a 3D NYC taxi OD flow map (deck.gl ArcLayer + MapLibre).

生成深色 3D 底图 + deck.gl ArcLayer 真三维 OD 飞线 + 按小时切换的交互 HTML。

依赖：仅需仓库根 requirements.txt（numpy, pandas），无额外包。

用法（在 order-predict-may_jun 根目录）：
  .venv\Scripts\python visualization\scripts\build_od_arc_3d.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "NYC_YellowTaxi_2015_MayJun_OD.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "visualization" / "output" / "od_arc_3d.html"
_TEMPLATE = Path(__file__).resolve().parents[1] / "templates" / "od_arc_3d.html"

NYC_MIN_LON = -74.35
NYC_MAX_LON = -73.55
NYC_MIN_LAT = 40.45
NYC_MAX_LAT = 41.05
DEFAULT_OD_GRID_DEGREES = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a 3D NYC taxi OD flow map (deck.gl + MapLibre).")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--od-grid-degrees", type=float, default=DEFAULT_OD_GRID_DEGREES)
    parser.add_argument("--min-lon", type=float, default=NYC_MIN_LON)
    parser.add_argument("--max-lon", type=float, default=NYC_MAX_LON)
    parser.add_argument("--min-lat", type=float, default=NYC_MIN_LAT)
    parser.add_argument("--max-lat", type=float, default=NYC_MAX_LAT)
    return parser.parse_args()


def resolve_column(columns: pd.Index, candidates: list[str], label: str) -> str:
    for c in candidates:
        if c in columns:
            return c
    raise KeyError(f"Missing {label}. Tried: {candidates}")


def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0088
    a = (math.sin(math.radians(lat2 - lat1) / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def grid_center(values, step):
    return pd.Series(np.floor(values.to_numpy(dtype=float) / step) * step + step / 2).round(6)


def load_orders(args):
    df = pd.read_csv(args.input)
    time_col = resolve_column(df.columns, ["pickup_datetime", "tpep_pickup_datetime"], "pickup datetime")
    p_lon_col = resolve_column(df.columns, ["pickup_longitude"], "pickup longitude")
    p_lat_col = resolve_column(df.columns, ["pickup_latitude"], "pickup latitude")
    d_lon_col = resolve_column(df.columns, ["dropoff_longitude"], "dropoff longitude")
    d_lat_col = resolve_column(df.columns, ["dropoff_latitude"], "dropoff latitude")

    p_dt = pd.to_datetime(df[time_col], errors="coerce")
    for c in [p_lon_col, p_lat_col, d_lon_col, d_lat_col]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    mask = (p_dt.notna()
            & df[p_lon_col].between(args.min_lon, args.max_lon)
            & df[d_lon_col].between(args.min_lon, args.max_lon)
            & df[p_lat_col].between(args.min_lat, args.max_lat)
            & df[d_lat_col].between(args.min_lat, args.max_lat))

    orders = pd.DataFrame({
        "pickup_longitude": df.loc[mask, p_lon_col].to_numpy(dtype=float),
        "pickup_latitude": df.loc[mask, p_lat_col].to_numpy(dtype=float),
        "dropoff_longitude": df.loc[mask, d_lon_col].to_numpy(dtype=float),
        "dropoff_latitude": df.loc[mask, d_lat_col].to_numpy(dtype=float),
    })
    orders["pickup_datetime"] = p_dt.loc[mask].values
    orders["hour"] = orders["pickup_datetime"].dt.hour.astype(int)
    orders["trip_distance_km"] = [haversine_km(*r) for r in zip(
        orders["pickup_longitude"], orders["pickup_latitude"],
        orders["dropoff_longitude"], orders["dropoff_latitude"])]

    stats = {"raw_rows": len(df), "valid_rows": len(orders), "dropped_rows": len(df) - len(orders),
             "date_start": orders["pickup_datetime"].min().isoformat(),
             "date_end": orders["pickup_datetime"].max().isoformat()}
    return orders.reset_index(drop=True), stats


def build_od_flows(orders, grid_degrees):
    go = orders.copy()
    go["origin_lat"] = grid_center(go["pickup_latitude"], grid_degrees)
    go["origin_lon"] = grid_center(go["pickup_longitude"], grid_degrees)
    go["target_lat"] = grid_center(go["dropoff_latitude"], grid_degrees)
    go["target_lon"] = grid_center(go["dropoff_longitude"], grid_degrees)

    groups = go.groupby(["hour", "origin_lat", "origin_lon", "target_lat", "target_lon"], observed=True)
    od = groups.agg(order_count=("hour", "size")).reset_index()
    od = od.sort_values(["hour", "order_count"], ascending=[True, True])

    flows = {str(h): [] for h in range(24)}
    stats = []
    for h in range(24):
        hf = od.loc[od["hour"] == h]
        flows[str(h)] = [[round(float(r.origin_lat), 6), round(float(r.origin_lon), 6),
                          round(float(r.target_lat), 6), round(float(r.target_lon), 6),
                          int(r.order_count)]
                         for r in hf.itertuples(index=False)]
        stats.append({"flow_count": len(hf), "max_count": int(hf["order_count"].max()) if len(hf) else 0})

    hour_totals = orders.groupby("hour", observed=True).size().reindex(range(24), fill_value=0).astype(int).tolist()
    return flows, stats, hour_totals, od


def main():
    args = parse_args()
    orders, stats = load_orders(args)
    flows, flow_stats, hour_totals, od_groups = build_od_flows(orders, args.od_grid_degrees)

    min_lon = min(orders["pickup_longitude"].min(), orders["dropoff_longitude"].min())
    max_lon = max(orders["pickup_longitude"].max(), orders["dropoff_longitude"].max())
    min_lat = min(orders["pickup_latitude"].min(), orders["dropoff_latitude"].min())
    max_lat = max(orders["pickup_latitude"].max(), orders["dropoff_latitude"].max())
    top_hour = int(max(range(24), key=lambda h: hour_totals[h]))
    max_flow_count = int(od_groups["order_count"].max()) if len(od_groups) else 0

    meta = {"od_trip_count": len(orders), "od_flow_count": len(od_groups),
            "od_grid_degrees": float(args.od_grid_degrees),
            "top_hour": top_hour, "max_hour_total": max(hour_totals) if hour_totals else 0,
            "max_flow_count": max_flow_count,
            "max_trip_distance_km": float(orders["trip_distance_km"].max()),
            "bounds": {"min_lon": float(min_lon), "max_lon": float(max_lon),
                       "min_lat": float(min_lat), "max_lat": float(max_lat)},
            "center": {"lon": float((min_lon + max_lon) / 2), "lat": float((min_lat + max_lat) / 2)},
            **stats}

    payload = {"meta": meta, "hour_totals": hour_totals, "flow_stats_by_hour": flow_stats,
               "flows_by_hour": flows}

    if not _TEMPLATE.is_file():
        raise SystemExit(f"模板缺失: {_TEMPLATE}")

    html = _TEMPLATE.read_text(encoding="utf-8")
    html = html.replace("@@EMBED_PAYLOAD@@", json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")

    try:
        output_url = f"http://127.0.0.1:8765/{args.output.resolve().relative_to(PROJECT_ROOT).as_posix()}"
    except ValueError:
        output_url = args.output.resolve().as_uri()

    print(f"3D OD map -> {args.output}")
    print("CDN scripts use absolute https:// URLs, so they will not resolve to file:///D:/unpkg/...")
    print("Recommended preview:")
    print(f"  cd {PROJECT_ROOT}")
    print("  python -m http.server 8765")
    print(f"  {output_url}")


if __name__ == "__main__":
    main()
