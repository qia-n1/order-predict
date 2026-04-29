import argparse
import math
from pathlib import Path

import h3
import pandas as pd


WEEKDAY_NAMES = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


def encode_h3(latitude: float, longitude: float, resolution: int) -> str | None:
    if pd.isna(latitude) or pd.isna(longitude):
        return None
    return h3.latlng_to_cell(float(latitude), float(longitude), resolution)


def resolve_column(columns: pd.Index, candidates: list[str], label: str) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise KeyError(f"Missing {label} column. Tried: {', '.join(candidates)}")


def build_features(df: pd.DataFrame, time_bins: int, h3_resolution: int) -> pd.DataFrame:
    if time_bins <= 0 or 1440 % time_bins != 0:
        raise ValueError("time_bins must be a positive divisor of 1440, e.g. 24 or 96")

    pickup_time_col = resolve_column(
        df.columns,
        ["pickup_datetime", "tpep_pickup_datetime"],
        "pickup datetime",
    )
    pickup_lat_col = resolve_column(
        df.columns,
        ["pickup_latitude"],
        "pickup latitude",
    )
    pickup_lon_col = resolve_column(
        df.columns,
        ["pickup_longitude"],
        "pickup longitude",
    )

    features = df.copy()
    pickup_dt = pd.to_datetime(features[pickup_time_col], errors="coerce")
    invalid_rows = pickup_dt.isna().sum()
    if invalid_rows:
        raise ValueError(f"Found {invalid_rows} rows with invalid pickup datetime values")

    minutes_per_bin = 1440 / time_bins
    minutes_since_midnight = (
        pickup_dt.dt.hour * 60
        + pickup_dt.dt.minute
        + pickup_dt.dt.second / 60
        + pickup_dt.dt.microsecond / 60000000
    )
    time_bin_index = (minutes_since_midnight // minutes_per_bin).astype(int)
    time_bin_index = time_bin_index.clip(upper=time_bins - 1)

    bin_start_minutes = time_bin_index * minutes_per_bin
    start_hours = (bin_start_minutes // 60).astype(int)
    start_minutes = (bin_start_minutes % 60).astype(int)
    bin_center_fraction = (time_bin_index + 0.5) / time_bins

    weekday_index = pickup_dt.dt.weekday

    features["h3_index"] = [
        encode_h3(lat, lon, h3_resolution)
        for lat, lon in zip(features[pickup_lat_col], features[pickup_lon_col])
    ]
    features["time_cat"] = [
        f"{hour:02d}:{minute:02d}"
        for hour, minute in zip(start_hours, start_minutes)
    ]
    features["time_num"] = bin_center_fraction
    features["time_cos"] = features["time_num"].map(lambda value: math.cos(value * 2 * math.pi))
    features["time_sin"] = features["time_num"].map(lambda value: math.sin(value * 2 * math.pi))
    features["day_cat"] = weekday_index.map(lambda idx: WEEKDAY_NAMES[idx])
    features["day_num"] = (weekday_index + features["time_num"]) / 7
    features["day_cos"] = features["day_num"].map(lambda value: math.cos(value * 2 * math.pi))
    features["day_sin"] = features["day_num"].map(lambda value: math.sin(value * 2 * math.pi))
    features["weekend"] = weekday_index.isin([5, 6]).astype(int)

    return features


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent
    default_input = project_root / "data" / "NYC_YellowTaxi_2015_100k.csv"
    default_output = project_root / "data" / "NYC_YellowTaxi_2015_100k_features.csv"

    parser = argparse.ArgumentParser(description="Generate documented feature-engineering columns")
    parser.add_argument("--input", type=Path, default=default_input, help="Input CSV path")
    parser.add_argument("--output", type=Path, default=default_output, help="Output CSV path")
    parser.add_argument("--time-bins", type=int, default=24, help="Number of daily time bins, e.g. 24 or 96")
    parser.add_argument("--h3-resolution", type=int, default=8, help="H3 resolution, e.g. 7 or 8")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Reading raw data from {args.input}...")
    df = pd.read_csv(args.input)

    print(
        f"Building features with {args.time_bins} time bins and H3 resolution {args.h3_resolution}..."
    )
    feature_df = build_features(df, args.time_bins, args.h3_resolution)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(args.output, index=False)

    preview_columns = [
        "h3_index",
        "time_cat",
        "time_num",
        "time_cos",
        "time_sin",
        "day_cat",
        "day_num",
        "day_cos",
        "day_sin",
        "weekend",
    ]
    print(f"Saved feature data to {args.output}")
    print(feature_df[preview_columns].head())


if __name__ == "__main__":
    main()
