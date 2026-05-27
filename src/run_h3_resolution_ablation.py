import argparse
import json
import time
from pathlib import Path

import h3
import numpy as np
import pandas as pd
import xgboost as xgb

from h3_utils import SPATIAL_COL, h3_center_latlng
from train_xgboost_v2 import (
    add_history_features,
    add_spatial_intensity_features,
    add_target_encoding,
    add_time_features,
    build_holiday_features,
    fetch_weather_features,
    load_orders,
    mae,
    make_splits,
    pcc,
    rmse,
    summarize_subset,
    within_tolerance,
)


FEATURE_COLUMNS = [
    "center_lat",
    "center_lon",
    "hour",
    "weekday",
    "day_of_month",
    "week_of_month",
    "weekend",
    "is_rush_hour",
    "is_late_night",
    "time_sin",
    "time_cos",
    "day_sin",
    "day_cos",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_24",
    "lag_48",
    "lag_168",
    "rolling_mean_3",
    "rolling_mean_6",
    "rolling_mean_12",
    "rolling_mean_24",
    "rolling_mean_48",
    "rolling_max_24",
    "rolling_max_48",
    "rolling_std_24",
    "rolling_std_48",
    "parent_total_lag_1",
    "parent_total_lag_24",
    "parent_mean_lag_1",
    "parent_mean_rolling_24",
    "temperature",
    "apparent_temperature",
    "precipitation",
    "rain",
    "cloud_cover",
    "wind_speed",
    "humidity",
    "weather_code",
    "weather_clear",
    "weather_cloudy",
    "weather_fog",
    "weather_drizzle",
    "weather_rain",
    "weather_snow",
    "weather_thunder",
    "is_precipitating",
    "is_hot",
    "is_humid",
    "is_holiday",
    "is_month_start",
    "is_month_end",
    "h3_mean_count",
    "parent_mean_count_train",
    "hour_mean_count",
    "weekday_mean_count",
    "holiday_mean_count",
]


WEATHER_COLUMNS = [
    "temperature",
    "apparent_temperature",
    "precipitation",
    "rain",
    "cloud_cover",
    "wind_speed",
    "humidity",
    "weather_code",
    "weather_clear",
    "weather_cloudy",
    "weather_fog",
    "weather_drizzle",
    "weather_rain",
    "weather_snow",
    "weather_thunder",
    "is_precipitating",
    "is_hot",
    "is_humid",
]


REQUIRED_HISTORY = [
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_24",
    "lag_48",
    "lag_168",
    "rolling_mean_3",
    "rolling_mean_6",
    "rolling_mean_12",
    "rolling_mean_24",
    "rolling_mean_48",
    "rolling_max_24",
    "parent_total_lag_1",
    "parent_total_lag_24",
    "parent_mean_lag_1",
    "parent_mean_rolling_24",
]


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Run H3 resolution ablation with the V2 feature set")
    parser.add_argument("--input", type=Path, default=project_root / "data" / "NYC_YellowTaxi_2015_100k_features.csv")
    parser.add_argument("--output-dir", type=Path, default=project_root / "models" / "h3_resolution_ablation")
    parser.add_argument("--picture-dir", type=Path, default=project_root / "models" / "picture")
    parser.add_argument("--weather-cache", type=Path, default=project_root / "models" / "xgboost_hourly_v2_h3" / "weather_features.csv")
    parser.add_argument("--resolutions", type=int, nargs="+", default=[6, 7, 8, 9])
    parser.add_argument("--freq", type=str, default="1h")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--valid-ratio", type=float, default=0.15)
    parser.add_argument("--test-start-date", type=str, default=None, help="Date boundary for test split, e.g. '2014-06-24'. Overrides ratio-based split.")
    parser.add_argument("--num-boost-round", type=int, default=350)
    parser.add_argument("--early-stopping-rounds", type=int, default=40)
    parser.add_argument("--weight-zero", type=float, default=1.0)
    parser.add_argument("--weight-nonzero", type=float, default=3.0)
    parser.add_argument("--weight-high", type=float, default=6.0)
    parser.add_argument("--weight-very-high", type=float, default=10.0)
    parser.add_argument("--high-count-threshold", type=float, default=4.0)
    parser.add_argument("--very-high-count-threshold", type=float, default=8.0)
    return parser.parse_args()


def aggregate_to_h3_panel(df: pd.DataFrame, freq: str, resolution: int) -> pd.DataFrame:
    orders = df.copy()
    orders[SPATIAL_COL] = [
        h3.latlng_to_cell(lat, lon, resolution)
        for lat, lon in zip(orders["pickup_latitude"], orders["pickup_longitude"])
    ]
    orders["time_slot"] = orders["pickup_datetime"].dt.floor(freq)

    counts = orders.groupby([SPATIAL_COL, "time_slot"]).size().reset_index(name="order_count")
    h3_features = counts[[SPATIAL_COL]].drop_duplicates().reset_index(drop=True)
    centers = h3_features[SPATIAL_COL].map(h3_center_latlng)
    h3_features["center_lat"] = centers.map(lambda value: value[0])
    h3_features["center_lon"] = centers.map(lambda value: value[1])

    h3_frame = h3_features[[SPATIAL_COL]].drop_duplicates().reset_index(drop=True)
    slot_frame = pd.DataFrame({"time_slot": pd.date_range(counts["time_slot"].min(), counts["time_slot"].max(), freq=freq)})
    panel = h3_frame.merge(slot_frame, how="cross")
    panel = panel.merge(counts, on=[SPATIAL_COL, "time_slot"], how="left")
    panel = panel.merge(h3_features, on=SPATIAL_COL, how="left")
    panel["order_count"] = panel["order_count"].fillna(0.0).astype(float)
    return panel.sort_values([SPATIAL_COL, "time_slot"]).reset_index(drop=True)


def load_or_fetch_weather(cache_path: Path, start_time: pd.Timestamp, end_time: pd.Timestamp) -> pd.DataFrame:
    if cache_path.exists():
        weather = pd.read_csv(cache_path)
        weather["time_slot"] = pd.to_datetime(weather["time_slot"])
        return weather
    weather = fetch_weather_features(start_time, end_time)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    weather.to_csv(cache_path, index=False)
    return weather


def compute_row_weights(df: pd.DataFrame, args: argparse.Namespace) -> np.ndarray:
    counts = df["order_count"].to_numpy()
    weights = np.full(len(df), args.weight_zero, dtype=np.float32)
    weights[counts > 0] = args.weight_nonzero
    weights[counts >= args.high_count_threshold] = args.weight_high
    weights[counts >= args.very_high_count_threshold] = args.weight_very_high
    return weights


def prepare_model_matrix(df: pd.DataFrame, weights: np.ndarray | None = None) -> xgb.DMatrix:
    matrix = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=True)
    labels = df["order_count"].to_numpy(dtype=np.float32, copy=True)
    return xgb.DMatrix(matrix, label=labels, weight=weights, feature_names=FEATURE_COLUMNS, nthread=1)


def split_panel(panel: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return make_splits(panel, args.train_ratio, args.valid_ratio, args.test_start_date)


def add_metric_block(prefix: str, df: pd.DataFrame, pred: np.ndarray, metrics: dict[str, float | int], threshold: float) -> None:
    pred = np.clip(pred, 0.0, None)
    nonzero_mask = df["order_count"].to_numpy() > 0
    nonzero = summarize_subset(df["order_count"], pred, 1.0)
    high = summarize_subset(df["order_count"], pred, threshold)
    metrics[f"{prefix}_rmse"] = rmse(df["order_count"], pred)
    metrics[f"{prefix}_mae"] = mae(df["order_count"], pred)
    metrics[f"{prefix}_pcc"] = pcc(df["order_count"], pred)
    metrics[f"{prefix}_nonzero_rows"] = nonzero["rows"]
    metrics[f"{prefix}_nonzero_rmse"] = nonzero["rmse"]
    metrics[f"{prefix}_nonzero_mae"] = nonzero["mae"]
    metrics[f"{prefix}_nonzero_pcc"] = nonzero["pcc"]
    metrics[f"{prefix}_nonzero_within_1"] = within_tolerance(df.loc[nonzero_mask, "order_count"], pred[nonzero_mask], 1.0)
    metrics[f"{prefix}_nonzero_within_2"] = within_tolerance(df.loc[nonzero_mask, "order_count"], pred[nonzero_mask], 2.0)
    metrics[f"{prefix}_high_rows"] = high["rows"]
    metrics[f"{prefix}_high_rmse"] = high["rmse"]
    metrics[f"{prefix}_high_mae"] = high["mae"]
    metrics[f"{prefix}_pcc_y_ge_4"] = high["pcc"]


def train_one_resolution(orders: pd.DataFrame, resolution: int, args: argparse.Namespace) -> dict[str, float | int | str]:
    started = time.perf_counter()
    parent_resolution = max(0, resolution - 1)
    panel = aggregate_to_h3_panel(orders, args.freq, resolution)
    weather = load_or_fetch_weather(args.weather_cache, pd.Timestamp(panel["time_slot"].min()), pd.Timestamp(panel["time_slot"].max()))
    holidays = build_holiday_features(panel["time_slot"])

    panel = add_time_features(panel)
    panel = add_history_features(panel)
    panel = add_spatial_intensity_features(panel, parent_resolution)
    panel = panel.merge(weather, on="time_slot", how="left")
    panel = panel.merge(holidays, on="time_slot", how="left")
    panel[WEATHER_COLUMNS] = panel[WEATHER_COLUMNS].fillna(0.0)
    panel[["is_holiday", "is_month_start", "is_month_end"]] = panel[["is_holiday", "is_month_start", "is_month_end"]].fillna(0).astype(int)
    panel = panel.dropna(subset=REQUIRED_HISTORY).reset_index(drop=True)

    train_df, valid_df, test_df = split_panel(panel, args)
    train_df = add_target_encoding(train_df, train_df)
    valid_df = add_target_encoding(train_df, valid_df)
    test_df = add_target_encoding(train_df, test_df)

    train_matrix = prepare_model_matrix(train_df, compute_row_weights(train_df, args))
    valid_matrix = prepare_model_matrix(valid_df)
    test_matrix = prepare_model_matrix(test_df)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": ["rmse", "mae"],
        "eta": 0.03,
        "max_depth": 9,
        "min_child_weight": 4,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "lambda": 1.0,
        "alpha": 0.1,
        "nthread": 1,
        "seed": 42,
    }
    evals_result: dict[str, dict[str, list[float]]] = {}
    booster = xgb.train(
        params=params,
        dtrain=train_matrix,
        num_boost_round=args.num_boost_round,
        evals=[(train_matrix, "train"), (valid_matrix, "valid")],
        early_stopping_rounds=args.early_stopping_rounds,
        evals_result=evals_result,
        verbose_eval=False,
    )

    valid_pred = booster.predict(valid_matrix)
    test_pred = booster.predict(test_matrix)
    metrics: dict[str, float | int | str] = {
        "h3_resolution": resolution,
        "parent_resolution": parent_resolution,
        "avg_edge_km": float(h3.average_hexagon_edge_length(resolution, unit="km")),
        "avg_area_km2": float(h3.average_hexagon_area(resolution, unit="km^2")),
        "grid_count": int(panel[SPATIAL_COL].nunique()),
        "panel_rows": int(len(panel)),
        "train_rows": int(len(train_df)),
        "valid_rows": int(len(valid_df)),
        "test_rows": int(len(test_df)),
        "train_zero_ratio": float((train_df["order_count"] == 0).mean()),
        "valid_zero_ratio": float((valid_df["order_count"] == 0).mean()),
        "test_zero_ratio": float((test_df["order_count"] == 0).mean()),
        "best_iteration": int(booster.best_iteration),
        "training_seconds": float(time.perf_counter() - started),
    }
    add_metric_block("valid", valid_df, valid_pred, metrics, args.high_count_threshold)
    add_metric_block("test", test_df, test_pred, metrics, args.high_count_threshold)

    result_dir = args.output_dir / f"res_{resolution}"
    result_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(result_dir / "model.json")
    with open(result_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=True, indent=2)
    with open(result_dir / "evals_result.json", "w", encoding="utf-8") as file:
        json.dump(evals_result, file, ensure_ascii=True, indent=2)
    return metrics


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    orders = load_orders(args.input)
    results = []
    for resolution in args.resolutions:
        print(f"Running H3 resolution {resolution}...", flush=True)
        metrics = train_one_resolution(orders, resolution, args)
        results.append(metrics)
        print(
            f"res={resolution} grids={metrics['grid_count']} "
            f"valid_mae={metrics['valid_mae']:.4f} test_mae={metrics['test_mae']:.4f}",
            flush=True,
        )
        pd.DataFrame(results).to_csv(args.output_dir / "h3_resolution_ablation.csv", index=False)


if __name__ == "__main__":
    main()
