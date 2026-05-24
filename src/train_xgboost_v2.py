import argparse
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd
import xgboost as xgb
from pandas.tseries.holiday import USFederalHolidayCalendar

from h3_utils import SPATIAL_COL, coarsen_h3_series, h3_center_latlng, normalize_h3_column, resolve_spatial_column


NYC_LATITUDE = 40.7128
NYC_LONGITUDE = -74.0060


def _empty_weather(time_slots: pd.Series) -> pd.DataFrame:
    slots = time_slots.drop_duplicates().sort_values()
    empty = pd.DataFrame({"time_slot": slots})
    for col in ["temperature", "apparent_temperature", "precipitation", "rain",
                "cloud_cover", "wind_speed", "humidity", "weather_code",
                "weather_clear", "weather_cloudy", "weather_fog", "weather_drizzle",
                "weather_rain", "weather_snow", "weather_thunder",
                "is_precipitating", "is_hot", "is_humid"]:
        empty[col] = 0.0
    return empty


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Train an XGBoost demand forecasting model with external features")
    parser.add_argument(
        "--input",
        type=Path,
        default=project_root / "data" / "NYC_YellowTaxi_2015_100k_features.csv",
        help="Engineered order-level CSV path",
    )
    parser.add_argument("--freq", type=str, default="1h", help="Aggregation frequency, e.g. 1h or 15min")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "models" / "xgboost_hourly_v2",
        help="Directory for model and reports",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Time-based train split ratio")
    parser.add_argument("--valid-ratio", type=float, default=0.15, help="Time-based validation split ratio")
    parser.add_argument("--num-boost-round", type=int, default=600, help="Max boosting rounds")
    parser.add_argument("--early-stopping-rounds", type=int, default=50, help="Early stopping rounds")
    parser.add_argument("--h3-resolution", type=int, default=8, help="Target H3 resolution for spatial aggregation")
    parser.add_argument("--parent-resolution", type=int, default=7, help="Coarser H3 parent resolution for context features")
    parser.add_argument("--weight-zero", type=float, default=1.0, help="Training weight for zero-demand rows")
    parser.add_argument("--weight-nonzero", type=float, default=3.0, help="Training weight for 0 < y < mid-threshold rows")
    parser.add_argument("--weight-high", type=float, default=6.0, help="Training weight for mid-threshold <= y < very-high-threshold rows")
    parser.add_argument("--weight-very-high", type=float, default=10.0, help="Training weight for y >= very-high-threshold rows")
    parser.add_argument("--high-count-threshold", type=float, default=4.0, help="Threshold for medium/high-demand metrics and weighting")
    parser.add_argument("--very-high-count-threshold", type=float, default=8.0, help="Threshold for highest training weight tier")
    parser.add_argument("--test-start-date", type=str, default=None, help="Date boundary for test split, e.g. '2015-06-24'. Overrides ratio-based split.")
    parser.add_argument("--weather-cache", type=Path, default=None, help="Cached weather CSV; fetches from API only when missing")
    return parser.parse_args()



def load_orders(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    spatial_col = resolve_spatial_column(df.columns)
    required_columns = ["pickup_datetime", spatial_col, "pickup_latitude", "pickup_longitude"]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise KeyError(f"Input file is missing required columns: {missing}")

    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
    df = df.dropna(subset=required_columns).copy()
    return normalize_h3_column(df, spatial_col)


def aggregate_to_panel(df: pd.DataFrame, freq: str, h3_resolution: int) -> pd.DataFrame:
    orders = df.copy()
    orders[SPATIAL_COL] = coarsen_h3_series(orders[SPATIAL_COL], h3_resolution)
    orders["time_slot"] = orders["pickup_datetime"].dt.floor(freq)

    counts = orders.groupby([SPATIAL_COL, "time_slot"]).size().reset_index(name="order_count")
    h3_features = counts[[SPATIAL_COL]].drop_duplicates().reset_index(drop=True)
    centers = h3_features[SPATIAL_COL].map(h3_center_latlng)
    h3_features["center_lat"] = centers.map(lambda value: value[0])
    h3_features["center_lon"] = centers.map(lambda value: value[1])

    h3_frame = h3_features[[SPATIAL_COL]].drop_duplicates().reset_index(drop=True)
    slot_range = pd.date_range(counts["time_slot"].min(), counts["time_slot"].max(), freq=freq)
    slot_frame = pd.DataFrame({"time_slot": slot_range})

    panel = h3_frame.merge(slot_frame, how="cross")
    panel = panel.merge(counts, on=[SPATIAL_COL, "time_slot"], how="left")
    panel = panel.merge(h3_features, on=SPATIAL_COL, how="left")
    panel["order_count"] = panel["order_count"].fillna(0.0).astype(float)
    return panel.sort_values([SPATIAL_COL, "time_slot"]).reset_index(drop=True)


def weather_code_to_flags(weather_code: int) -> dict[str, int]:
    return {
        "weather_clear": int(weather_code in {0, 1}),
        "weather_cloudy": int(weather_code in {2, 3}),
        "weather_fog": int(weather_code in {45, 48}),
        "weather_drizzle": int(weather_code in {51, 53, 55, 56, 57}),
        "weather_rain": int(weather_code in {61, 63, 65, 66, 67, 80, 81, 82}),
        "weather_snow": int(weather_code in {71, 73, 75, 77, 85, 86}),
        "weather_thunder": int(weather_code in {95, 96, 99}),
    }


def fetch_weather_features(start_time: pd.Timestamp, end_time: pd.Timestamp, time_slots: pd.Series) -> pd.DataFrame:
    query = urlencode(
        {
            "latitude": NYC_LATITUDE,
            "longitude": NYC_LONGITUDE,
            "start_date": start_time.strftime("%Y-%m-%d"),
            "end_date": end_time.strftime("%Y-%m-%d"),
            "hourly": "temperature_2m,apparent_temperature,precipitation,rain,cloud_cover,wind_speed_10m,relative_humidity_2m,weather_code",
            "timezone": "America/New_York",
        }
    )
    url = f"https://archive-api.open-meteo.com/v1/archive?{query}"

    try:
        with urlopen(url, timeout=60) as response:
            payload = json.load(response)
    except Exception as exc:
        print(f"Weather API unreachable ({exc}), using empty weather features")
        return _empty_weather(time_slots)

    hourly = pd.DataFrame(payload["hourly"])
    hourly["time_slot"] = pd.to_datetime(hourly["time"])
    hourly = hourly.drop(columns=["time"])
    hourly = hourly.rename(
        columns={
            "temperature_2m": "temperature",
            "apparent_temperature": "apparent_temperature",
            "precipitation": "precipitation",
            "rain": "rain",
            "cloud_cover": "cloud_cover",
            "wind_speed_10m": "wind_speed",
            "relative_humidity_2m": "humidity",
        }
    )

    weather_flags = hourly["weather_code"].fillna(-1).astype(int).map(weather_code_to_flags)
    weather_flag_df = pd.DataFrame(weather_flags.tolist())
    hourly = pd.concat([hourly, weather_flag_df], axis=1)
    hourly["is_precipitating"] = (hourly["precipitation"] > 0).astype(int)
    hourly["is_hot"] = (hourly["temperature"] >= 30).astype(int)
    hourly["is_humid"] = (hourly["humidity"] >= 80).astype(int)
    return hourly


def build_holiday_features(time_slots: pd.Series) -> pd.DataFrame:
    calendar = USFederalHolidayCalendar()
    start_date = time_slots.min().normalize()
    end_date = time_slots.max().normalize()
    holidays = set(calendar.holidays(start=start_date, end=end_date))

    frame = pd.DataFrame({"time_slot": pd.Series(time_slots.drop_duplicates().sort_values())})
    frame["date"] = frame["time_slot"].dt.normalize()
    frame["is_holiday"] = frame["date"].isin(holidays).astype(int)
    frame["is_month_start"] = frame["time_slot"].dt.is_month_start.astype(int)
    frame["is_month_end"] = frame["time_slot"].dt.is_month_end.astype(int)
    return frame.drop(columns=["date"])


def add_time_features(panel: pd.DataFrame) -> pd.DataFrame:
    features = panel.copy()
    features["hour"] = features["time_slot"].dt.hour.astype(int)
    features["weekday"] = features["time_slot"].dt.weekday.astype(int)
    features["day_of_month"] = features["time_slot"].dt.day.astype(int)
    features["week_of_month"] = (((features["day_of_month"] - 1) // 7) + 1).astype(int)
    features["weekend"] = features["weekday"].isin([5, 6]).astype(int)
    features["is_rush_hour"] = features["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)
    features["is_late_night"] = features["hour"].isin([0, 1, 2, 3, 4]).astype(int)
    features["hour_fraction"] = (features["hour"] + 0.5) / 24.0
    features["time_sin"] = np.sin(features["hour_fraction"] * 2 * np.pi)
    features["time_cos"] = np.cos(features["hour_fraction"] * 2 * np.pi)
    features["day_fraction"] = (features["weekday"] + features["hour_fraction"]) / 7.0
    features["day_sin"] = np.sin(features["day_fraction"] * 2 * np.pi)
    features["day_cos"] = np.cos(features["day_fraction"] * 2 * np.pi)
    return features


def add_history_features(panel: pd.DataFrame) -> pd.DataFrame:
    features = panel.copy()
    grouped = features.groupby(SPATIAL_COL)["order_count"]

    for lag in [1, 2, 3, 24, 48, 168]:
        features[f"lag_{lag}"] = grouped.shift(lag)

    for window in [3, 6, 12, 24, 48]:
        features[f"rolling_mean_{window}"] = grouped.transform(
            lambda values, w=window: values.shift(1).rolling(w).mean()
        )
        features[f"rolling_max_{window}"] = grouped.transform(
            lambda values, w=window: values.shift(1).rolling(w).max()
        )

    features["rolling_std_24"] = grouped.transform(lambda values: values.shift(1).rolling(24).std())
    features["rolling_std_48"] = grouped.transform(lambda values: values.shift(1).rolling(48).std())
    features[["rolling_std_24", "rolling_std_48"]] = features[["rolling_std_24", "rolling_std_48"]].fillna(0.0)
    return features


def add_spatial_intensity_features(panel: pd.DataFrame, parent_resolution: int) -> pd.DataFrame:
    features = panel.copy()
    features["h3_parent"] = coarsen_h3_series(features[SPATIAL_COL], parent_resolution)

    parent_panel = features.groupby(["h3_parent", "time_slot"]).agg(
        parent_total_count=("order_count", "sum"),
        parent_mean_count=("order_count", "mean"),
    ).reset_index()
    parent_panel = parent_panel.sort_values(["h3_parent", "time_slot"]).reset_index(drop=True)
    parent_group = parent_panel.groupby("h3_parent")

    parent_panel["parent_total_lag_1"] = parent_group["parent_total_count"].shift(1)
    parent_panel["parent_total_lag_24"] = parent_group["parent_total_count"].shift(24)
    parent_panel["parent_mean_lag_1"] = parent_group["parent_mean_count"].shift(1)
    parent_panel["parent_mean_rolling_24"] = parent_group["parent_mean_count"].transform(
        lambda values: values.shift(1).rolling(24).mean()
    )

    features = features.merge(
        parent_panel[
            [
                "h3_parent",
                "time_slot",
                "parent_total_lag_1",
                "parent_total_lag_24",
                "parent_mean_lag_1",
                "parent_mean_rolling_24",
            ]
        ],
        on=["h3_parent", "time_slot"],
        how="left",
    )
    return features


def add_target_encoding(train_df: pd.DataFrame, other_df: pd.DataFrame) -> pd.DataFrame:
    global_mean = float(train_df["order_count"].mean())
    mappings = {
        "h3_mean_count": train_df.groupby(SPATIAL_COL)["order_count"].mean(),
        "hour_mean_count": train_df.groupby("hour")["order_count"].mean(),
        "weekday_mean_count": train_df.groupby("weekday")["order_count"].mean(),
        "holiday_mean_count": train_df.groupby("is_holiday")["order_count"].mean(),
        "parent_mean_count_train": train_df.groupby("h3_parent")["order_count"].mean(),
    }

    encoded = other_df.copy()
    encoded["h3_mean_count"] = encoded[SPATIAL_COL].map(mappings["h3_mean_count"]).fillna(global_mean)
    encoded["parent_mean_count_train"] = encoded["h3_parent"].map(mappings["parent_mean_count_train"]).fillna(global_mean)
    encoded["hour_mean_count"] = encoded["hour"].map(mappings["hour_mean_count"]).fillna(global_mean)
    encoded["weekday_mean_count"] = encoded["weekday"].map(mappings["weekday_mean_count"]).fillna(global_mean)
    encoded["holiday_mean_count"] = encoded["is_holiday"].map(mappings["holiday_mean_count"]).fillna(global_mean)
    return encoded


def make_splits(panel: pd.DataFrame, train_ratio: float, valid_ratio: float, test_start_date: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    unique_slots = panel["time_slot"].drop_duplicates().sort_values().reset_index(drop=True)

    if test_start_date is not None:
        test_start = pd.Timestamp(test_start_date)
        train_valid_mask = unique_slots < test_start
        test_mask = unique_slots >= test_start
        train_valid_slots = unique_slots[train_valid_mask].tolist()
        test_slots = unique_slots[test_mask].tolist()
        n_train_valid = len(train_valid_slots)
        ratio_sum = train_ratio + valid_ratio
        if ratio_sum <= 0:
            raise ValueError("train_ratio + valid_ratio must be > 0 for date-based split")
        train_end = int(n_train_valid * (train_ratio / ratio_sum))
        if train_end <= 0 or train_end >= n_train_valid:
            raise ValueError("Date-based train/validation split leaves an empty partition; adjust ratios or data span")
        train_slots = train_valid_slots[:train_end]
        valid_slots = train_valid_slots[train_end:]
    else:
        total_slots = len(unique_slots)
        train_end = int(total_slots * train_ratio)
        valid_end = int(total_slots * (train_ratio + valid_ratio))
        if train_end <= 0 or valid_end <= train_end or valid_end >= total_slots:
            raise ValueError("Train/validation split leaves an empty partition; adjust ratios or data span")
        train_slots = unique_slots.iloc[:train_end].tolist()
        valid_slots = unique_slots.iloc[train_end:valid_end].tolist()
        test_slots = unique_slots.iloc[valid_end:].tolist()

    train_df = panel[panel["time_slot"].isin(train_slots)].copy()
    valid_df = panel[panel["time_slot"].isin(valid_slots)].copy()
    test_df = panel[panel["time_slot"].isin(test_slots)].copy()
    return train_df, valid_df, test_df


def rmse(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y_true.to_numpy() - y_pred))))


def mae(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true.to_numpy() - y_pred)))


def pcc(y_true: pd.Series, y_pred: np.ndarray) -> float:
    true = y_true.to_numpy(dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if true.size == 0 or np.std(true) == 0 or np.std(pred) == 0:
        return float("nan")
    return float(np.corrcoef(true, pred)[0, 1])


def within_tolerance(y_true: pd.Series, y_pred: np.ndarray, tolerance: float) -> float:
    true = y_true.to_numpy(dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if true.size == 0:
        return float("nan")
    return float((np.abs(true - pred) <= tolerance).mean())


def summarize_subset(y_true: pd.Series, y_pred: np.ndarray, threshold: float) -> dict[str, float | int]:
    mask = y_true.to_numpy(dtype=float) >= threshold
    if not mask.any():
        return {
            "rows": 0,
            "rmse": float("nan"),
            "mae": float("nan"),
            "pcc": float("nan"),
        }
    subset_true = y_true.loc[mask]
    subset_pred = y_pred[mask]
    return {
        "rows": int(mask.sum()),
        "rmse": rmse(subset_true, subset_pred),
        "mae": mae(subset_true, subset_pred),
        "pcc": pcc(subset_true, subset_pred),
    }


def compute_row_weights(
    df: pd.DataFrame,
    zero_weight: float,
    nonzero_weight: float,
    high_weight: float,
    very_high_weight: float,
    high_count_threshold: float,
    very_high_count_threshold: float,
) -> np.ndarray:
    counts = df["order_count"].to_numpy()
    weights = np.full(len(df), zero_weight, dtype=float)
    weights[counts > 0] = nonzero_weight
    weights[counts >= high_count_threshold] = high_weight
    weights[counts >= very_high_count_threshold] = very_high_weight
    return weights


def prepare_model_matrix(df: pd.DataFrame, feature_columns: list[str], weights: np.ndarray | None = None) -> xgb.DMatrix:
    matrix = df[feature_columns].astype(float)
    labels = df["order_count"].astype(float)
    return xgb.DMatrix(matrix, label=labels, weight=weights, feature_names=feature_columns)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading engineered data from {args.input}...")
    orders = load_orders(args.input)
    print(f"Aggregating orders into {args.freq} demand panels...")
    panel = aggregate_to_panel(orders, args.freq, args.h3_resolution)

    start_time = pd.Timestamp(panel["time_slot"].min())
    end_time = pd.Timestamp(panel["time_slot"].max())
    if args.weather_cache and args.weather_cache.exists():
        print(f"Loading cached weather from {args.weather_cache}...")
        weather = pd.read_csv(args.weather_cache)
        weather["time_slot"] = pd.to_datetime(weather["time_slot"])
    else:
        print("Fetching historical weather features...")
        weather = fetch_weather_features(start_time, end_time, panel["time_slot"])
        if args.weather_cache:
            args.weather_cache.parent.mkdir(parents=True, exist_ok=True)
            weather.to_csv(args.weather_cache, index=False)
    holidays = build_holiday_features(panel["time_slot"])

    panel = add_time_features(panel)
    panel = add_history_features(panel)
    panel = add_spatial_intensity_features(panel, args.parent_resolution)
    panel = panel.merge(weather, on="time_slot", how="left")
    panel = panel.merge(holidays, on="time_slot", how="left")

    weather_columns = [
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
    panel[weather_columns] = panel[weather_columns].fillna(0.0)
    panel[["is_holiday", "is_month_start", "is_month_end"]] = panel[["is_holiday", "is_month_start", "is_month_end"]].fillna(0).astype(int)

    required_history = [
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
    panel = panel.dropna(subset=required_history).reset_index(drop=True)

    train_df, valid_df, test_df = make_splits(panel, args.train_ratio, args.valid_ratio, args.test_start_date)
    train_df = add_target_encoding(train_df, train_df)
    valid_df = add_target_encoding(train_df, valid_df)
    test_df = add_target_encoding(train_df, test_df)

    train_weights = compute_row_weights(
        train_df,
        args.weight_zero,
        args.weight_nonzero,
        args.weight_high,
        args.weight_very_high,
        args.high_count_threshold,
        args.very_high_count_threshold,
    )

    feature_columns = [
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

    train_matrix = prepare_model_matrix(train_df, feature_columns, train_weights)
    valid_matrix = prepare_model_matrix(valid_df, feature_columns)
    test_matrix = prepare_model_matrix(test_df, feature_columns)

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
        "seed": 42,
    }

    print("Training XGBoost v2 model...")
    evals_result: dict[str, dict[str, list[float]]] = {}
    booster = xgb.train(
        params=params,
        dtrain=train_matrix,
        num_boost_round=args.num_boost_round,
        evals=[(train_matrix, "train"), (valid_matrix, "valid")],
        early_stopping_rounds=args.early_stopping_rounds,
        evals_result=evals_result,
        verbose_eval=20,
    )

    valid_pred = booster.predict(valid_matrix)
    test_pred = booster.predict(test_matrix)
    valid_pred = np.clip(valid_pred, 0.0, None)
    test_pred = np.clip(test_pred, 0.0, None)
    valid_nonzero = summarize_subset(valid_df["order_count"], valid_pred, 1.0)
    test_nonzero = summarize_subset(test_df["order_count"], test_pred, 1.0)
    valid_high = summarize_subset(valid_df["order_count"], valid_pred, args.high_count_threshold)
    test_high = summarize_subset(test_df["order_count"], test_pred, args.high_count_threshold)
    valid_pcc = pcc(valid_df["order_count"], valid_pred)
    test_pcc = pcc(test_df["order_count"], test_pred)
    valid_nonzero_mask = valid_df["order_count"].to_numpy() > 0
    test_nonzero_mask = test_df["order_count"].to_numpy() > 0
    valid_nonzero_within_1 = within_tolerance(valid_df.loc[valid_nonzero_mask, "order_count"], valid_pred[valid_nonzero_mask], 1.0)
    valid_nonzero_within_2 = within_tolerance(valid_df.loc[valid_nonzero_mask, "order_count"], valid_pred[valid_nonzero_mask], 2.0)
    test_nonzero_within_1 = within_tolerance(test_df.loc[test_nonzero_mask, "order_count"], test_pred[test_nonzero_mask], 1.0)
    test_nonzero_within_2 = within_tolerance(test_df.loc[test_nonzero_mask, "order_count"], test_pred[test_nonzero_mask], 2.0)
    metrics = {
        "version": "v2",
        "freq": args.freq,
        "objective": "reg:squarederror",
        "h3_resolution": args.h3_resolution,
        "parent_resolution": args.parent_resolution,
        "train_rows": len(train_df),
        "valid_rows": len(valid_df),
        "test_rows": len(test_df),
        "train_timeslots": int(train_df["time_slot"].nunique()),
        "valid_timeslots": int(valid_df["time_slot"].nunique()),
        "test_timeslots": int(test_df["time_slot"].nunique()),
        "best_iteration": int(booster.best_iteration),
        "valid_rmse": rmse(valid_df["order_count"], valid_pred),
        "valid_mae": mae(valid_df["order_count"], valid_pred),
        "test_rmse": rmse(test_df["order_count"], test_pred),
        "test_mae": mae(test_df["order_count"], test_pred),
        "feature_count": len(feature_columns),
        "train_zero_ratio": float((train_df["order_count"] == 0).mean()),
        "valid_zero_ratio": float((valid_df["order_count"] == 0).mean()),
        "test_zero_ratio": float((test_df["order_count"] == 0).mean()),
        "weight_zero": args.weight_zero,
        "weight_nonzero": args.weight_nonzero,
        "weight_high": args.weight_high,
        "weight_very_high": args.weight_very_high,
        "high_count_threshold": args.high_count_threshold,
        "very_high_count_threshold": args.very_high_count_threshold,
        "valid_pcc": valid_pcc,
        "test_pcc": test_pcc,
        "valid_nonzero_rows": valid_nonzero["rows"],
        "valid_nonzero_rmse": valid_nonzero["rmse"],
        "valid_nonzero_mae": valid_nonzero["mae"],
        "valid_nonzero_pcc": valid_nonzero["pcc"],
        "valid_nonzero_within_1": valid_nonzero_within_1,
        "valid_nonzero_within_2": valid_nonzero_within_2,
        "test_nonzero_rows": test_nonzero["rows"],
        "test_nonzero_rmse": test_nonzero["rmse"],
        "test_nonzero_mae": test_nonzero["mae"],
        "test_nonzero_pcc": test_nonzero["pcc"],
        "test_nonzero_within_1": test_nonzero_within_1,
        "test_nonzero_within_2": test_nonzero_within_2,
        "valid_high_rows": valid_high["rows"],
        "valid_high_rmse": valid_high["rmse"],
        "valid_high_mae": valid_high["mae"],
        "valid_pcc_y_ge_4": valid_high["pcc"],
        "test_high_rows": test_high["rows"],
        "test_high_rmse": test_high["rmse"],
        "test_high_mae": test_high["mae"],
        "test_pcc_y_ge_4": test_high["pcc"],
    }

    importance_pairs = list(booster.get_score(importance_type="gain").items())
    importance = pd.DataFrame(importance_pairs, columns=["feature", "gain"]).sort_values("gain", ascending=False)
    predictions = test_df[[SPATIAL_COL, "time_slot", "order_count"]].copy()
    predictions["prediction"] = test_pred

    weather.to_csv(output_dir / "weather_features.csv", index=False)
    booster.save_model(output_dir / "model.json")
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=True, indent=2)
    with open(output_dir / "evals_result.json", "w", encoding="utf-8") as file:
        json.dump(evals_result, file, ensure_ascii=True, indent=2)
    importance.to_csv(output_dir / "feature_importance.csv", index=False)
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)

    print("Training complete.")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("Top feature importance:")
    print(importance.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
