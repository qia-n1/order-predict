from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from h3_utils import SPATIAL_COL
from train_xgboost import (
    add_history_features,
    add_time_features,
    aggregate_to_panel,
    load_orders,
    mae,
    make_splits,
    pcc,
    rmse,
    summarize_subset,
    within_tolerance,
)


VS_DIR = ROOT / "vs"
DATA_PATH = ROOT / "data" / "Uber_2014_MayJun_features.csv"
PROJECT_MODEL_DIRS = {
    "v1": ROOT / "models" / "xgboost_hourly_v1_may_jun" / "metrics.json",
    "v2": ROOT / "models" / "xgboost_hourly_v2_may_jun" / "metrics.json",
    "v3": ROOT / "models" / "xgboost_hourly_v3_may_jun" / "metrics.json",
}

ARGS = SimpleNamespace(train_ratio=0.7, valid_ratio=0.15, test_start_date="2014-06-24")
FREQ = "1h"
H3_RESOLUTION = 8
HIGH_COUNT_THRESHOLD = 4.0

METRIC_COLUMNS = [
    "valid_rmse",
    "valid_mae",
    "valid_pcc",
    "valid_nonzero_rmse",
    "valid_nonzero_mae",
    "valid_nonzero_pcc",
    "valid_nonzero_within_1",
    "valid_nonzero_within_2",
    "valid_high_rmse",
    "valid_high_mae",
    "valid_pcc_y_ge_4",
    "test_rmse",
    "test_mae",
    "test_pcc",
    "test_nonzero_rmse",
    "test_nonzero_mae",
    "test_nonzero_pcc",
    "test_nonzero_within_1",
    "test_nonzero_within_2",
    "test_high_rmse",
    "test_high_mae",
    "test_pcc_y_ge_4",
]


def load_project_metrics() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model_name, path in PROJECT_MODEL_DIRS.items():
        with path.open("r", encoding="utf-8") as file:
            metrics = json.load(file)
        metrics["model"] = model_name
        metrics["model_family"] = "project"
        rows.append(metrics)
    return rows


def build_panel() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    orders = load_orders(DATA_PATH)
    panel = aggregate_to_panel(orders, FREQ, H3_RESOLUTION)
    panel = add_time_features(panel)
    panel = add_history_features(panel)
    panel = panel.dropna(
        subset=["lag_1", "lag_2", "lag_24", "lag_168", "rolling_mean_3", "rolling_mean_6", "rolling_mean_24"]
    ).reset_index(drop=True)
    return make_splits(panel, ARGS.train_ratio, ARGS.valid_ratio, ARGS.test_start_date)


def predict_historical_average(train_df: pd.DataFrame, target_df: pd.DataFrame) -> np.ndarray:
    global_mean = float(train_df["order_count"].mean())
    h3_hour_weekday = train_df.groupby([SPATIAL_COL, "weekday", "hour"])["order_count"].mean()
    h3_hour = train_df.groupby([SPATIAL_COL, "hour"])["order_count"].mean()
    hour_weekday = train_df.groupby(["weekday", "hour"])["order_count"].mean()

    keys_full = pd.MultiIndex.from_frame(target_df[[SPATIAL_COL, "weekday", "hour"]])
    keys_h3_hour = pd.MultiIndex.from_frame(target_df[[SPATIAL_COL, "hour"]])
    keys_hour_weekday = pd.MultiIndex.from_frame(target_df[["weekday", "hour"]])

    pred = pd.Series(h3_hour_weekday.reindex(keys_full).to_numpy(), index=target_df.index, dtype=float)
    pred = pred.fillna(pd.Series(h3_hour.reindex(keys_h3_hour).to_numpy(), index=target_df.index, dtype=float))
    pred = pred.fillna(pd.Series(hour_weekday.reindex(keys_hour_weekday).to_numpy(), index=target_df.index, dtype=float))
    return pred.fillna(global_mean).to_numpy(dtype=float)


def predict_seasonal_naive(target_df: pd.DataFrame) -> np.ndarray:
    pred = target_df["lag_168"].fillna(target_df["lag_24"]).fillna(target_df["lag_1"]).fillna(0.0)
    return pred.to_numpy(dtype=float)


def split_metrics(df: pd.DataFrame, pred: np.ndarray, prefix: str) -> dict[str, float | int]:
    pred = np.clip(np.asarray(pred, dtype=float), 0.0, None)
    nonzero = summarize_subset(df["order_count"], pred, 1.0)
    high = summarize_subset(df["order_count"], pred, HIGH_COUNT_THRESHOLD)
    nonzero_mask = df["order_count"].to_numpy(dtype=float) > 0
    return {
        f"{prefix}_rmse": rmse(df["order_count"], pred),
        f"{prefix}_mae": mae(df["order_count"], pred),
        f"{prefix}_pcc": pcc(df["order_count"], pred),
        f"{prefix}_nonzero_rows": nonzero["rows"],
        f"{prefix}_nonzero_rmse": nonzero["rmse"],
        f"{prefix}_nonzero_mae": nonzero["mae"],
        f"{prefix}_nonzero_pcc": nonzero["pcc"],
        f"{prefix}_nonzero_within_1": within_tolerance(df.loc[nonzero_mask, "order_count"], pred[nonzero_mask], 1.0),
        f"{prefix}_nonzero_within_2": within_tolerance(df.loc[nonzero_mask, "order_count"], pred[nonzero_mask], 2.0),
        f"{prefix}_high_rows": high["rows"],
        f"{prefix}_high_rmse": high["rmse"],
        f"{prefix}_high_mae": high["mae"],
        f"{prefix}_pcc_y_ge_4": high["pcc"],
    }


def save_predictions(model_dir: Path, df: pd.DataFrame, pred: np.ndarray, split: str) -> None:
    predictions = df[[SPATIAL_COL, "time_slot", "order_count"]].copy()
    predictions["prediction"] = np.clip(pred, 0.0, None)
    predictions.to_csv(model_dir / f"{split}_predictions.csv", index=False)


def evaluate_baseline(
    model_name: str,
    display_name: str,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    valid_pred: np.ndarray,
    test_pred: np.ndarray,
) -> dict[str, object]:
    model_dir = VS_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    save_predictions(model_dir, valid_df, valid_pred, "valid")
    save_predictions(model_dir, test_df, test_pred, "test")

    metrics: dict[str, object] = {
        "model": display_name,
        "model_family": "baseline",
        "freq": FREQ,
        "objective": "baseline_forecast",
        "h3_resolution": H3_RESOLUTION,
        "train_ratio": ARGS.train_ratio,
        "valid_ratio": ARGS.valid_ratio,
        "test_start_date": ARGS.test_start_date,
        "train_rows": len(train_df),
        "valid_rows": len(valid_df),
        "test_rows": len(test_df),
        "train_timeslots": int(train_df["time_slot"].nunique()),
        "valid_timeslots": int(valid_df["time_slot"].nunique()),
        "test_timeslots": int(test_df["time_slot"].nunique()),
        "train_zero_ratio": float((train_df["order_count"] == 0).mean()),
        "valid_zero_ratio": float((valid_df["order_count"] == 0).mean()),
        "test_zero_ratio": float((test_df["order_count"] == 0).mean()),
        "high_count_threshold": HIGH_COUNT_THRESHOLD,
    }
    metrics.update(split_metrics(valid_df, valid_pred, "valid"))
    metrics.update(split_metrics(test_df, test_pred, "test"))

    with (model_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=True, indent=2)
    return metrics


def build_summary(rows: list[dict[str, object]]) -> pd.DataFrame:
    summary = pd.DataFrame(rows)
    columns = ["model", "model_family", "objective", "train_rows", "valid_rows", "test_rows", *METRIC_COLUMNS]
    existing_columns = [column for column in columns if column in summary.columns]
    summary = summary[existing_columns]
    summary.to_csv(VS_DIR / "full_metrics_comparison.csv", index=False)
    return summary


def plot_metric_group(summary: pd.DataFrame, metric_names: list[str], output_path: Path, title: str) -> None:
    plot_df = summary.set_index("model")[metric_names]
    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    axes_flat = axes.ravel()
    colors = ["#7dd3fc", "#86efac", "#fbbf24", "#f472b6", "#c4b5fd"]
    for ax, metric in zip(axes_flat, metric_names):
        values = plot_df[metric].astype(float)
        ax.bar(values.index, values.values, color=colors[: len(values)])
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
        for idx, value in enumerate(values.values):
            ax.text(idx, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    for ax in axes_flat[len(metric_names) :]:
        ax.axis("off")
    fig.suptitle(title, fontsize=20, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_dashboard(summary: pd.DataFrame) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.unicode_minus": False})
    plot_metric_group(
        summary,
        ["test_rmse", "test_mae", "test_pcc", "test_nonzero_rmse", "test_nonzero_mae", "test_nonzero_pcc"],
        VS_DIR / "test_core_metrics_comparison.png",
        "Test Core Metrics: v1/v2/v3 vs Baselines",
    )
    plot_metric_group(
        summary,
        ["test_nonzero_within_1", "test_nonzero_within_2", "test_high_rmse", "test_high_mae", "test_pcc_y_ge_4"],
        VS_DIR / "test_full_metrics_comparison.png",
        "Test Full Metrics: Nonzero and High-Demand Subsets",
    )
    plot_metric_group(
        summary,
        ["valid_rmse", "valid_mae", "valid_pcc", "valid_nonzero_rmse", "valid_nonzero_mae", "valid_nonzero_pcc"],
        VS_DIR / "valid_core_metrics_comparison.png",
        "Validation Core Metrics: v1/v2/v3 vs Baselines",
    )
    plot_metric_group(
        summary,
        ["valid_nonzero_within_1", "valid_nonzero_within_2", "valid_high_rmse", "valid_high_mae", "valid_pcc_y_ge_4"],
        VS_DIR / "valid_full_metrics_comparison.png",
        "Validation Full Metrics: Nonzero and High-Demand Subsets",
    )


def main() -> None:
    VS_DIR.mkdir(parents=True, exist_ok=True)
    train_df, valid_df, test_df = build_panel()

    historical_valid = predict_historical_average(train_df, valid_df)
    historical_test = predict_historical_average(train_df, test_df)
    seasonal_valid = predict_seasonal_naive(valid_df)
    seasonal_test = predict_seasonal_naive(test_df)

    rows = load_project_metrics()
    rows.append(
        evaluate_baseline(
            "baseline_historical_average",
            "HistoricalAvg",
            train_df,
            valid_df,
            test_df,
            historical_valid,
            historical_test,
        )
    )
    rows.append(
        evaluate_baseline(
            "baseline_seasonal_naive",
            "SeasonalNaive",
            train_df,
            valid_df,
            test_df,
            seasonal_valid,
            seasonal_test,
        )
    )

    summary = build_summary(rows)
    plot_dashboard(summary)
    print(summary.to_string(index=False))
    print(f"Saved comparison files to {VS_DIR}")


if __name__ == "__main__":
    main()
