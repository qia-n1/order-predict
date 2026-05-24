import os
import argparse
import json
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import xgboost as xgb

from h3_utils import SPATIAL_COL

from train_xgboost_v2 import (
    add_history_features,
    add_spatial_intensity_features,
    add_target_encoding,
    add_time_features,
    aggregate_to_panel,
    build_holiday_features,
    fetch_weather_features,
    load_orders,
    make_splits,
)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Train Version 3: deep embedding + XGBoost")
    parser.add_argument(
        "--input",
        type=Path,
        default=project_root / "data" / "NYC_YellowTaxi_2015_100k_features.csv",
        help="Engineered order-level CSV path",
    )
    parser.add_argument("--freq", type=str, default="1h", help="Aggregation frequency")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "models" / "xgboost_hourly_v3",
        help="Directory for model and reports",
    )
    parser.add_argument(
        "--weather-cache",
        type=Path,
        default=project_root / "models" / "xgboost_hourly_v2_h3" / "weather_features.csv",
        help="Cached hourly weather feature CSV; fetches from API only when missing",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Time-based train split ratio")
    parser.add_argument("--valid-ratio", type=float, default=0.15, help="Time-based validation split ratio")
    parser.add_argument("--h3-resolution", type=int, default=8, help="Target H3 resolution for panel aggregation")
    parser.add_argument("--parent-resolution", type=int, default=7, help="Coarser H3 parent resolution for context features")
    parser.add_argument("--window-len", type=int, default=24, help="Temporal lookback window")
    parser.add_argument("--patch-size", type=int, default=5, help="Spatial neighborhood patch size")
    parser.add_argument("--embedding-dim", type=int, default=32, help="Fusion embedding size")
    parser.add_argument("--stage1-hidden-dim", type=int, default=32, help="GRU hidden size")
    parser.add_argument("--batch-size", type=int, default=256, help="Stage1 batch size")
    parser.add_argument("--max-epochs", type=int, default=80, help="Maximum Stage1 epochs")
    parser.add_argument("--patience", type=int, default=12, help="Stage1 early stopping patience")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Stage1 learning rate")
    parser.add_argument("--num-boost-round", type=int, default=600, help="Stage2 XGBoost max boosting rounds")
    parser.add_argument("--early-stopping-rounds", type=int, default=50, help="Stage2 early stopping rounds")
    parser.add_argument("--weight-zero", type=float, default=1.0, help="Sample weight for zero-demand rows")
    parser.add_argument("--weight-nonzero", type=float, default=3.0, help="Sample weight for 0 < y < 4 rows")
    parser.add_argument("--weight-high", type=float, default=6.0, help="Sample weight for 4 <= y < 8 rows")
    parser.add_argument("--weight-very-high", type=float, default=10.0, help="Sample weight for y >= 8 rows")
    parser.add_argument("--high-count-threshold", type=float, default=4.0, help="High-count threshold for metrics")
    parser.add_argument("--very-high-count-threshold", type=float, default=8.0, help="Very-high-count threshold for weighting")
    parser.add_argument("--reuse-stage1-artifacts", action="store_true", help="Reuse saved stage-1 model and embeddings when available")
    parser.add_argument("--test-start-date", type=str, default=None, help="Date boundary for test split, e.g. '2015-06-24'. Overrides ratio-based split.")
    return parser.parse_args()


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y_true - y_pred))))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def pcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def within_tolerance(y_true: np.ndarray, y_pred: np.ndarray, tolerance: float) -> float:
    if y_true.size == 0:
        return float("nan")
    return float((np.abs(y_true - y_pred) <= tolerance).mean())


def summarize_subset(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> dict[str, float | int]:
    mask = y_true >= threshold
    if not mask.any():
        return {"rows": 0, "rmse": float("nan"), "mae": float("nan"), "pcc": float("nan")}
    return {
        "rows": int(mask.sum()),
        "rmse": rmse(y_true[mask], y_pred[mask]),
        "mae": mae(y_true[mask], y_pred[mask]),
        "pcc": pcc(y_true[mask], y_pred[mask]),
    }


def compute_row_weights(
    counts: np.ndarray,
    zero_weight: float,
    nonzero_weight: float,
    high_weight: float,
    very_high_weight: float,
    high_count_threshold: float,
    very_high_count_threshold: float,
) -> np.ndarray:
    weights = np.full(len(counts), zero_weight, dtype=np.float32)
    weights[counts > 0] = nonzero_weight
    weights[counts >= high_count_threshold] = high_weight
    weights[counts >= very_high_count_threshold] = very_high_weight
    return weights


class PanelSequenceDataset(Dataset):
    def __init__(
        self,
        rows: pd.DataFrame,
        count_matrix: np.ndarray,
        spatial_to_idx: dict[str, int],
        time_to_idx: dict[pd.Timestamp, int],
        neighbors: dict[str, np.ndarray],
        ext_columns: list[str],
        ext_mean: np.ndarray,
        ext_std: np.ndarray,
        window_len: int,
        patch_size: int,
        weight_args: argparse.Namespace,
    ) -> None:
        self.rows = rows.reset_index(drop=True).copy()
        self.count_matrix = count_matrix
        self.spatial_to_idx = spatial_to_idx
        self.time_to_idx = time_to_idx
        self.neighbors = neighbors
        self.ext_columns = ext_columns
        self.ext_mean = ext_mean.astype(np.float32)
        self.ext_std = np.where(ext_std == 0, 1.0, ext_std).astype(np.float32)
        self.window_len = window_len
        self.patch_size = patch_size
        self.weights = compute_row_weights(
            self.rows["order_count"].to_numpy(dtype=np.float32),
            weight_args.weight_zero,
            weight_args.weight_nonzero,
            weight_args.weight_high,
            weight_args.weight_very_high,
            weight_args.high_count_threshold,
            weight_args.very_high_count_threshold,
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        row = self.rows.iloc[index]
        spatial_id = str(row[SPATIAL_COL])
        time_slot = pd.Timestamp(row["time_slot"])
        g_idx = self.spatial_to_idx[spatial_id]
        t_idx = self.time_to_idx[time_slot]
        seq = self.count_matrix[t_idx - self.window_len : t_idx, g_idx]
        spatial = self.count_matrix[t_idx - 1, self.neighbors[spatial_id]].reshape(self.patch_size, self.patch_size)
        external = row[self.ext_columns].to_numpy(dtype=np.float32)
        external = (external - self.ext_mean) / self.ext_std
        return {
            "spatial": torch.tensor(np.log1p(spatial), dtype=torch.float32).unsqueeze(0),
            "temporal": torch.tensor(np.log1p(seq), dtype=torch.float32).unsqueeze(-1),
            "external": torch.tensor(external, dtype=torch.float32),
            "target": torch.tensor(float(row["order_count"]), dtype=torch.float32),
            "weight": torch.tensor(float(self.weights[index]), dtype=torch.float32),
            SPATIAL_COL: spatial_id,
            "time_slot": time_slot.value,
        }


class V3DeepModel(nn.Module):
    def __init__(self, patch_size: int, ext_dim: int, hidden_dim: int, embedding_dim: int) -> None:
        super().__init__()
        self.spatial_cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * patch_size * patch_size, embedding_dim),
            nn.ReLU(),
        )
        self.temporal_gru = nn.GRU(input_size=1, hidden_size=hidden_dim, batch_first=True)
        self.temporal_head = nn.Sequential(nn.Linear(hidden_dim, embedding_dim), nn.ReLU())
        self.external_mlp = nn.Sequential(
            nn.Linear(ext_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
        )
        self.attention = nn.Linear(embedding_dim, 1)
        self.prediction_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, 1),
        )

    def encode(self, spatial: torch.Tensor, temporal: torch.Tensor, external: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        spatial_embedding = self.spatial_cnn(spatial)
        _, hidden = self.temporal_gru(temporal)
        temporal_embedding = self.temporal_head(hidden[-1])
        external_embedding = self.external_mlp(external)
        stacked = torch.stack([spatial_embedding, temporal_embedding, external_embedding], dim=1)
        weights = torch.softmax(self.attention(stacked).squeeze(-1), dim=1)
        fused = torch.sum(stacked * weights.unsqueeze(-1), dim=1)
        return fused, weights

    def forward(self, spatial: torch.Tensor, temporal: torch.Tensor, external: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        fused, weights = self.encode(spatial, temporal, external)
        prediction = self.prediction_head(fused).squeeze(-1)
        return prediction, fused


def weighted_mse(prediction: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.mean(weight * torch.square(prediction - target))


def run_stage1_epoch(model: V3DeepModel, loader: DataLoader, device: torch.device, optimizer: torch.optim.Optimizer | None) -> tuple[float, float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    losses = []
    predictions = []
    targets = []

    for batch in loader:
        spatial = batch["spatial"].to(device)
        temporal = batch["temporal"].to(device)
        external = batch["external"].to(device)
        target = batch["target"].to(device)
        weight = batch["weight"].to(device)

        with torch.set_grad_enabled(is_train):
            prediction, _ = model(spatial, temporal, external)
            loss = weighted_mse(prediction, target, weight)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        losses.append(loss.detach().cpu().item())
        predictions.append(torch.clamp(prediction.detach().cpu(), min=0.0).numpy())
        targets.append(target.detach().cpu().numpy())

    y_true = np.concatenate(targets)
    y_pred = np.concatenate(predictions)
    return float(np.mean(losses)), mae(y_true, y_pred), rmse(y_true, y_pred)


def extract_embeddings(
    model: V3DeepModel,
    loader: DataLoader,
    device: torch.device,
    embedding_dim: int,
) -> pd.DataFrame:
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            spatial = batch["spatial"].to(device)
            temporal = batch["temporal"].to(device)
            external = batch["external"].to(device)
            _, embeddings = model(spatial, temporal, external)
            embeddings_np = embeddings.cpu().numpy()
            for idx, vector in enumerate(embeddings_np):
                row = {
                    SPATIAL_COL: batch[SPATIAL_COL][idx],
                    "time_slot": pd.Timestamp(batch["time_slot"][idx].item()),
                }
                for dim in range(embedding_dim):
                    row[f"emb_{dim}"] = float(vector[dim])
                rows.append(row)
    return pd.DataFrame(rows)


def build_neighbor_lookup(geo_frame: pd.DataFrame, patch_size: int) -> dict[str, np.ndarray]:
    coords = geo_frame[["center_lat", "center_lon"]].to_numpy(dtype=float)
    geohashes = geo_frame[SPATIAL_COL].tolist()
    distance = np.square(coords[:, None, :] - coords[None, :, :]).sum(axis=2)
    neighbor_count = patch_size * patch_size
    lookup: dict[str, np.ndarray] = {}
    for idx, geohash in enumerate(geohashes):
        nearest = np.argsort(distance[idx])[:neighbor_count]
        if len(nearest) < neighbor_count:
            nearest = np.pad(nearest, (0, neighbor_count - len(nearest)), constant_values=idx)
        lookup[geohash] = nearest.astype(int)
    return lookup


def prepare_model_matrix(df: pd.DataFrame, feature_columns: list[str], weights: np.ndarray | None = None) -> xgb.DMatrix:
    matrix = df[feature_columns].to_numpy(dtype=np.float32, copy=True)
    labels = df["order_count"].to_numpy(dtype=np.float32, copy=True)
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float32)
    return xgb.DMatrix(matrix, label=labels, weight=weights, feature_names=feature_columns, nthread=1)


def collect_metrics(df: pd.DataFrame, pred: np.ndarray, high_count_threshold: float) -> dict[str, float | int]:
    y_true = df["order_count"].to_numpy(dtype=float)
    y_pred = np.clip(pred, 0.0, None)
    nonzero_mask = y_true > 0
    high_summary = summarize_subset(y_true, y_pred, high_count_threshold)
    nonzero_summary = summarize_subset(y_true, y_pred, 1.0)
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "pcc": pcc(y_true, y_pred),
        "nonzero_rows": int(nonzero_mask.sum()),
        "nonzero_rmse": nonzero_summary["rmse"],
        "nonzero_mae": nonzero_summary["mae"],
        "nonzero_pcc": nonzero_summary["pcc"],
        "nonzero_within_1": within_tolerance(y_true[nonzero_mask], y_pred[nonzero_mask], 1.0),
        "nonzero_within_2": within_tolerance(y_true[nonzero_mask], y_pred[nonzero_mask], 2.0),
        "high_rows": high_summary["rows"],
        "high_rmse": high_summary["rmse"],
        "high_mae": high_summary["mae"],
        "pcc_y_ge_4": high_summary["pcc"],
    }


def load_or_fetch_weather_features(cache_path: Path, start_time: pd.Timestamp, end_time: pd.Timestamp, time_slots: pd.Series) -> pd.DataFrame:
    if cache_path.exists():
        weather = pd.read_csv(cache_path)
        weather["time_slot"] = pd.to_datetime(weather["time_slot"])
        return weather
    weather = fetch_weather_features(start_time, end_time, time_slots)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    weather.to_csv(cache_path, index=False)
    return weather


def read_embeddings(path: Path, embedding_dim: int) -> pd.DataFrame:
    dtype = {f"emb_{idx}": "float32" for idx in range(embedding_dim)}
    embeddings = pd.read_csv(path, dtype=dtype)
    embeddings["time_slot"] = pd.to_datetime(embeddings["time_slot"])
    return embeddings


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
    weather = load_or_fetch_weather_features(args.weather_cache, start_time, end_time, panel["time_slot"])
    holidays = build_holiday_features(panel["time_slot"])

    panel = add_time_features(panel)
    panel = add_history_features(panel)
    panel = add_spatial_intensity_features(panel, args.parent_resolution)
    panel = panel.merge(weather, on="time_slot", how="left")
    panel = panel.merge(holidays, on="time_slot", how="left")

    weather_columns = [
        "temperature", "apparent_temperature", "precipitation", "rain", "cloud_cover", "wind_speed",
        "humidity", "weather_code", "weather_clear", "weather_cloudy", "weather_fog", "weather_drizzle",
        "weather_rain", "weather_snow", "weather_thunder", "is_precipitating", "is_hot", "is_humid",
    ]
    panel[weather_columns] = panel[weather_columns].fillna(0.0)
    panel[["is_holiday", "is_month_start", "is_month_end"]] = panel[["is_holiday", "is_month_start", "is_month_end"]].fillna(0).astype(int)
    sequence_panel = panel.copy()

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

    count_pivot = sequence_panel.pivot(index="time_slot", columns=SPATIAL_COL, values="order_count").sort_index().sort_index(axis=1)
    time_slots = count_pivot.index.tolist()
    geohashes = count_pivot.columns.tolist()
    count_matrix = count_pivot.to_numpy(dtype=np.float32)
    spatial_to_idx = {spatial_id: idx for idx, spatial_id in enumerate(geohashes)}
    time_to_idx = {pd.Timestamp(slot): idx for idx, slot in enumerate(time_slots)}
    train_df = train_df[train_df["time_slot"].map(time_to_idx) >= args.window_len].reset_index(drop=True)
    valid_df = valid_df[valid_df["time_slot"].map(time_to_idx) >= args.window_len].reset_index(drop=True)
    test_df = test_df[test_df["time_slot"].map(time_to_idx) >= args.window_len].reset_index(drop=True)
    geo_frame = sequence_panel[[SPATIAL_COL, "center_lat", "center_lon"]].drop_duplicates().sort_values(SPATIAL_COL).reset_index(drop=True)
    geo_frame = geo_frame.set_index(SPATIAL_COL).reindex(geohashes).reset_index()
    neighbors = build_neighbor_lookup(geo_frame, args.patch_size)

    external_columns = [
        "hour", "weekday", "weekend", "is_rush_hour", "is_late_night", "time_sin", "time_cos", "day_sin",
        "day_cos", "day_of_month", "week_of_month", "temperature", "apparent_temperature", "precipitation",
        "rain", "cloud_cover", "wind_speed", "humidity", "weather_clear", "weather_cloudy", "weather_rain",
        "weather_snow", "weather_thunder", "is_precipitating", "is_hot", "is_humid", "is_holiday",
        "is_month_start", "is_month_end", "center_lat", "center_lon",
    ]
    ext_mean = train_df[external_columns].to_numpy(dtype=np.float32).mean(axis=0)
    ext_std = train_df[external_columns].to_numpy(dtype=np.float32).std(axis=0)

    train_ds = PanelSequenceDataset(train_df, count_matrix, spatial_to_idx, time_to_idx, neighbors, external_columns, ext_mean, ext_std, args.window_len, args.patch_size, args)
    valid_ds = PanelSequenceDataset(valid_df, count_matrix, spatial_to_idx, time_to_idx, neighbors, external_columns, ext_mean, ext_std, args.window_len, args.patch_size, args)
    test_ds = PanelSequenceDataset(test_df, count_matrix, spatial_to_idx, time_to_idx, neighbors, external_columns, ext_mean, ext_std, args.window_len, args.patch_size, args)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = V3DeepModel(args.patch_size, len(external_columns), args.stage1_hidden_dim, args.embedding_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    stage1_model_path = output_dir / "stage1_model.pt"
    train_embedding_path = output_dir / "train_embeddings.csv"
    valid_embedding_path = output_dir / "valid_embeddings.csv"
    test_embedding_path = output_dir / "test_embeddings.csv"

    stage1_history: list[dict[str, float | int]] = []
    best_valid_mae = float("inf")
    best_epoch = -1

    if args.reuse_stage1_artifacts and stage1_model_path.exists() and train_embedding_path.exists() and valid_embedding_path.exists() and test_embedding_path.exists():
        print("Reusing existing stage-1 artifacts...", flush=True)
        best_state = torch.load(stage1_model_path, map_location=device)
        model.load_state_dict(best_state)
        train_embeddings = read_embeddings(train_embedding_path, args.embedding_dim)
        valid_embeddings = read_embeddings(valid_embedding_path, args.embedding_dim)
        test_embeddings = read_embeddings(test_embedding_path, args.embedding_dim)
    else:
        best_state = None
        patience_left = args.patience
        print("Training stage-1 CNN/GRU/Attention model...")
        for epoch in range(1, args.max_epochs + 1):
            train_loss, train_mae, train_rmse = run_stage1_epoch(model, train_loader, device, optimizer)
            valid_loss, valid_mae, valid_rmse = run_stage1_epoch(model, valid_loader, device, optimizer=None)
            stage1_history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_mae": train_mae,
                "train_rmse": train_rmse,
                "valid_loss": valid_loss,
                "valid_mae": valid_mae,
                "valid_rmse": valid_rmse,
            })
            print(f"Epoch {epoch:03d} | train_mae={train_mae:.4f} valid_mae={valid_mae:.4f} valid_rmse={valid_rmse:.4f}")
            if valid_mae < best_valid_mae:
                best_valid_mae = valid_mae
                best_epoch = epoch
                patience_left = args.patience
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        if best_state is None:
            raise RuntimeError("Stage-1 model did not train successfully")
        model.load_state_dict(best_state)
        torch.save(best_state, stage1_model_path)

        print("Extracting embeddings...")
        train_embeddings = extract_embeddings(model, DataLoader(train_ds, batch_size=args.batch_size, shuffle=False), device, args.embedding_dim)
        valid_embeddings = extract_embeddings(model, valid_loader, device, args.embedding_dim)
        test_embeddings = extract_embeddings(model, test_loader, device, args.embedding_dim)
        train_embeddings.to_csv(train_embedding_path, index=False)
        valid_embeddings.to_csv(valid_embedding_path, index=False)
        test_embeddings.to_csv(test_embedding_path, index=False)

    print("Merging embeddings with v2 feature tables...", flush=True)
    train_stage2 = train_df.merge(train_embeddings, on=[SPATIAL_COL, "time_slot"], how="left")
    valid_stage2 = valid_df.merge(valid_embeddings, on=[SPATIAL_COL, "time_slot"], how="left")
    test_stage2 = test_df.merge(test_embeddings, on=[SPATIAL_COL, "time_slot"], how="left")
    train_stage2 = add_target_encoding(train_stage2, train_stage2)
    valid_stage2 = add_target_encoding(train_stage2, valid_stage2)
    test_stage2 = add_target_encoding(train_stage2, test_stage2)

    embedding_columns = [f"emb_{idx}" for idx in range(args.embedding_dim)]
    v2_feature_columns = [
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
    feature_columns = v2_feature_columns + embedding_columns

    train_weights = compute_row_weights(
        train_stage2["order_count"].to_numpy(dtype=np.float32),
        args.weight_zero,
        args.weight_nonzero,
        args.weight_high,
        args.weight_very_high,
        args.high_count_threshold,
        args.very_high_count_threshold,
    )
    print(f"Preparing XGBoost matrices with {len(feature_columns)} features...", flush=True)
    train_matrix = prepare_model_matrix(train_stage2, feature_columns, train_weights)
    valid_matrix = prepare_model_matrix(valid_stage2, feature_columns)
    test_matrix = prepare_model_matrix(test_stage2, feature_columns)

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

    print("Training stage-2 XGBoost on embeddings...")
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
    valid_metrics = collect_metrics(valid_stage2, valid_pred, args.high_count_threshold)
    test_metrics = collect_metrics(test_stage2, test_pred, args.high_count_threshold)

    metrics = {
        "version": "v3",
        "freq": args.freq,
        "objective": "reg:squarederror",
        "stage1_model": "cnn_gru_attention",
        "stage2_features": "v2_features_plus_embeddings",
        "embedding_dim": args.embedding_dim,
        "window_len": args.window_len,
        "patch_size": args.patch_size,
        "best_stage1_epoch": best_epoch,
        "best_stage1_valid_mae": best_valid_mae,
        "best_iteration": int(booster.best_iteration),
        "train_rows": len(train_stage2),
        "valid_rows": len(valid_stage2),
        "test_rows": len(test_stage2),
        "feature_count": len(feature_columns),
        "v2_feature_count": len(v2_feature_columns),
        "valid_rmse": valid_metrics["rmse"],
        "valid_mae": valid_metrics["mae"],
        "valid_pcc": valid_metrics["pcc"],
        "test_rmse": test_metrics["rmse"],
        "test_mae": test_metrics["mae"],
        "test_pcc": test_metrics["pcc"],
        "valid_nonzero_rows": valid_metrics["nonzero_rows"],
        "valid_nonzero_rmse": valid_metrics["nonzero_rmse"],
        "valid_nonzero_mae": valid_metrics["nonzero_mae"],
        "valid_nonzero_pcc": valid_metrics["nonzero_pcc"],
        "valid_nonzero_within_1": valid_metrics["nonzero_within_1"],
        "valid_nonzero_within_2": valid_metrics["nonzero_within_2"],
        "test_nonzero_rows": test_metrics["nonzero_rows"],
        "test_nonzero_rmse": test_metrics["nonzero_rmse"],
        "test_nonzero_mae": test_metrics["nonzero_mae"],
        "test_nonzero_pcc": test_metrics["nonzero_pcc"],
        "test_nonzero_within_1": test_metrics["nonzero_within_1"],
        "test_nonzero_within_2": test_metrics["nonzero_within_2"],
        "valid_high_rows": valid_metrics["high_rows"],
        "valid_high_rmse": valid_metrics["high_rmse"],
        "valid_high_mae": valid_metrics["high_mae"],
        "valid_pcc_y_ge_4": valid_metrics["pcc_y_ge_4"],
        "test_high_rows": test_metrics["high_rows"],
        "test_high_rmse": test_metrics["high_rmse"],
        "test_high_mae": test_metrics["high_mae"],
        "test_pcc_y_ge_4": test_metrics["pcc_y_ge_4"],
    }

    predictions = test_stage2[[SPATIAL_COL, "time_slot", "order_count"]].copy()
    predictions["prediction"] = np.clip(test_pred, 0.0, None)
    importance_pairs = list(booster.get_score(importance_type="gain").items())
    importance = pd.DataFrame(importance_pairs, columns=["feature", "gain"]).sort_values("gain", ascending=False)

    booster.save_model(output_dir / "model.json")
    weather.to_csv(output_dir / "weather_features.csv", index=False)
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    importance.to_csv(output_dir / "feature_importance.csv", index=False)
    pd.DataFrame(stage1_history).to_csv(output_dir / "stage1_history.csv", index=False)
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=True, indent=2)
    with open(output_dir / "evals_result.json", "w", encoding="utf-8") as file:
        json.dump(evals_result, file, ensure_ascii=True, indent=2)

    print("Training complete.")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("Top feature importance:")
    print(importance.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
