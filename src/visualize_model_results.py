from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import h3
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
PICTURE_DIR = MODELS_DIR / "picture"
OUTPUT_DIR = MODELS_DIR / "picture2"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as file_handle:
        return list(csv.DictReader(file_handle))


def style_figure() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.unicode_minus": False,
            "figure.facecolor": "#071427",
            "axes.facecolor": "#0b1f3a",
            "axes.edgecolor": "#5f7ea5",
            "axes.labelcolor": "#dfeeff",
            "xtick.color": "#c8daef",
            "ytick.color": "#c8daef",
            "text.color": "#e7f0ff",
            "font.size": 12,
            "axes.titleweight": "bold",
            "axes.grid": True,
            "grid.color": "#2a4268",
            "grid.alpha": 0.45,
            "grid.linestyle": "-",
        }
    )


def latlon_to_xy(
    lat: float,
    lon: float,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
) -> tuple[float, float]:
    x_span = lon_max - lon_min if not np.isclose(lon_max, lon_min) else 1.0
    y_span = lat_max - lat_min if not np.isclose(lat_max, lat_min) else 1.0
    x = (lon - lon_min) / x_span
    y = (lat - lat_min) / y_span
    return float(x), float(y)


def normalize_points(values: np.ndarray) -> np.ndarray:
    lower = float(values.min())
    upper = float(values.max())
    if np.isclose(lower, upper):
        return np.full_like(values, 0.5, dtype=float)
    return (values - lower) / (upper - lower)


def load_h3_cells(path: Path) -> list[dict[str, float | str]]:
    rows = read_csv(path)
    grouped: dict[str, dict[str, float]] = defaultdict(
        lambda: {"count": 0.0, "actual_sum": 0.0, "pred_sum": 0.0}
    )
    for row in rows:
        cell_id = row["h3_index"]
        grouped[cell_id]["count"] += 1.0
        grouped[cell_id]["actual_sum"] += float(row["order_count"])
        grouped[cell_id]["pred_sum"] += float(row["prediction"])

    cells: list[dict[str, float | str]] = []
    for cell_id, stats in grouped.items():
        lat, lon = h3.cell_to_latlng(cell_id)
        row_count = stats["count"]
        actual_mean = stats["actual_sum"] / row_count
        pred_mean = stats["pred_sum"] / row_count
        score = pred_mean * 0.65 + actual_mean * 0.35 + np.log1p(row_count) * 0.03
        cells.append(
            {
                "h3_index": cell_id,
                "lat": float(lat),
                "lon": float(lon),
                "count": row_count,
                "actual_mean": actual_mean,
                "pred_mean": pred_mean,
                "score": score,
            }
        )

    cells.sort(key=lambda item: float(item["score"]), reverse=True)
    return cells


def build_projected_cells(
    cells: list[dict[str, float | str]], max_cells: int = 160
) -> list[dict[str, object]]:
    selected = cells[:max_cells]
    lat_values: list[float] = []
    lon_values: list[float] = []
    for cell in selected:
        cell_id = str(cell["h3_index"])
        lat, lon = h3.cell_to_latlng(cell_id)
        lat_values.append(float(lat))
        lon_values.append(float(lon))
        for boundary_lat, boundary_lon in h3.cell_to_boundary(cell_id):
            lat_values.append(float(boundary_lat))
            lon_values.append(float(boundary_lon))

    lat_min = min(lat_values)
    lat_max = max(lat_values)
    lon_min = min(lon_values)
    lon_max = max(lon_values)

    projected: list[dict[str, object]] = []
    for cell in selected:
        cell_id = str(cell["h3_index"])
        lat, lon = h3.cell_to_latlng(cell_id)
        center_x, center_y = latlon_to_xy(
            float(lat), float(lon), lon_min, lon_max, lat_min, lat_max
        )
        boundary = [
            latlon_to_xy(
                float(boundary_lat),
                float(boundary_lon),
                lon_min,
                lon_max,
                lat_min,
                lat_max,
            )
            for boundary_lat, boundary_lon in h3.cell_to_boundary(cell_id)
        ]
        projected.append(
            {
                **cell,
                "center_x": center_x,
                "center_y": center_y,
                "boundary_xy": np.array(boundary, dtype=float),
            }
        )

    projected.sort(key=lambda item: float(item["score"]), reverse=True)
    return projected


def darken_color(color: tuple[float, float, float, float], factor: float = 0.62) -> tuple[float, float, float, float]:
    red, green, blue, alpha = color
    return red * factor, green * factor, blue * factor, alpha


def draw_map_background(ax: plt.Axes) -> None:
    ax.set_facecolor("#07111f")
    gradient = np.linspace(0, 1, 400)
    bg = np.outer(np.ones(400), gradient)
    cmap = LinearSegmentedColormap.from_list(
        "map_bg", ["#04101d", "#07192e", "#0b2745", "#10375f"]
    )
    ax.imshow(bg, extent=(0, 1, 0, 1), origin="lower", cmap=cmap, alpha=0.92, zorder=0)

    t = np.linspace(0, 1, 500)
    for idx in range(18):
        phase = idx / 18.0
        y = 0.06 + idx * 0.045 + 0.014 * np.sin(2 * np.pi * (t * (1.12 + idx * 0.01) + phase))
        ax.plot(t, y, color="#9db8dd", alpha=0.05, linewidth=1.0, zorder=1)
    for idx in range(14):
        phase = idx / 14.0
        x = 0.04 + idx * 0.067 + 0.014 * np.sin(2 * np.pi * (t * (1.08 + idx * 0.02) + phase))
        ax.plot(x, t, color="#8fb2db", alpha=0.045, linewidth=1.0, zorder=1)

    labels = [
        (0.09, 0.87, "North"),
        (0.30, 0.95, "Shifang"),
        (0.52, 0.89, "Downtown"),
        (0.78, 0.83, "Queens"),
        (0.10, 0.25, "Westside"),
        (0.46, 0.13, "Central"),
        (0.83, 0.26, "Brooklyn"),
    ]
    for x, y, label in labels:
        ax.text(x, y, label, color="#a9c4e6", fontsize=12, alpha=0.55, zorder=2)


def draw_prism_hex_map(ax: plt.Axes, projected_cells: list[dict[str, object]]) -> None:
    scores = np.array([float(cell["pred_mean"]) for cell in projected_cells], dtype=float)
    score_norm = normalize_points(scores)
    hex_cmap = LinearSegmentedColormap.from_list(
        "hex_demand", ["#dfeaff", "#9ac5ff", "#4f9ef0", "#236cbf", "#0b54b0"]
    )

    order = np.argsort(scores)
    for idx in order:
        cell = projected_cells[int(idx)]
        polygon = np.array(cell["boundary_xy"], dtype=float)
        center_x = float(cell["center_x"])
        center_y = float(cell["center_y"])
        value_norm = float(score_norm[int(idx)])
        fill_color = hex_cmap(value_norm)
        side_color = darken_color(fill_color, 0.5)
        shadow_color = darken_color(fill_color, 0.34)

        prism_depth_x = 0.012 + 0.016 * value_norm
        prism_depth_y = 0.020 + 0.038 * value_norm
        offset = np.array([prism_depth_x, -prism_depth_y], dtype=float)
        shadow = polygon + offset

        for point_idx in range(len(polygon)):
            next_idx = (point_idx + 1) % len(polygon)
            quad = np.array(
                [shadow[point_idx], shadow[next_idx], polygon[next_idx], polygon[point_idx]],
                dtype=float,
            )
            ax.add_patch(
                patches.Polygon(
                    quad,
                    closed=True,
                    facecolor=side_color,
                    edgecolor="#15314f",
                    linewidth=0.25,
                    alpha=0.72,
                    zorder=2,
                )
            )

        ax.add_patch(
            patches.Polygon(
                shadow,
                closed=True,
                facecolor=shadow_color,
                edgecolor="#17304c",
                linewidth=0.3,
                alpha=0.68,
                zorder=3,
            )
        )
        ax.add_patch(
            patches.Polygon(
                polygon,
                closed=True,
                facecolor=fill_color,
                edgecolor="#d8e9ff",
                linewidth=0.45,
                alpha=0.95,
                zorder=4,
            )
        )

        tower_height = 0.042 + 0.29 * np.power(value_norm, 0.8)
        tower_width = 0.007 + 0.012 * np.power(value_norm, 0.6)
        tower_bottom = center_y + 0.004
        tower_left = center_x - tower_width / 2
        ax.add_patch(
            patches.Rectangle(
                (tower_left, tower_bottom),
                tower_width,
                tower_height,
                facecolor=fill_color,
                edgecolor="#75b3ff",
                linewidth=0.55,
                alpha=0.95,
                zorder=5,
            )
        )
        ax.add_patch(
            patches.Circle(
                (center_x, tower_bottom + tower_height),
                radius=max(0.006, tower_width * 0.65),
                facecolor=fill_color,
                edgecolor="#d9ebff",
                linewidth=0.55,
                alpha=0.98,
                zorder=6,
            )
        )

    if projected_cells:
        max_idx = int(np.argmax(scores))
        peak = projected_cells[max_idx]
        ax.scatter(
            [float(peak["center_x"])],
            [float(peak["center_y"])],
            s=1400,
            c=["#0f63ff"],
            marker="o",
            alpha=0.12,
            zorder=1,
        )
        ax.scatter(
            [float(peak["center_x"])],
            [float(peak["center_y"])],
            s=360,
            c=["#d7ebff"],
            marker="o",
            alpha=0.16,
            zorder=7,
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_reference_sidebar(
    ax: plt.Axes,
    projected_cells: list[dict[str, object]],
    ablation_rows: list[dict[str, str]],
    version_rows: list[dict[str, str]],
) -> None:
    ax.set_facecolor("#0b1724")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    card = patches.FancyBboxPatch(
        (0.02, 0.04),
        0.96,
        0.92,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        transform=ax.transAxes,
        facecolor="#0d2131",
        edgecolor="#263a4f",
        linewidth=1.1,
        alpha=0.98,
        zorder=1,
    )
    ax.add_patch(card)

    values = np.array([float(cell["pred_mean"]) for cell in projected_cells], dtype=float)
    bins = np.quantile(values, [0.0, 0.2, 0.45, 0.7, 1.0]) if values.size else np.array([0.0, 0.2, 0.45, 0.7, 1.0])
    swatches = ["#e6eef8", "#9fc4f7", "#5ea0eb", "#2f74c5"]

    ax.text(0.06, 0.90, "Layer Legend", fontsize=18, fontweight="bold", color="#f8fbff", ha="left", va="top")
    ax.text(0.06, 0.84, "pred_mean", fontsize=11.5, color="#9fb8d8", ha="left")
    ax.text(0.06, 0.80, "Height Scale by", fontsize=11, color="#8fa7c4", ha="left")
    ax.text(0.06, 0.76, "prediction", fontsize=12, fontweight="bold", color="#f3f7ff", ha="left")

    for idx, color in enumerate(swatches):
        top = 0.68 - idx * 0.07
        sw_x = 0.06
        sw_w = 0.16
        rect = patches.FancyBboxPatch(
            (sw_x, top),
            sw_w,
            0.052,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            transform=ax.transAxes,
            facecolor=color,
            edgecolor="#152b40",
            linewidth=0.8,
            zorder=3,
        )
        ax.add_patch(rect)
        ax.text(
            sw_x + sw_w + 0.03,
            top + 0.01,
            f"{bins[idx]:.3f} to {bins[idx + 1]:.3f}",
            fontsize=10.2,
            color="#d6e4f6",
            va="bottom",
        )

    ax.text(0.06, 0.44, "Selected model", fontsize=15.5, fontweight="bold", color="#f3f7ff", ha="left")
    h3_rows = [row for row in version_rows if row["label"].endswith("_h3")]
    v2_row = next((row for row in h3_rows if row["label"] == "v2_h3"), h3_rows[0]) if h3_rows else {"rmse": "0", "mae": "0", "pcc": "0"}
    ax.text(0.06, 0.36, "Version: V2-H3", fontsize=12.2, color="#cfe0f7", ha="left")
    ax.text(0.06, 0.32, f"RMSE  {float(v2_row['rmse']):.3f}", fontsize=11.6, color="#eaf2ff", ha="left")
    ax.text(0.06, 0.29, f"MAE   {float(v2_row['mae']):.3f}", fontsize=11.6, color="#eaf2ff", ha="left")
    ax.text(0.06, 0.26, f"PCC   {float(v2_row['pcc']):.3f}", fontsize=11.6, color="#eaf2ff", ha="left")

    ablation_row = next((row for row in ablation_rows if int(float(row["h3_resolution"])) == 8), ablation_rows[0]) if ablation_rows else {"h3_resolution": "8", "grid_count": "0", "test_zero_ratio": "0", "training_seconds": "0"}
    sel_box = patches.FancyBboxPatch(
        (0.06, 0.14),
        0.88,
        0.12,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        transform=ax.transAxes,
        facecolor="#11293d",
        edgecolor="#3f6e9b",
        linewidth=1.2,
        alpha=0.95,
        zorder=4,
    )
    ax.add_patch(sel_box)
    ax.text(0.08, 0.22, "Selected", fontsize=11.2, color="#9fd0ff", ha="left")
    ax.text(0.25, 0.22, f"H3 res={int(float(ablation_row['h3_resolution']))}", fontsize=12.8, fontweight="bold", color="#f6fbff", ha="left")
    ax.text(
        0.08,
        0.17,
        f"grid={int(float(ablation_row['grid_count']))}  zero={float(ablation_row['test_zero_ratio']) * 100:.1f}%  train={float(ablation_row['training_seconds']):.1f}s",
        fontsize=10.0,
        color="#d0e2f6",
        ha="left",
    )


def build_timeline_series(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, str, str]:
    rows = read_csv(path)
    first_day = datetime.strptime(rows[0]["time_slot"], "%Y-%m-%d %H:%M:%S").strftime("%m/%d/%Y")
    count_by_hour = defaultdict(int)
    pred_sum_by_hour = defaultdict(float)
    row_count_by_hour = defaultdict(int)
    for row in rows:
        slot = datetime.strptime(row["time_slot"], "%Y-%m-%d %H:%M:%S")
        hour = slot.hour
        count_by_hour[hour] += int(round(float(row["order_count"])))
        pred_sum_by_hour[hour] += float(row["prediction"])
        row_count_by_hour[hour] += 1

    hours = np.arange(24)
    volume = np.array([count_by_hour[h] for h in hours], dtype=float)
    mean_pred = np.array(
        [
            pred_sum_by_hour[h] / row_count_by_hour[h] if row_count_by_hour[h] else 0.0
            for h in hours
        ],
        dtype=float,
    )
    all_slots = [datetime.strptime(row["time_slot"], "%Y-%m-%d %H:%M:%S") for row in rows]
    start_time = min(all_slots).strftime("%I:%M:%S %p")
    end_time = max(all_slots).strftime("%I:%M:%S %p")
    return hours, volume, mean_pred, first_day, start_time, end_time


def draw_reference_timeline(
    ax: plt.Axes,
    hours: np.ndarray,
    volume: np.ndarray,
    mean_pred: np.ndarray,
    day_label: str,
    start_time: str,
    end_time: str,
) -> None:
    ax.set_facecolor("#1a2030")
    gradient = np.linspace(0, 1, 256)
    bg = np.outer(np.ones(256), gradient)
    cmap = LinearSegmentedColormap.from_list(
        "timeline_bg", ["#141b28", "#1d2333", "#20293b", "#252f45"]
    )
    ax.imshow(bg, extent=(0, 1, 0, 1), origin="lower", cmap=cmap, alpha=0.55, zorder=0)

    ax.text(0.02, 0.95, "Interval", fontsize=11, color="#d8e6fa", va="top")
    ax.text(0.14, 0.95, "Y Axis", fontsize=11, color="#b5c9e4", va="top")

    max_volume = float(volume.max()) if volume.size else 1.0
    scaled = volume / max_volume if max_volume > 0 else volume
    colors = [
        "#7b8a9d" if hour not in (8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19) else "#a3b8cc"
        for hour in hours
    ]
    ax.bar(hours, scaled * 0.7 + 0.12, width=0.32, color=colors, alpha=0.9, zorder=2)

    peak_hour = int(hours[int(np.argmax(volume))]) if volume.size else 12
    selection_x = peak_hour - 0.16
    selection_box = patches.FancyBboxPatch(
        (selection_x, 0.22),
        0.32,
        0.26,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        facecolor="#f3f7fb",
        edgecolor="#d0d8e8",
        linewidth=0.8,
        alpha=0.92,
        zorder=3,
    )
    ax.add_patch(selection_box)
    ax.text(0.50, 0.90, day_label, ha="center", va="center", fontsize=11.5, color="#c8d8eb", transform=ax.transAxes)
    ax.text(
        0.50,
        0.82,
        f"{start_time}  -  {end_time}",
        ha="center",
        va="center",
        fontsize=17,
        fontweight="bold",
        color="#f1f6ff",
        transform=ax.transAxes,
    )

    for tick, label in [
        (0, "November"),
        (3, "03 AM"),
        (6, "06 AM"),
        (9, "09 AM"),
        (12, "12 PM"),
        (15, "03 PM"),
        (18, "06 PM"),
        (21, "09 PM"),
    ]:
        ax.text(tick, -0.02, label, transform=ax.get_xaxis_transform(), fontsize=9.5, color="#b9c9db", ha="center", va="top")

    control_y = 0.08
    control_xs = [0.79, 0.84, 0.89, 0.94]
    labels = ["[]", ">", "R", "||"]
    for x, label in zip(control_xs, labels):
        button = patches.FancyBboxPatch(
            (x, control_y),
            0.035,
            0.095,
            boxstyle="round,pad=0.01,rounding_size=0.01",
            transform=ax.transAxes,
            facecolor="#7e8aa0",
            edgecolor="#aab4c8",
            linewidth=0.8,
            alpha=0.92,
            zorder=4,
        )
        ax.add_patch(button)
        ax.text(x + 0.0175, control_y + 0.047, label, ha="center", va="center", fontsize=10.5, color="#f5f8fb", transform=ax.transAxes, zorder=5)

    slider_x = 0.74
    ax.add_line(plt.Line2D([slider_x, 0.90], [0.135, 0.135], transform=ax.transAxes, color="#394a62", linewidth=2.0, zorder=4))
    ax.add_patch(patches.Rectangle((slider_x + 0.01, 0.115), 0.012, 0.04, transform=ax.transAxes, facecolor="#f0f4fa", edgecolor="#e8edf5", linewidth=0.7, zorder=5))
    ax.text(0.92, 0.13, "0.4x", transform=ax.transAxes, fontsize=10.5, color="#d2dce9", va="center")

    ax.set_xlim(-0.5, 23.5)
    ax.set_ylim(0, 1.02)
    ax.set_yticks([])
    ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_map_controls(ax: plt.Axes) -> None:
    icon_x = 0.92
    icon_y = 0.88
    spacing = 0.055
    for i in range(3):
        rect = patches.FancyBboxPatch(
            (icon_x, icon_y - i * spacing),
            0.045,
            0.045,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            transform=ax.transAxes,
            facecolor="#111821",
            edgecolor="#39485f",
            linewidth=0.8,
            alpha=0.95,
            zorder=12,
        )
        ax.add_patch(rect)
    cx = icon_x + 0.055
    cy = icon_y - 0.055
    cube = patches.Polygon(
        [[cx, cy], [cx + 0.03, cy + 0.01], [cx + 0.03, cy + 0.04], [cx, cy + 0.03]],
        closed=True,
        transform=ax.transAxes,
        facecolor="#1b324d",
        edgecolor="#2f4f78",
        linewidth=0.6,
        zorder=13,
    )
    ax.add_patch(cube)
    ax.text(icon_x + 0.017, icon_y + 0.06, "EN", color="#bcd6f7", fontsize=9, transform=ax.transAxes, zorder=14)


if __name__ == "__main__":
    style_figure()

    ablation_rows = read_csv(PICTURE_DIR / "h3_resolution_ablation.csv")
    version_rows = read_csv(MODELS_DIR / "h3_migration_comparison.csv")
    predictions_path = MODELS_DIR / "xgboost_hourly_v2_h3" / "test_predictions.csv"
    cells = load_h3_cells(predictions_path)
    projected_cells = build_projected_cells(cells, max_cells=150)
    hours, volume, mean_pred, day_label, start_time, end_time = build_timeline_series(predictions_path)

    fig = plt.figure(figsize=(18.5, 10.5), dpi=170)
    gs = fig.add_gridspec(2, 2, height_ratios=[3.0, 1.0], width_ratios=[3.2, 1.0], hspace=0.12, wspace=0.08)

    map_ax = fig.add_subplot(gs[0, 0])
    sidebar_ax = fig.add_subplot(gs[0, 1])
    timeline_ax = fig.add_subplot(gs[1, :])

    draw_map_background(map_ax)
    draw_prism_hex_map(map_ax, projected_cells)
    map_ax.text(0.50, 1.05, "H3 Grid Demand Visualization", transform=map_ax.transAxes, ha="center", va="bottom", fontsize=24, fontweight="bold", color="#f6fbff")
    map_ax.text(0.50, 1.00, "H3 hex demand map with prism height and dark dashboard styling", transform=map_ax.transAxes, ha="center", va="bottom", fontsize=12.5, color="#c2d8f0")
    pill = patches.FancyBboxPatch((0.37, 0.05), 0.26, 0.09, boxstyle="round,pad=0.02,rounding_size=0.03", transform=map_ax.transAxes, facecolor="#172233", edgecolor="#3e5878", linewidth=0.9, alpha=0.88, zorder=8)
    map_ax.add_patch(pill)
    map_ax.text(0.50, 0.103, day_label, transform=map_ax.transAxes, ha="center", va="center", fontsize=11.0, color="#ceddf0", zorder=9)
    peak_hour = int(hours[int(np.argmax(volume))]) if volume.size else 12
    map_ax.text(0.50, 0.067, f"{max(0, peak_hour - 1):02d}:00 AM  —  {min(23, peak_hour + 1):02d}:00 AM", transform=map_ax.transAxes, ha="center", va="center", fontsize=15.0, fontweight="bold", color="#f6fbff", zorder=9)

    version_rows_h3 = [row for row in version_rows if row["label"].endswith("_h3")]
    draw_reference_sidebar(sidebar_ax, projected_cells, ablation_rows, version_rows_h3)
    draw_reference_timeline(timeline_ax, hours, volume, mean_pred, day_label, start_time, end_time)
    draw_map_controls(map_ax)

    fig.suptitle("JTJM Model Results Dashboard", fontsize=28, fontweight="bold", y=0.985, color="#f6fbff")

    output = OUTPUT_DIR / "jtjm_model_results_dashboard_dark.png"
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved visualization to: {output}")