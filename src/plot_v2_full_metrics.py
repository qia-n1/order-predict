from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "xgboost_hourly_v2_may_jun"
OUTPUT_DIR = ROOT / "models" / "picture"

BG = "#071427"
PANEL = "#0b1f3a"
TEXT = "#e7f0ff"
MUTED = "#a9bfd8"
GRID = "#2a4268"
BLUE = "#5ea0eb"
CYAN = "#70d6ff"
GREEN = "#7bd88f"
AMBER = "#f5b76b"
RED = "#f08a8a"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.facecolor": BG,
            "axes.facecolor": PANEL,
            "axes.edgecolor": "#5f7ea5",
            "axes.labelcolor": TEXT,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": TEXT,
            "axes.titleweight": "bold",
            "axes.grid": True,
            "grid.color": GRID,
            "grid.alpha": 0.45,
        }
    )


def metric_card(ax: plt.Axes, title: str, value: str, note: str, color: str) -> None:
    ax.set_axis_off()
    ax.text(0.04, 0.72, title, color=MUTED, fontsize=11, transform=ax.transAxes)
    ax.text(0.04, 0.38, value, color=color, fontsize=22, weight="bold", transform=ax.transAxes)
    ax.text(0.04, 0.14, note, color="#cbdcf0", fontsize=9.5, transform=ax.transAxes)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#284b6f")


def grouped_bars(ax: plt.Axes, labels: list[str], valid: list[float], test: list[float], title: str) -> None:
    x = np.arange(len(labels))
    width = 0.36
    ax.bar(x - width / 2, valid, width, label="Valid", color=BLUE)
    ax.bar(x + width / 2, test, width, label="Test", color=CYAN)
    ax.set_title(title, fontsize=13)
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.legend(frameon=False)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()

    metrics = load_json(MODEL_DIR / "metrics.json")
    evals = load_json(MODEL_DIR / "evals_result.json")
    importance = pd.read_csv(MODEL_DIR / "feature_importance.csv").head(10).iloc[::-1]

    fig = plt.figure(figsize=(18, 12), dpi=180)
    gs = fig.add_gridspec(4, 4, height_ratios=[0.85, 1.35, 1.35, 1.45], hspace=0.62, wspace=0.34)

    fig.text(0.03, 0.975, "V2 全量测评指标图", fontsize=24, weight="bold", ha="left", va="top")
    subtitle = (
        f"数据：Uber_2014_MayJun_features.csv | 划分：train_ratio=0.7, "
        f"valid_ratio=0.15, test_start_date=2014-06-24 | H3={metrics['h3_resolution']} "
        f"parent={metrics['parent_resolution']} | features={metrics['feature_count']} | freq={metrics['freq']}"
    )
    fig.text(0.03, 0.937, subtitle, fontsize=11.5, color=MUTED, ha="left", va="top")

    cards = [
        ("Test RMSE", f"{metrics['test_rmse']:.4f}", "整体误差", CYAN),
        ("Test MAE", f"{metrics['test_mae']:.4f}", "平均绝对误差", GREEN),
        ("Test PCC", f"{metrics['test_pcc']:.4f}", "预测相关性", AMBER),
        ("Best Iteration", str(metrics["best_iteration"]), "early stopping 结果", BLUE),
    ]
    for idx, card in enumerate(cards):
        metric_card(fig.add_subplot(gs[0, idx]), *card)

    ax = fig.add_subplot(gs[1, 0:2])
    grouped_bars(
        ax,
        ["RMSE", "MAE", "PCC"],
        [metrics["valid_rmse"], metrics["valid_mae"], metrics["valid_pcc"]],
        [metrics["test_rmse"], metrics["test_mae"], metrics["test_pcc"]],
        "整体指标",
    )

    ax = fig.add_subplot(gs[1, 2:4])
    grouped_bars(
        ax,
        ["Nonzero RMSE", "Nonzero MAE", "Nonzero PCC"],
        [metrics["valid_nonzero_rmse"], metrics["valid_nonzero_mae"], metrics["valid_nonzero_pcc"]],
        [metrics["test_nonzero_rmse"], metrics["test_nonzero_mae"], metrics["test_nonzero_pcc"]],
        "非零需求指标",
    )

    ax = fig.add_subplot(gs[2, 0:2])
    grouped_bars(
        ax,
        ["High RMSE", "High MAE", "PCC y>=4"],
        [metrics["valid_high_rmse"], metrics["valid_high_mae"], metrics["valid_pcc_y_ge_4"]],
        [metrics["test_high_rmse"], metrics["test_high_mae"], metrics["test_pcc_y_ge_4"]],
        "高需求指标（y >= 4）",
    )

    ax = fig.add_subplot(gs[2, 2])
    grouped_bars(
        ax,
        ["<=1", "<=2"],
        [metrics["valid_nonzero_within_1"], metrics["valid_nonzero_within_2"]],
        [metrics["test_nonzero_within_1"], metrics["test_nonzero_within_2"]],
        "非零样本容差命中率",
    )
    ax.set_ylim(0, 1)

    ax = fig.add_subplot(gs[2, 3])
    split_labels = ["Train", "Valid", "Test"]
    rows = [metrics["train_rows"], metrics["valid_rows"], metrics["test_rows"]]
    zeros = [metrics["train_zero_ratio"], metrics["valid_zero_ratio"], metrics["test_zero_ratio"]]
    ax.bar(split_labels, rows, color=[BLUE, CYAN, GREEN], alpha=0.85)
    ax.set_title("样本规模与零值比例", fontsize=13)
    ax.set_ylabel("Rows")
    ax2 = ax.twinx()
    ax2.plot(split_labels, zeros, color=AMBER, marker="o", linewidth=2.5)
    ax2.set_ylim(0.94, 0.97)
    ax2.tick_params(colors=MUTED)
    ax2.set_ylabel("Zero ratio", color=MUTED)

    ax = fig.add_subplot(gs[3, 0:2])
    ax.plot(evals["train"]["rmse"], color=BLUE, label="Train RMSE")
    ax.plot(evals["valid"]["rmse"], color=CYAN, label="Valid RMSE")
    ax.plot(evals["train"]["mae"], color=GREEN, label="Train MAE")
    ax.plot(evals["valid"]["mae"], color=AMBER, label="Valid MAE")
    ax.set_title("训练过程曲线", fontsize=13)
    ax.set_xlabel("Boosting round")
    ax.legend(frameon=False, ncol=2)

    ax = fig.add_subplot(gs[3, 2:4])
    ax.barh(importance["feature"], importance["gain"], color=RED, alpha=0.86)
    ax.set_title("Top 10 特征重要性（gain）", fontsize=13)
    ax.set_xlabel("Gain")

    output = OUTPUT_DIR / "v2图.png"
    fig.savefig(output, bbox_inches="tight", pad_inches=0.18, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
