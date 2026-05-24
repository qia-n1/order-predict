from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches
from matplotlib.colors import LinearSegmentedColormap


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
PICTURE_DIR = MODELS_DIR / "picture"

BG = "#071427"
PANEL = "#0b1f3a"
PANEL_2 = "#102845"
TEXT = "#e7f0ff"
MUTED = "#a9bfd8"
GRID = "#2a4268"
BLUE = "#5ea0eb"
CYAN = "#70d6ff"
AMBER = "#f5b76b"
GREEN = "#7bd88f"
RED = "#f08a8a"


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": ["Kaiti SC", "Kai", "Arial Unicode MS", "Hiragino Sans GB", "Songti SC", "DejaVu Sans"],
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
            "grid.linestyle": "-",
        }
    )


def save(fig: plt.Figure, name: str, size: tuple[int, int]) -> None:
    path = PICTURE_DIR / name
    fig.set_size_inches(size[0] / 180, size[1] / 180)
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.18, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved {path}")


def add_title(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.text(0.035, 0.955, title, fontsize=22, weight="bold", color="#f6fbff", ha="left", va="top")
    fig.text(0.035, 0.875, subtitle, fontsize=12, color=MUTED, ha="left", va="top")


def rounded_panel(ax: plt.Axes, color: str = PANEL, edge: str = "#263a4f") -> None:
    ax.set_facecolor("none")
    for spine in ax.spines.values():
        spine.set_visible(False)
    rect = patches.FancyBboxPatch(
        (0, 0),
        1,
        1,
        transform=ax.transAxes,
        boxstyle="round,pad=0.018,rounding_size=0.04",
        facecolor=color,
        edgecolor=edge,
        linewidth=1.1,
        alpha=0.98,
        zorder=-10,
    )
    ax.add_patch(rect)


def metric_card(ax: plt.Axes, label: str, value: str, note: str, color: str) -> None:
    ax.set_axis_off()
    rounded_panel(ax, PANEL_2, "#3f6e9b")
    value_size = 18 if len(value) >= 7 else 23
    ax.text(0.07, 0.70, label, fontsize=12, color=MUTED, transform=ax.transAxes)
    ax.text(0.07, 0.39, value, fontsize=value_size, weight="bold", color=color, transform=ax.transAxes)
    ax.text(0.07, 0.15, note, fontsize=10, color="#cbdcf0", transform=ax.transAxes)


def load_ablation() -> pd.DataFrame:
    return pd.read_csv(PICTURE_DIR / "h3_resolution_ablation.csv")


def load_feature_importance() -> pd.DataFrame:
    return pd.read_csv(PICTURE_DIR / "v2_h3_feature_importance_annotated.csv")


def load_feature_summary() -> pd.DataFrame:
    return pd.read_csv(PICTURE_DIR / "v2_h3_feature_category_summary.csv")


def draw_table(ax: plt.Axes, df: pd.DataFrame, columns: list[str], labels: list[str]) -> None:
    ax.set_axis_off()
    rounded_panel(ax)
    values = df[columns].copy()
    for col in values.columns:
        if values[col].dtype.kind in "fc":
            values[col] = values[col].map(lambda value: f"{value:.3f}")
    table = ax.table(
        cellText=values.astype(str).values,
        colLabels=labels,
        loc="center",
        cellLoc="center",
        colLoc="center",
        bbox=[0.03, 0.05, 0.94, 0.86],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#2b4a6b")
        cell.set_linewidth(0.7)
        if row == 0:
            cell.set_facecolor("#17375a")
            cell.set_text_props(color="#f6fbff", weight="bold")
        else:
            cell.set_facecolor("#0f2945" if row % 2 else "#0c2139")
            cell.set_text_props(color="#dbeaff")


def plot_ablation_table() -> None:
    df = load_ablation()
    fig = plt.figure(figsize=(15, 7.7), dpi=180)
    add_title(fig, "H3 分辨率消融实验", "基于 5、6 月数据，比较不同 H3 网格分辨率下的需求预测效果")
    ax = fig.add_axes([0.035, 0.08, 0.93, 0.78])
    cols = ["h3_resolution", "grid_count", "test_rows", "test_zero_ratio", "test_rmse", "test_mae", "test_pcc", "training_seconds"]
    labels = ["H3", "网格数", "测试行数", "零值比例", "RMSE", "MAE", "PCC", "训练秒数"]
    draw_table(ax, df, cols, labels)
    save(fig, "h3_resolution_ablation_table.png", (2717, 1388))


def plot_ablation_trends() -> None:
    df = load_ablation()
    fig, axes = plt.subplots(2, 2, figsize=(15.4, 9), dpi=180)
    add_title(fig, "消融指标趋势", "展示 H3 分辨率变化时，误差、相关性和稀疏性的变化")
    metrics = [
        ("test_rmse", "测试集 RMSE", BLUE),
        ("test_mae", "测试集 MAE", CYAN),
        ("test_pcc", "测试集 PCC", GREEN),
        ("test_zero_ratio", "零需求比例", AMBER),
    ]
    for ax, (col, title, color) in zip(axes.ravel(), metrics):
        rounded_panel(ax)
        ax.plot(df["h3_resolution"], df[col], marker="o", color=color, linewidth=3, markersize=8)
        ax.fill_between(df["h3_resolution"], df[col], alpha=0.12, color=color)
        ax.set_title(title, fontsize=16, color=TEXT, pad=14)
        ax.set_xlabel("H3 分辨率")
        ax.set_xticks(df["h3_resolution"])
        ax.grid(True)
    fig.tight_layout(rect=[0.02, 0.04, 0.98, 0.88])
    save(fig, "h3_resolution_ablation_trends.png", (2769, 1616))


def plot_sparsity_tradeoff() -> None:
    df = load_ablation()
    fig, ax = plt.subplots(figsize=(14.4, 8.2), dpi=180)
    add_title(fig, "分辨率与稀疏性权衡", "更高 H3 分辨率降低 MAE，但也带来更高零需求比例和更大数据规模")
    rounded_panel(ax)
    sizes = np.sqrt(df["grid_count"]) * 34
    scatter = ax.scatter(df["test_zero_ratio"] * 100, df["test_mae"], s=sizes, c=df["h3_resolution"], cmap="Blues", edgecolors="#e7f0ff", linewidths=1.2)
    for _, row in df.iterrows():
        ax.text(row["test_zero_ratio"] * 100 + 0.12, row["test_mae"], f"分辨率 {int(row['h3_resolution'])}", fontsize=12, color=TEXT)
    ax.set_xlabel("测试集零需求比例（%）")
    ax.set_ylabel("测试集 MAE")
    ax.invert_yaxis()
    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("H3 分辨率", color=TEXT)
    cbar.ax.yaxis.set_tick_params(color=MUTED)
    plt.setp(cbar.ax.get_yticklabels(), color=MUTED)
    fig.tight_layout(rect=[0.03, 0.05, 0.96, 0.88])
    save(fig, "h3_resolution_sparsity_tradeoff.png", (2592, 1478))


def plot_ablation_ppt_summary() -> None:
    df = load_ablation()
    best = df.loc[df["test_mae"].idxmin()]
    balanced = df.loc[df["h3_resolution"].eq(8)].iloc[0]
    fig = plt.figure(figsize=(15.9, 7.0), dpi=180)
    add_title(fig, "H3 分辨率消融总结", "5、6 月消融结果：网格越细误差越低，但稀疏性和计算规模快速上升")
    card_positions = [[0.04, 0.60, 0.21, 0.22], [0.28, 0.60, 0.21, 0.22], [0.52, 0.60, 0.21, 0.22], [0.76, 0.60, 0.20, 0.22]]
    cards = [
        ("最低 MAE", f"分辨率 {int(best['h3_resolution'])}", f"MAE {best['test_mae']:.3f}", CYAN),
        ("平衡选择", "分辨率 8", f"{int(balanced['grid_count'])} 个网格", BLUE),
        ("测试窗口", "168 小时", "6 月 24 日至 6 月 30 日", GREEN),
        ("最大规模", f"{int(df['test_rows'].max()):,}", "分辨率 9 的测试行数", AMBER),
    ]
    for pos, card in zip(card_positions, cards):
        metric_card(fig.add_axes(pos), *card)
    ax = fig.add_axes([0.05, 0.12, 0.90, 0.34])
    rounded_panel(ax)
    x = np.arange(len(df))
    ax.bar(x - 0.18, df["test_mae"], width=0.36, color=CYAN, label="MAE")
    ax2 = ax.twinx()
    ax2.plot(x + 0.18, df["grid_count"], marker="o", color=AMBER, linewidth=3, label="网格数")
    ax.set_xticks(x, [f"分辨率 {int(v)}" for v in df["h3_resolution"]])
    ax.set_ylabel("MAE")
    ax2.set_ylabel("网格数", color=AMBER)
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    save(fig, "h3_resolution_ablation_ppt_summary.png", (2866, 1266))


def plot_feature_top15() -> None:
    df = load_feature_importance().head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(15.2, 8.2), dpi=180)
    add_title(fig, "V2-H3 特征重要性前 15", "基于 5、6 月数据的 XGBoost 增益排序")
    rounded_panel(ax)
    colors = LinearSegmentedColormap.from_list("bars", ["#2f74c5", "#70d6ff"])(np.linspace(0.2, 0.9, len(df)))
    ax.barh(df["display"], df["gain_pct"], color=colors, edgecolor="#cce8ff", linewidth=0.6)
    for idx, value in enumerate(df["gain_pct"]):
        ax.text(value + 0.3, idx, f"{value:.1f}%", va="center", fontsize=11)
    ax.set_xlabel("增益占比（%）")
    ax.set_ylabel("")
    fig.tight_layout(rect=[0.03, 0.05, 0.98, 0.88])
    save(fig, "v2_h3_feature_importance_top15.png", (2740, 1473))


def plot_category_contribution() -> None:
    df = load_feature_summary()
    fig = plt.figure(figsize=(14.4, 7.8), dpi=180)
    add_title(fig, "特征类别贡献", "按语义类别汇总 5、6 月 V2-H3 模型的特征增益")
    ax = fig.add_axes([0.05, 0.08, 0.52, 0.70])
    rounded_panel(ax)
    colors = [BLUE, CYAN, AMBER, GREEN, "#9ac5ff", RED][: len(df)]
    wedges, _ = ax.pie(df["gain_pct"], colors=colors, startangle=110, radius=0.82, wedgeprops={"width": 0.34, "edgecolor": BG, "linewidth": 2})
    ax.text(0, 0.05, "V2-H3", ha="center", va="center", fontsize=24, weight="bold")
    ax.text(0, -0.16, "5、6 月", ha="center", va="center", fontsize=13, color=MUTED)
    legend_ax = fig.add_axes([0.61, 0.15, 0.34, 0.56])
    legend_ax.set_axis_off()
    rounded_panel(legend_ax, "#0c2139", "#274866")
    for idx, row in enumerate(df.itertuples(index=False)):
        y = 0.82 - idx * 0.15
        legend_ax.add_patch(
            patches.Rectangle(
                (0.08, y - 0.035),
                0.10,
                0.07,
                transform=legend_ax.transAxes,
                facecolor=colors[idx],
                edgecolor="#d8e9ff",
                linewidth=0.8,
            )
        )
        legend_ax.text(0.23, y, f"{row.category}", transform=legend_ax.transAxes, ha="left", va="center", fontsize=15, color=TEXT)
        legend_ax.text(0.88, y, f"{row.gain_pct:.1f}%", transform=legend_ax.transAxes, ha="right", va="center", fontsize=15, color="#f6fbff")
    save(fig, "v2_h3_feature_category_contribution.png", (2598, 1403))


def plot_feature_group_breakdown() -> None:
    df = load_feature_importance()
    categories = load_feature_summary()["category"].tolist()
    fig, axes = plt.subplots(len(categories), 1, figsize=(15.4, 9.3), dpi=180, sharex=True)
    add_title(fig, "特征组分解", "展示 5、6 月 V2-H3 模型中各类别内部的主要特征")
    for ax, category in zip(axes, categories):
        rounded_panel(ax, "#0c2139")
        sub = df[df["category"] == category].head(5).iloc[::-1]
        ax.barh(sub["display"], sub["gain_pct"], color=BLUE, alpha=0.88)
        ax.set_title(category, loc="left", fontsize=13, color=TEXT)
        ax.tick_params(axis="y", labelsize=10)
    axes[-1].set_xlabel("增益占比（%）")
    fig.tight_layout(rect=[0.03, 0.04, 0.98, 0.78], h_pad=1.2)
    save(fig, "v2_h3_feature_group_breakdown.png", (2769, 1679))


def plot_feature_ppt_summary() -> None:
    imp = load_feature_importance()
    summary = load_feature_summary()
    top = imp.iloc[0]
    fig = plt.figure(figsize=(17.8, 7.1), dpi=180)
    add_title(fig, "V2-H3 特征重要性总结", "5、6 月 V2 模型主要依赖区域历史需求与滚动需求记忆")
    metric_card(fig.add_axes([0.04, 0.58, 0.25, 0.24]), "首要特征", str(top["display"]), f"增益占比 {top['gain_pct']:.1f}%", CYAN)
    metric_card(fig.add_axes([0.32, 0.58, 0.20, 0.24]), "有效特征数", f"{len(imp)}", "非零增益特征", BLUE)
    metric_card(fig.add_axes([0.55, 0.58, 0.20, 0.24]), "主导类别", str(summary.iloc[0]["category"]), f"{summary.iloc[0]['gain_pct']:.1f}%", GREEN)
    metric_card(fig.add_axes([0.78, 0.58, 0.18, 0.24]), "前五累计", f"{imp.head(5)['gain_pct'].sum():.1f}%", "累计增益占比", AMBER)
    ax = fig.add_axes([0.05, 0.11, 0.90, 0.34])
    rounded_panel(ax)
    top10 = imp.head(10).iloc[::-1]
    ax.barh(top10["display"], top10["gain_pct"], color=CYAN)
    ax.set_xlabel("增益占比（%）")
    save(fig, "v2_h3_feature_importance_ppt_summary.png", (3209, 1274))


def plot_version_comparison_table() -> None:
    df = pd.read_csv(MODELS_DIR / "v1_v2_v3_comparison.csv")
    fig = plt.figure(figsize=(14.1, 7.5), dpi=180)
    add_title(fig, "V1 / V2 / V3 模型对比", "基于 5、6 月数据，比较三版模型在 6 月最后一周测试集上的表现")
    ax = fig.add_axes([0.035, 0.10, 0.93, 0.74])
    cols = ["version", "rmse_end", "mae_end", "pcc_end", "nonzero_rmse_end", "nonzero_mae_end", "nonzero_pcc_end", "within_1_end", "within_2_end"]
    labels = ["版本", "RMSE", "MAE", "PCC", "非零 RMSE", "非零 MAE", "非零 PCC", "误差≤1", "误差≤2"]
    draw_table(ax, df, cols, labels)
    save(fig, "h3_v1_v2_v3_comparison_table.png", (2538, 1354))


def main() -> None:
    PICTURE_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()
    plot_ablation_ppt_summary()
    plot_ablation_table()
    plot_ablation_trends()
    plot_sparsity_tradeoff()
    plot_feature_top15()
    plot_category_contribution()
    plot_feature_group_breakdown()
    plot_feature_ppt_summary()
    plot_version_comparison_table()


if __name__ == "__main__":
    main()
