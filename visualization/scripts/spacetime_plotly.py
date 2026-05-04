"""
订单预测结果：地理散点（H3 中心）+ 时间直方图联动（Plotly animation frames）。

数据：默认使用 models/xgboost_hourly_v3_h3/test_predictions.csv
（h3_index, time_slot, order_count, prediction）

说明：Plotly 6 已移除 hexbin_mapbox；本脚本以 H3 元心点 + 颜色/大小编码密度，
     与参考图「六边形 + 时间轴」在交互逻辑上对齐。需真 3D 六边形柱请使用 deck.gl/Kepler。

用法（在 order-predict 根目录）：
  .venv\\Scripts\\python visualization\\scripts\\spacetime_plotly.py
  .venv\\Scripts\\python visualization\\scripts\\spacetime_plotly.py --date 2015-08-29 --metric prediction
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from h3_utils import h3_center_latlng  # noqa: E402

# 与参考图例接近的蓝色系（末段由数据最大值决定 colorbar 上限）
BLUES_CONTINUOUS = "Blues"
PAPER_BG = "#0b0c0f"
PLOT_BG = "#14151a"
BAR_BASE = "rgba(66, 133, 244, 0.45)"
BAR_HIGH = "rgba(138, 180, 248, 0.95)"


def _prepare(
    path: Path,
    date: str | None,
    metric: str,
) -> tuple[pd.DataFrame, str, str]:
    df = pd.read_csv(path)
    if "h3_index" not in df.columns or "time_slot" not in df.columns:
        raise SystemExit("CSV 需包含 h3_index, time_slot 列")
    df = df.copy()
    df["time_slot"] = pd.to_datetime(df["time_slot"])
    if date:
        day = pd.Timestamp(date).normalize()
        df = df.loc[df["time_slot"].dt.normalize() == day]
    else:
        last = df["time_slot"].max()
        day = pd.Timestamp(last).normalize()
        df = df.loc[df["time_slot"].dt.normalize() == day]
    if df.empty:
        raise SystemExit("筛选后无数据，请检查 --date 与 CSV。")

    if metric == "count" and "order_count" in df.columns:
        df["value"] = df["order_count"].astype(float)
    elif metric == "prediction" and "prediction" in df.columns:
        df["value"] = df["prediction"].astype(float)
    else:
        raise SystemExit("metric 需为 count（order_count）或 prediction")

    centers = df["h3_index"].astype(str).map(h3_center_latlng)
    df["lat"] = centers.map(lambda t: t[0])
    df["lon"] = centers.map(lambda t: t[1])
    # 与参考图 count 图例：弱区仍可见
    df["value_pos"] = df["value"].clip(lower=0.0)
    return df, metric, day.strftime("%Y-%m-%d")


def _marker_size(arr: np.ndarray, ref_max: float) -> np.ndarray:
    ref_max = max(float(ref_max), 1e-6)
    s = np.sqrt(np.maximum(arr, 0.0) / ref_max)
    return 4.0 + 22.0 * s


def build_figure(
    df: pd.DataFrame,
    metric: str,
    day_label: str,
    mapbox_token: str | None,
) -> go.Figure:
    if mapbox_token:
        px.set_mapbox_access_token(mapbox_token)

    slot_list = sorted(df["time_slot"].unique())
    n = len(slot_list)
    totals: list[float] = []
    for t in slot_list:
        totals.append(float(df.loc[df["time_slot"] == t, "value"].sum()))
    totals_arr = np.array(totals, dtype=float)
    vmax = float(df["value"].max())
    vmax = max(vmax, 1e-6)
    ref_max = float(df["value"].max())
    ref_max = max(ref_max, 1e-6)

    slot_labels = [pd.Timestamp(s).strftime("%H:%M") for s in slot_list]

    center_lat = float(df["lat"].mean())
    center_lon = float(df["lon"].mean())

    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.72, 0.28],
        specs=[[{"type": "scattermap"}], [{"type": "xy"}]],
        vertical_spacing=0.06,
    )

    def bar_colors(selected_i: int) -> list[str]:
        out = []
        for i in range(n):
            if i == selected_i:
                out.append(BAR_HIGH)
            else:
                out.append(BAR_BASE)
        return out

    # 初始帧 0
    sub0 = df[df["time_slot"] == slot_list[0]]
    sz0 = _marker_size(sub0["value_pos"].values, ref_max)
    fig.add_trace(
        go.Scattermap(
            lat=sub0["lat"],
            lon=sub0["lon"],
            mode="markers",
            marker=dict(
                size=sz0,
                color=sub0["value_pos"],
                colorscale=BLUES_CONTINUOUS,
                cmin=0,
                cmax=vmax,
                opacity=0.88,
                showscale=True,
                colorbar=dict(
                    title=dict(text=metric, side="right"),
                    tickmode="linear",
                    tick0=0,
                    dtick=max(vmax / 5, 1) if vmax > 5 else 1,
                    len=0.55,
                    y=0.82,
                    bgcolor="rgba(20,21,26,0.85)",
                    bordercolor="#3c4043",
                ),
            ),
            text=sub0["h3_index"],
            hovertemplate=(
                "<b>%{text}</b><br>"
                + metric
                + ": %{marker.color:.3f}<br>"
                + "lat %{lat:.4f}, lon %{lon:.4f}<extra></extra>"
            ),
            name="cells",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=list(range(n)),
            y=totals_arr.tolist(),
            marker=dict(color=bar_colors(0)),
            hovertemplate="%{customdata}<br>合计 %{y:.1f}<extra></extra>",
            customdata=slot_labels,
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    frames: list[go.Frame] = []
    for i, ts in enumerate(slot_list):
        sub = df[df["time_slot"] == ts]
        sz = _marker_size(sub["value_pos"].values, ref_max)
        frames.append(
            go.Frame(
                name=str(i),
                data=[
                    go.Scattermap(
                        lat=sub["lat"],
                        lon=sub["lon"],
                        mode="markers",
                        marker=dict(
                            size=sz,
                            color=sub["value_pos"],
                            colorscale=BLUES_CONTINUOUS,
                            cmin=0,
                            cmax=vmax,
                            opacity=0.88,
                            showscale=False,
                        ),
                        text=sub["h3_index"],
                        hovertemplate=(
                            "<b>%{text}</b><br>"
                            + metric
                            + ": %{marker.color:.3f}<br>"
                            + "lat %{lat:.4f}, lon %{lon:.4f}<extra></extra>"
                        ),
                    ),
                    go.Bar(
                        x=list(range(n)),
                        y=totals_arr.tolist(),
                        marker=dict(color=bar_colors(i)),
                        customdata=slot_labels,
                    ),
                ],
            )
        )

    fig.frames = frames

    tick_idx = list(range(0, n, max(1, n // 8)))
    if (n - 1) not in tick_idx:
        tick_idx.append(n - 1)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(color="#e8eaed", size=12),
        height=820,
        margin=dict(l=8, r=8, t=48, b=8),
        title=dict(
            text=f"订单时空分布 · {day_label} · {metric}",
            x=0.5,
            font=dict(size=15),
        ),
        map=dict(
            style="carto-darkmatter",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=10.8,
        ),
        annotations=[
            dict(
                x=1.0,
                y=1.12,
                xref="paper",
                yref="paper",
                showarrow=False,
                align="right",
                text=(
                    "<b>Layer Legend</b><br>"
                    + f"<span style='font-size:11px'>{metric}</span><br>"
                    "<span style='font-size:9px;color:#9aa0a6'>Height/size ∝ "
                    + metric
                    + "</span>"
                ),
                bgcolor="rgba(20,21,26,0.9)",
                bordercolor="#3c4043",
                borderwidth=1,
                borderpad=6,
            )
        ],
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                x=0.18,
                y=-0.02,
                xanchor="left",
                yanchor="top",
                direction="left",
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=650, redraw=True),
                                fromcurrent=True,
                                transition=dict(duration=0),
                                mode="immediate",
                            ),
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            dict(frame=dict(duration=0, redraw=False), mode="immediate"),
                        ],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                active=0,
                pad=dict(t=24),
                steps=[
                    dict(
                        method="animate",
                        args=[
                            [str(i)],
                            dict(
                                mode="immediate",
                                frame=dict(duration=400, redraw=True),
                                transition=dict(duration=0),
                            ),
                        ],
                        label=slot_labels[i],
                    )
                    for i in range(n)
                ],
                x=0.12,
                len=0.86,
                y=-0.18,
                currentvalue=dict(
                    prefix=f"{day_label} · ",
                    visible=True,
                    xanchor="left",
                    font=dict(size=12, color="#bdc1c6"),
                ),
            )
        ],
    )

    fig.update_xaxes(
        title_text="时间（当日各小时槽）",
        tickvals=tick_idx,
        ticktext=[slot_labels[j] for j in tick_idx],
        gridcolor="#2d2f36",
        zeroline=False,
        row=2,
        col=1,
    )
    fig.update_yaxes(
        title_text="订单量合计（该槽全图）",
        gridcolor="#2d2f36",
        zeroline=False,
        row=2,
        col=1,
    )

    # 底部时间轴标题区：模拟参考图「November / 03 AM …」
    month_name = pd.Timestamp(slot_list[0]).strftime("%B")
    fig.add_annotation(
        x=0,
        y=1.02,
        xref="x2",
        yref="y2 domain",
        text=month_name,
        showarrow=False,
        font=dict(size=11, color="#9aa0a6"),
        xanchor="left",
    )

    return fig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--predictions",
        type=Path,
        default=_ROOT / "models" / "xgboost_hourly_v3_h3" / "test_predictions.csv",
        help="面板 CSV（h3_index, time_slot, order_count, prediction）",
    )
    p.add_argument(
        "--date",
        type=str,
        default="",
        help="单日 YYYY-MM-DD；省略则取数据中最后一天",
    )
    p.add_argument(
        "--metric",
        choices=("count", "prediction"),
        default="count",
        help="地图颜色字段：count=order_count，prediction=预测值",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=_ROOT / "visualization" / "output" / "order_spacetime.html",
        help="输出 HTML 路径",
    )
    p.add_argument(
        "--mapbox-token",
        type=str,
        default="",
        help="可选 Mapbox token；留空则使用 Carto 深色底图（无需 token）",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.predictions.is_file():
        raise SystemExit(f"找不到文件: {args.predictions}")

    df, metric, _day = _prepare(
        args.predictions,
        args.date.strip() or None,
        args.metric,
    )
    tok = args.mapbox_token.strip() or None
    fig = build_figure(df, metric, _day, tok)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(args.output, include_plotlyjs="cdn", full_html=True)
    print(f"已写入: {args.output}")


if __name__ == "__main__":
    main()
