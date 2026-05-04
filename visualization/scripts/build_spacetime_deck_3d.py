"""
生成 3D H3 六边形挤出 + 底部时间直方图 / 播放 的独立 HTML（deck.gl 9 + MapLibre）。

依赖：仅标准库 + pandas（与仓库根 requirements 一致），无需 pydeck。

用法（在 order-predict 根目录）：
  .venv\\Scripts\\python visualization\\scripts\\build_spacetime_deck_3d.py
  .venv\\Scripts\\python visualization\\scripts\\build_spacetime_deck_3d.py --date 2015-08-29 --output visualization/output/order_spacetime_3d.html
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from h3_utils import h3_center_latlng  # noqa: E402

_TPL = Path(__file__).resolve().parents[1] / "templates" / "spacetime_deck_3d.html"


def _prepare(
    path: Path,
    date: str | None,
    window_hours: int,
) -> tuple[dict, list[dict]]:
    df = pd.read_csv(path)
    if "h3_index" not in df.columns or "time_slot" not in df.columns:
        raise SystemExit("CSV 需包含 h3_index, time_slot")
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
        raise SystemExit("筛选后无数据，请检查 --date。")

    df["t"] = (df["time_slot"].astype("int64") // 10**6).astype("int64")
    df["count"] = df["order_count"].astype(float)
    df["prediction"] = df["prediction"].astype(float)

    centers = df["h3_index"].astype(str).map(h3_center_latlng)
    df["latitude"] = centers.map(lambda t: t[0])
    df["longitude"] = centers.map(lambda t: t[1])

    slots = sorted(df["t"].unique().tolist())
    n = len(slots)
    wh = max(1, min(window_hours, n))
    hourly_totals = [float(df.loc[df["t"] == t, "count"].sum()) for t in slots]

    count_max = float(df["count"].max())
    prediction_max = float(df["prediction"].max())
    vmax = max(count_max, prediction_max, 1e-6)
    elev_scale = max(30.0, min(220.0, 220.0 / vmax))

    ts0 = pd.Timestamp(df["time_slot"].min())
    display_date = ts0.strftime("%m/%d/%Y")

    meta = {
        "time_slots_ms": slots,
        "hourly_totals": hourly_totals,
        "display_date": display_date,
        "center_lat": float(df["latitude"].mean()),
        "center_lng": float(df["longitude"].mean()),
        "zoom": 10.8,
        "pitch": 52.0,
        "bearing": 22.0,
        "elevation_scale": elev_scale,
        "window_hours": wh,
        "count_max": count_max,
        "prediction_max": prediction_max,
    }
    records = df[["h3_index", "t", "count", "prediction"]].to_dict(orient="records")
    return meta, records


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--predictions",
        type=Path,
        default=_ROOT / "models" / "xgboost_hourly_v3_h3" / "test_predictions.csv",
    )
    p.add_argument("--date", type=str, default="", help="YYYY-MM-DD；省略为数据中最后一天")
    p.add_argument(
        "--window-hours",
        type=int,
        default=1,
        help="时间窗宽度（小时槽数），1–6",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=_ROOT / "visualization" / "output" / "order_spacetime_3d.html",
    )
    args = p.parse_args()

    if not args.predictions.is_file():
        raise SystemExit(f"找不到: {args.predictions}")
    if not _TPL.is_file():
        raise SystemExit(f"缺少模板: {_TPL}")

    meta, records = _prepare(
        args.predictions,
        args.date.strip() or None,
        max(1, min(args.window_hours, 6)),
    )
    html = _TPL.read_text(encoding="utf-8")
    html = html.replace("@@EMBED_META@@", json.dumps(meta), 1)
    html = html.replace("@@EMBED_DATA@@", json.dumps(records, separators=(",", ":")), 1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(f"已写入: {args.output}")
    print("请用本地 HTTP 打开；需可访问 unpkg/esm.sh 与 CARTO 瓦片（import 已默认 unpkg，避免 jsdelivr 对 mapbox 包 400）。")


if __name__ == "__main__":
    main()
