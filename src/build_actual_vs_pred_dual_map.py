"""
Build dual-map HTML: actual vs prediction (H3 hex, MapLibre).

Embeds one day of test_predictions with time histogram + playback (Kepler-style).

From repo root:
  python src/build_actual_vs_pred_dual_map.py
  python src/build_actual_vs_pred_dual_map.py --date 2015-06-30

Open via local HTTP. Template: visualization/templates/actual_vs_pred_dual.html
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h3
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "visualization" / "templates" / "actual_vs_pred_dual.html"


def h3_to_ring(cell: str) -> list[list[float]]:
    boundary = h3.cell_to_boundary(cell)
    ring: list[list[float]] = [[float(lng), float(lat)] for lat, lng in boundary]
    if len(ring) >= 2 and ring[0] != ring[-1]:
        ring.append(ring[0])
    return ring


def slot_to_ms(ts: pd.Timestamp) -> int:
    return int(pd.Timestamp(ts).value // 10**6)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--predictions",
        type=Path,
        default=ROOT / "models" / "xgboost_hourly_v3_may_jun" / "test_predictions.csv",
    )
    p.add_argument("--date", type=str, default="", help="YYYY-MM-DD; default = last day in test set")
    p.add_argument(
        "--output",
        type=Path,
        default=ROOT / "visualization" / "output" / "actual_vs_pred_dual.html",
    )
    p.add_argument("--window-hours", type=int, default=1, help="Hours shown per time step (1–3)")
    p.add_argument("--zoom", type=float, default=10.5)
    args = p.parse_args()

    if not TEMPLATE.is_file():
        raise SystemExit(f"Missing template: {TEMPLATE}")
    if not args.predictions.is_file():
        raise SystemExit(f"Missing predictions: {args.predictions}")

    df = pd.read_csv(args.predictions)
    need = {"h3_index", "time_slot", "order_count", "prediction"}
    if not need.issubset(df.columns):
        raise SystemExit(f"CSV must contain columns: {sorted(need)}")

    df = df.copy()
    df["time_slot"] = pd.to_datetime(df["time_slot"])
    if args.date.strip():
        day = pd.Timestamp(args.date.strip()).normalize()
    else:
        day = pd.Timestamp(df["time_slot"].max()).normalize()

    df = df.loc[df["time_slot"].dt.normalize() == day]
    if df.empty:
        raise SystemExit(f"No rows for date {day.date()}. Try another --date.")

    vmax = max(
        float(df["order_count"].max()),
        float(df["prediction"].max()),
        1e-6,
    )

    slots = sorted(df["time_slot"].unique())
    time_slots_ms = [slot_to_ms(s) for s in slots]
    time_labels = [pd.Timestamp(s).strftime("%Y-%m-%d %H:%M") for s in slots]
    hourly_actual = [float(df.loc[df["time_slot"] == s, "order_count"].sum()) for s in slots]
    hourly_pred = [float(df.loc[df["time_slot"] == s, "prediction"].sum()) for s in slots]

    cells: dict[str, list[list[float]]] = {}
    for cell in df["h3_index"].astype(str).unique():
        if h3.is_valid_cell(cell):
            cells[cell] = h3_to_ring(cell)

    records: list[dict] = []
    for _, row in df.iterrows():
        cell = str(row["h3_index"])
        if cell not in cells:
            continue
        records.append(
            {
                "h3_index": cell,
                "t": slot_to_ms(row["time_slot"]),
                "count": float(row["order_count"]),
                "prediction": float(row["prediction"]),
            }
        )

    centers: list[tuple[float, float]] = []
    for cell in cells:
        lat, lng = h3.cell_to_latlng(cell)
        centers.append((lat, lng))
    clat = sum(t[0] for t in centers) / len(centers)
    clng = sum(t[1] for t in centers) / len(centers)

    wh = max(1, min(int(args.window_hours), 3))
    meta = {
        "center": [clng, clat],
        "zoom": float(args.zoom),
        "bearing": 0.0,
        "vmax": vmax,
        "time_slots_ms": time_slots_ms,
        "time_labels": time_labels,
        "hourly_actual": hourly_actual,
        "hourly_pred": hourly_pred,
        "display_date": day.strftime("%m/%d/%Y"),
        "window_hours": wh,
        "default_idx": int(pd.Series(hourly_actual).idxmax()) if hourly_actual else 0,
    }

    html = TEMPLATE.read_text(encoding="utf-8")
    html = html.replace("@@JSON_META@@", json.dumps(meta, separators=(",", ":")))
    html = html.replace("@@JSON_CELLS@@", json.dumps(cells, separators=(",", ":")))
    html = html.replace("@@JSON_RECORDS@@", json.dumps(records, separators=(",", ":")))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(f"Written: {args.output} ({len(slots)} hours, {len(cells)} cells, {len(records)} records)")
    print("Open with local HTTP, e.g. python -m http.server 8765 from repo root, then:")
    print(f"  http://127.0.0.1:8765/{args.output.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
