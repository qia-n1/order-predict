import argparse
import calendar
import json
from datetime import datetime
from pathlib import Path
from urllib.parse import quote_plus, urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd

base_url = "https://data.cityofnewyork.us/resource/2yzn-sicd.csv"

NYC_LATITUDE = 40.7128
NYC_LONGITUDE = -74.0060

WEATHER_COLUMNS = [
    "temperature", "apparent_temperature", "precipitation", "rain",
    "cloud_cover", "wind_speed", "humidity", "weather_code",
    "weather_clear", "weather_cloudy", "weather_fog", "weather_drizzle",
    "weather_rain", "weather_snow", "weather_thunder",
    "is_precipitating", "is_hot", "is_humid",
]


def month_range_to_bounds(month_str):
    y, m = map(int, month_str.split("-"))
    start = datetime(y, m, 1, 0, 0, 0)
    last_day = calendar.monthrange(y, m)[1]
    end = datetime(y, m, last_day, 23, 59, 59)
    return start.isoformat(), end.isoformat()


def build_where_clause(start_date=None, end_date=None, months=None):
    clauses = []
    if start_date and end_date:
        clauses.append(f"pickup_datetime between '{start_date}' and '{end_date}'")
    if months:
        for m in months:
            try:
                s, e = month_range_to_bounds(m)
                clauses.append(f"pickup_datetime between '{s}' and '{e}'")
            except Exception:
                continue
    if not clauses:
        return None
    return "(" + " OR ".join(clauses) + ")"


def _weather_code_to_flags(weather_code: int) -> dict[str, int]:
    return {
        "weather_clear": int(weather_code in {0, 1}),
        "weather_cloudy": int(weather_code in {2, 3}),
        "weather_fog": int(weather_code in {45, 48}),
        "weather_drizzle": int(weather_code in {51, 53, 55, 56, 57}),
        "weather_rain": int(weather_code in {61, 63, 65, 66, 67, 80, 81, 82}),
        "weather_snow": int(weather_code in {71, 73, 75, 77, 85, 86}),
        "weather_thunder": int(weather_code in {95, 96, 99}),
    }


def _empty_weather(time_slots: pd.Series) -> pd.DataFrame:
    slots = time_slots.drop_duplicates().sort_values()
    empty = pd.DataFrame({"time_slot": slots})
    for col in WEATHER_COLUMNS:
        empty[col] = 0.0
    return empty


def fetch_weather_data(start_date: str, end_date: str, out_path: Path, mock: bool = False) -> pd.DataFrame:
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    slot_range = pd.date_range(start_ts, end_ts, freq="1h")
    time_slots = pd.Series(slot_range, name="time_slot")

    if mock:
        print("Generating mock weather data...")
        return _generate_mock_weather(time_slots, out_path)

    query = urlencode({
        "latitude": NYC_LATITUDE,
        "longitude": NYC_LONGITUDE,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,apparent_temperature,precipitation,rain,cloud_cover,wind_speed_10m,relative_humidity_2m,weather_code",
        "timezone": "America/New_York",
    })
    url = f"https://archive-api.open-meteo.com/v1/archive?{query}"

    try:
        print(f"Fetching weather from Open-Meteo ({start_date} ~ {end_date})...")
        with urlopen(url, timeout=60) as response:
            payload = json.load(response)
    except Exception as exc:
        print(f"Weather API unreachable ({exc})")
        print("Falling back to mock weather data...")
        return _generate_mock_weather(time_slots, out_path)

    hourly = pd.DataFrame(payload["hourly"])
    hourly["time_slot"] = pd.to_datetime(hourly["time"])
    hourly = hourly.drop(columns=["time"])
    hourly = hourly.rename(columns={
        "temperature_2m": "temperature",
        "apparent_temperature": "apparent_temperature",
        "precipitation": "precipitation",
        "rain": "rain",
        "cloud_cover": "cloud_cover",
        "wind_speed_10m": "wind_speed",
        "relative_humidity_2m": "humidity",
    })

    weather_flags = hourly["weather_code"].fillna(-1).astype(int).map(_weather_code_to_flags)
    weather_flag_df = pd.DataFrame(weather_flags.tolist())
    hourly = pd.concat([hourly, weather_flag_df], axis=1)
    hourly["is_precipitating"] = (hourly["precipitation"] > 0).astype(int)
    hourly["is_hot"] = (hourly["temperature"] >= 30).astype(int)
    hourly["is_humid"] = (hourly["humidity"] >= 80).astype(int)

    hourly.to_csv(out_path, index=False)
    print(f"Weather data saved to {out_path} ({len(hourly)} hourly records)")
    return hourly


def _generate_mock_weather(time_slots: pd.Series, out_path: Path) -> pd.DataFrame:
    n = len(time_slots)
    rng = np.random.default_rng(42)
    df = pd.DataFrame({"time_slot": time_slots.sort_values().reset_index(drop=True)})
    hour_of_day = df["time_slot"].dt.hour.astype(float)
    day_of_year = df["time_slot"].dt.dayofyear.astype(float)

    base_temp = 20 + 8 * np.sin((day_of_year - 100) / 365 * 2 * np.pi)
    diurnal = 5 * np.sin((hour_of_day - 6) / 24 * 2 * np.pi)
    df["temperature"] = (base_temp + diurnal + rng.normal(0, 1.5, n)).round(1)
    df["apparent_temperature"] = (df["temperature"] - rng.uniform(0, 2, n)).round(1)
    df["precipitation"] = np.clip(rng.exponential(0.5, n) * (rng.random(n) < 0.12), 0, 30).round(1)
    df["rain"] = df["precipitation"].copy()
    df["cloud_cover"] = np.clip(rng.integers(0, 100, n).astype(float), 0, 100)
    df["wind_speed"] = np.clip(rng.weibull(2, n) * 8, 0, 25).round(1)
    df["humidity"] = np.clip(60 + rng.normal(0, 15, n), 20, 100).round(0)
    df["weather_code"] = np.select(
        [df["precipitation"] > 5, df["precipitation"] > 0, df["cloud_cover"] > 60],
        [61, 51, 2],
        default=0,
    ).astype(int)

    flags = df["weather_code"].map(_weather_code_to_flags)
    for col in ["weather_clear", "weather_cloudy", "weather_fog", "weather_drizzle",
                 "weather_rain", "weather_snow", "weather_thunder"]:
        df[col] = flags.map(lambda d, c=col: d[c])

    df["is_precipitating"] = (df["precipitation"] > 0).astype(int)
    df["is_hot"] = (df["temperature"] >= 30).astype(int)
    df["is_humid"] = (df["humidity"] >= 80).astype(int)

    df.to_csv(out_path, index=False)
    print(f"Mock weather data saved to {out_path} ({len(df)} hourly records)")
    return df


def cmd_taxi(args):
    total_rows_needed = args.total_rows
    chunk_size = args.chunk_size
    months = args.months.split(",") if args.months else None
    where = build_where_clause(start_date=args.start_date, end_date=args.end_date, months=months)
    df_list = []

    print("Fetching NYC Yellow Taxi data...")
    for offset in range(0, total_rows_needed, chunk_size):
        print(f"Fetching rows {offset} ~ {offset + chunk_size}...")
        params = f"$limit={chunk_size}&$offset={offset}"
        if where:
            params += "&$where=" + quote_plus(where)
        api_url = f"{base_url}?{params}"
        chunk = pd.read_csv(api_url)
        df_list.append(chunk)
        if len(chunk) < chunk_size:
            print("All data retrieved.")
            break

    if not df_list:
        print("No data retrieved.")
        return

    final_df = pd.concat(df_list, ignore_index=True)
    preview_columns = [col for col in ["pickup_datetime", "pickup_latitude", "pickup_longitude"] if col in final_df.columns]
    print(f"Retrieved {len(final_df)} rows.")
    print(final_df[preview_columns].head())
    final_df.to_csv(args.out, index=False)
    print(f"Saved to {args.out}")


def cmd_weather(args):
    project_root = Path(__file__).resolve().parent.parent
    out_path = Path(args.out) if args.out else project_root / "data" / f"nyc_weather_{args.start_date}_to_{args.end_date}.csv"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fetch_weather_data(args.start_date, args.end_date, out_path, mock=args.mock)


def main():
    parser = argparse.ArgumentParser(description="NYC Open Data fetcher & weather data fetcher")
    subparsers = parser.add_subparsers(dest="mode", help="Mode: taxi or weather")

    taxi_parser = subparsers.add_parser("taxi", help="Fetch taxi trip records")
    taxi_parser.add_argument("--total-rows", type=int, default=100000, help="Total rows to fetch")
    taxi_parser.add_argument("--chunk-size", type=int, default=50000, help="Chunk size per request")
    taxi_parser.add_argument("--start-date", type=str, help="Start ISO time, e.g. 2015-07-01T00:00:00")
    taxi_parser.add_argument("--end-date", type=str, help="End ISO time, e.g. 2015-07-31T23:59:59")
    taxi_parser.add_argument("--months", type=str, help="Comma-separated months, e.g. 2015-07,2015-08")
    taxi_parser.add_argument("--out", type=str, default="../data/NYC_YellowTaxi_2015_100k.csv", help="Output CSV path")

    weather_parser = subparsers.add_parser("weather", help="Fetch weather data from Open-Meteo")
    weather_parser.add_argument("--start-date", type=str, required=True, help="Start date YYYY-MM-DD")
    weather_parser.add_argument("--end-date", type=str, required=True, help="End date YYYY-MM-DD")
    weather_parser.add_argument("--out", type=str, default=None, help="Output CSV path")
    weather_parser.add_argument("--mock", action="store_true", help="Generate mock weather data instead of calling API")

    args = parser.parse_args()

    if args.mode == "weather":
        cmd_weather(args)
    elif args.mode == "taxi":
        cmd_taxi(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
