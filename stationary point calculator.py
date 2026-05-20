# ============================================================
# NVDA OPTION A STATIONARY POINT DETECTOR
# No quadratic fitting
# Uses slope / gradient sign changes on EMA-smoothed close
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed


# ============================================================
# 1. CONFIG
# ============================================================

load_dotenv()

#ALPACA_API_KEY = ""
#ALPACA_SECRET_KEY = ""

if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
    raise ValueError("Missing Alpaca keys. Add ALPACA_API_KEY and ALPACA_SECRET_KEY to your .env file.")

data_client = StockHistoricalDataClient(
    ALPACA_API_KEY,
    ALPACA_SECRET_KEY
)

NY_TZ = ZoneInfo("America/New_York")
SYMBOL = "NVDA"

DATA_FEED = DataFeed.IEX
# DATA_FEED = DataFeed.SIP  # use if your Alpaca plan supports SIP

EMA_PRICE_SPAN = 9

# How many minutes apart to estimate slope
SLOPE_LOOKBACK = 3

# How many minutes apart to estimate curvature
CURVATURE_LOOKBACK = 3

# Filters
MIN_ABS_SLOPE = 0.015          # dollars per SLOPE_LOOKBACK minutes
MIN_ABS_CURVATURE = 0.005      # change in slope threshold
COOLDOWN_MINUTES = 8           # prevents repeated markers near same turn

IGNORE_BEFORE_TIME = "09:45"


# ============================================================
# 2. GET PREVIOUS TRADING DAY
# ============================================================

def get_previous_trading_day_ny():
    today_ny = datetime.now(NY_TZ).date()
    d = today_ny - timedelta(days=1)

    while d.weekday() >= 5:
        d -= timedelta(days=1)

    return d


target_day = get_previous_trading_day_ny()

# Manual override if needed:
# target_day = pd.Timestamp("2026-05-19").date()

market_open_ny = datetime(
    target_day.year, target_day.month, target_day.day,
    9, 30, tzinfo=NY_TZ
)

market_close_ny = datetime(
    target_day.year, target_day.month, target_day.day,
    16, 0, tzinfo=NY_TZ
)

start_utc = market_open_ny.astimezone(timezone.utc)
end_utc = market_close_ny.astimezone(timezone.utc)

print(f"Downloading {SYMBOL} 1-minute bars for {target_day}")


# ============================================================
# 3. DOWNLOAD DATA
# ============================================================

request = StockBarsRequest(
    symbol_or_symbols=[SYMBOL],
    timeframe=TimeFrame.Minute,
    start=start_utc,
    end=end_utc,
    feed=DATA_FEED
)

bars = data_client.get_stock_bars(request).df

if bars.empty:
    raise ValueError("No data returned. Check date, Alpaca permissions, or data feed.")

df = bars.reset_index()
df = df[["symbol", "timestamp", "open", "high", "low", "close", "volume"]].copy()

df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(NY_TZ)

df = df[
    (df["timestamp"] >= market_open_ny) &
    (df["timestamp"] <= market_close_ny)
].copy()

df = df.sort_values("timestamp").reset_index(drop=True)

print(f"Downloaded {len(df):,} bars.")
print(df.head().to_string())


# ============================================================
# 4. OPTION A SIGNAL
# ============================================================

df["smooth_close"] = df["close"].ewm(
    span=EMA_PRICE_SPAN,
    adjust=False
).mean()


# ============================================================
# 5. ESTIMATE GRADIENT / SLOPE AND CURVATURE
# ============================================================

# Slope = change in smoothed price over previous few minutes
df["slope"] = df["smooth_close"] - df["smooth_close"].shift(SLOPE_LOOKBACK)

# Curvature = change in slope
df["curvature"] = df["slope"] - df["slope"].shift(CURVATURE_LOOKBACK)

# Previous slope for sign-change detection
df["prev_slope"] = df["slope"].shift(1)

# Ignore early noisy section
ignore_before = pd.Timestamp(f"{target_day} {IGNORE_BEFORE_TIME}", tz=NY_TZ)
df["valid_time"] = df["timestamp"] >= ignore_before


# ============================================================
# 6. CONFIRMED STATIONARY POINTS
# These are detected after the sign change happens.
# ============================================================

df["confirmed_local_min"] = (
    df["valid_time"]
    & (df["prev_slope"] < 0)
    & (df["slope"] >= 0)
    & (df["curvature"] > MIN_ABS_CURVATURE)
)

df["confirmed_local_max"] = (
    df["valid_time"]
    & (df["prev_slope"] > 0)
    & (df["slope"] <= 0)
    & (df["curvature"] < -MIN_ABS_CURVATURE)
)


# ============================================================
# 7. EARLY WARNING STATIONARY POINTS
# These are predictions before the slope crosses zero.
# ============================================================

# Possible future local minimum:
# price is still falling, but slope is becoming less negative
df["early_local_min_warning"] = (
    df["valid_time"]
    & (df["slope"] < 0)
    & (df["curvature"] > MIN_ABS_CURVATURE)
    & (df["slope"].abs() >= MIN_ABS_SLOPE)
)

# Possible future local maximum:
# price is still rising, but slope is becoming less positive
df["early_local_max_warning"] = (
    df["valid_time"]
    & (df["slope"] > 0)
    & (df["curvature"] < -MIN_ABS_CURVATURE)
    & (df["slope"].abs() >= MIN_ABS_SLOPE)
)


# ============================================================
# 8. COOLDOWN FILTER
# Prevents repeated warnings every minute around the same turn.
# ============================================================

def apply_cooldown(data, signal_col, cooldown_minutes):
    data = data.copy()
    keep = []
    last_signal_time = None

    for _, row in data.iterrows():
        if not row[signal_col]:
            keep.append(False)
            continue

        if last_signal_time is None:
            keep.append(True)
            last_signal_time = row["timestamp"]
            continue

        minutes_since = (row["timestamp"] - last_signal_time).total_seconds() / 60

        if minutes_since >= cooldown_minutes:
            keep.append(True)
            last_signal_time = row["timestamp"]
        else:
            keep.append(False)

    return keep


df["early_local_min_warning_cd"] = apply_cooldown(
    df,
    "early_local_min_warning",
    COOLDOWN_MINUTES
)

df["early_local_max_warning_cd"] = apply_cooldown(
    df,
    "early_local_max_warning",
    COOLDOWN_MINUTES
)

df["confirmed_local_min_cd"] = apply_cooldown(
    df,
    "confirmed_local_min",
    COOLDOWN_MINUTES
)

df["confirmed_local_max_cd"] = apply_cooldown(
    df,
    "confirmed_local_max",
    COOLDOWN_MINUTES
)


# ============================================================
# 9. PRINT EVENTS
# ============================================================

early_min = df[df["early_local_min_warning_cd"]].copy()
early_max = df[df["early_local_max_warning_cd"]].copy()
confirmed_min = df[df["confirmed_local_min_cd"]].copy()
confirmed_max = df[df["confirmed_local_max_cd"]].copy()

print("\nEarly local min warnings:")
print(early_min[["timestamp", "close", "smooth_close", "slope", "curvature"]].head(30).to_string(index=False))

print("\nEarly local max warnings:")
print(early_max[["timestamp", "close", "smooth_close", "slope", "curvature"]].head(30).to_string(index=False))

print("\nConfirmed local mins:")
print(confirmed_min[["timestamp", "close", "smooth_close", "slope", "curvature"]].head(30).to_string(index=False))

print("\nConfirmed local maxs:")
print(confirmed_max[["timestamp", "close", "smooth_close", "slope", "curvature"]].head(30).to_string(index=False))


# ============================================================
# 10. PLOT PRICE + STATIONARY POINTS
# ============================================================

plt.figure(figsize=(14, 6))

plt.plot(
    df["timestamp"],
    df["close"],
    label="NVDA raw close",
    linewidth=1.0,
    alpha=0.35
)

plt.plot(
    df["timestamp"],
    df["smooth_close"],
    label=f"Option A: EMA{EMA_PRICE_SPAN} smoothed close",
    linewidth=1.8
)

# Early warnings
plt.scatter(
    early_min["timestamp"],
    early_min["smooth_close"],
    marker="^",
    s=90,
    label="Early local min warning"
)

plt.scatter(
    early_max["timestamp"],
    early_max["smooth_close"],
    marker="v",
    s=90,
    label="Early local max warning"
)

# Confirmed points
plt.scatter(
    confirmed_min["timestamp"],
    confirmed_min["smooth_close"],
    marker="o",
    s=80,
    label="Confirmed local min"
)

plt.scatter(
    confirmed_max["timestamp"],
    confirmed_max["smooth_close"],
    marker="x",
    s=90,
    label="Confirmed local max"
)

plt.title("NVDA Stationary Points using Gradient / Slope Sign Changes")
plt.xlabel("Time")
plt.ylabel("NVDA Price")
plt.legend()
plt.grid(True)
plt.show()


# ============================================================
# 11. PLOT SLOPE AND CURVATURE
# This helps you see why the detector made each call.
# ============================================================

plt.figure(figsize=(14, 5))

plt.plot(
    df["timestamp"],
    df["slope"],
    label="Slope / gradient",
    linewidth=1.5
)

plt.plot(
    df["timestamp"],
    df["curvature"],
    label="Curvature = change in slope",
    linewidth=1.2,
    linestyle="--"
)

plt.axhline(0, linestyle=":", linewidth=1)

plt.title("NVDA Option A Slope and Curvature")
plt.xlabel("Time")
plt.ylabel("Slope / Curvature")
plt.legend()
plt.grid(True)
plt.show()


# ============================================================
# 12. FORWARD RETURN CHECK
# Are early min/max warnings useful?
# ============================================================

FORWARD_WINDOWS = [5, 10, 15, 30]

for w in FORWARD_WINDOWS:
    df[f"future_return_{w}m"] = df["close"].shift(-w) / df["close"] - 1


def summarise_events(event_df, label):
    print(f"\n================ {label} ================")
    print(f"Number of events: {len(event_df)}")

    if event_df.empty:
        return

    for w in FORWARD_WINDOWS:
        col = f"future_return_{w}m"
        avg_ret = event_df[col].mean()
        median_ret = event_df[col].median()
        win_rate = (event_df[col] > 0).mean()

        print(f"{w}m forward:")
        print(f"  avg return:    {avg_ret:.4%}")
        print(f"  median return: {median_ret:.4%}")
        print(f"  win rate:      {win_rate:.2%}")


summarise_events(early_min, "Early local min warnings")
summarise_events(early_max, "Early local max warnings")
summarise_events(confirmed_min, "Confirmed local mins")
summarise_events(confirmed_max, "Confirmed local maxs")


# ============================================================
# 13. SAVE OUTPUT
# ============================================================

out_file = f"nvda_gradient_stationary_points_{target_day}.csv"
df.to_csv(out_file, index=False)

print(f"\nSaved output to: {out_file}")