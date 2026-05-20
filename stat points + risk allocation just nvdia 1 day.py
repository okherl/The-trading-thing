# ============================================================
# NVDA STATIONARY POINT STRATEGY BACKTEST
# Option A: EMA-smoothed close stationary detector
# Risk-based + signal-strength-based allocation
# Yesterday's NVDA 1-minute Alpaca data
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

ALPACA_API_KEY = ""
ALPACA_SECRET_KEY = ""

if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
    raise ValueError(
        "Missing Alpaca keys. Add ALPACA_API_KEY and ALPACA_SECRET_KEY to your .env file."
    )

data_client = StockHistoricalDataClient(
    ALPACA_API_KEY,
    ALPACA_SECRET_KEY
)

NY_TZ = ZoneInfo("America/New_York")
SYMBOL = "NVDA"

DATA_FEED = DataFeed.IEX
# DATA_FEED = DataFeed.SIP  # use if your Alpaca plan supports SIP

# Starting account
STARTING_CASH = 1500.00

# Stationary point detector settings
EMA_PRICE_SPAN = 9
SLOPE_LOOKBACK = 3
CURVATURE_LOOKBACK = 3

MIN_ABS_SLOPE = 0.015
MIN_ABS_CURVATURE = 0.005
COOLDOWN_MINUTES = 8

IGNORE_BEFORE_TIME = "09:45"
LAST_BUY_TIME = "15:30"
FORCE_SELL_TIME = "15:55"

# Risk / allocation settings
RISK_PER_TRADE_PCT = 0.004       # 0.4% of equity risked per trade
MAX_POSITION_FRACTION = 0.25     # max 25% of account in NVDA
CASH_BUFFER_FRACTION = 0.05      # keep 5% cash buffer
MIN_POSITION_VALUE = 50.00

VOL_LOOKBACK = 30                # use last 30 minutes of returns
VOL_MULTIPLIER = 5.0             # stop = volatility * multiplier
MIN_STOP_LOSS_PCT = 0.006        # 0.6% minimum stop
MAX_STOP_LOSS_PCT = 0.025        # 2.5% maximum stop

# Execution assumptions
SLIPPAGE_BPS = 2                 # 0.02% slippage each side
SLIPPAGE_RATE = SLIPPAGE_BPS / 10000

# Selling rules
TRAILING_STOP_PCT = 0.012        # 1.2% trailing stop from high after entry
PARTIAL_SELL_FRACTION = 0.50     # sell 50% at 1R


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
# 4. OPTION A SIGNAL + VWAP + RETURN DATA
# ============================================================

df["smooth_close"] = df["close"].ewm(
    span=EMA_PRICE_SPAN,
    adjust=False
).mean()

df["return_1m"] = df["close"].pct_change()

# Daily VWAP
df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
df["tpv"] = df["typical_price"] * df["volume"]
df["cum_tpv"] = df["tpv"].cumsum()
df["cum_vol"] = df["volume"].cumsum()
df["vwap"] = df["cum_tpv"] / df["cum_vol"]

# Slope = change in smoothed price over previous few minutes
df["slope"] = df["smooth_close"] - df["smooth_close"].shift(SLOPE_LOOKBACK)

# Curvature = change in slope
df["curvature"] = df["slope"] - df["slope"].shift(CURVATURE_LOOKBACK)

# Previous slope for sign-change detection
df["prev_slope"] = df["slope"].shift(1)

# Time filters
ignore_before = pd.Timestamp(f"{target_day} {IGNORE_BEFORE_TIME}", tz=NY_TZ)
last_buy_time = pd.Timestamp(f"{target_day} {LAST_BUY_TIME}", tz=NY_TZ)
force_sell_time = pd.Timestamp(f"{target_day} {FORCE_SELL_TIME}", tz=NY_TZ)

df["valid_time"] = df["timestamp"] >= ignore_before


# ============================================================
# 5. STATIONARY POINT DETECTION
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

df["early_local_min_warning"] = (
    df["valid_time"]
    & (df["slope"] < 0)
    & (df["curvature"] > MIN_ABS_CURVATURE)
    & (df["slope"].abs() >= MIN_ABS_SLOPE)
)

df["early_local_max_warning"] = (
    df["valid_time"]
    & (df["slope"] > 0)
    & (df["curvature"] < -MIN_ABS_CURVATURE)
    & (df["slope"].abs() >= MIN_ABS_SLOPE)
)


# ============================================================
# 6. COOLDOWN FILTER
# ============================================================

def apply_cooldown(data, signal_col, cooldown_minutes):
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
    df, "early_local_min_warning", COOLDOWN_MINUTES
)

df["early_local_max_warning_cd"] = apply_cooldown(
    df, "early_local_max_warning", COOLDOWN_MINUTES
)

df["confirmed_local_min_cd"] = apply_cooldown(
    df, "confirmed_local_min", COOLDOWN_MINUTES
)

df["confirmed_local_max_cd"] = apply_cooldown(
    df, "confirmed_local_max", COOLDOWN_MINUTES
)


# ============================================================
# 7. RISK + SIGNAL STRENGTH ALLOCATION FUNCTIONS
# ============================================================

def clamp(x, low, high):
    return max(low, min(x, high))


def estimate_recent_volatility(data, i, lookback=VOL_LOOKBACK):
    """
    Uses past 1-minute close returns only.
    Returns volatility as decimal, e.g. 0.002 = 0.2%.
    """
    start_i = max(0, i - lookback)
    recent_returns = data.loc[start_i:i, "return_1m"].dropna()

    if len(recent_returns) < max(5, lookback // 3):
        return None

    vol = recent_returns.std()

    if pd.isna(vol) or vol <= 0:
        return None

    return vol


def calculate_stop_loss_pct(data, i):
    """
    Volatility-adjusted stop.
    More volatile stock conditions get wider stop, which reduces position size.
    """
    recent_vol = estimate_recent_volatility(data, i)

    if recent_vol is None:
        return None, None

    stop_loss_pct = VOL_MULTIPLIER * recent_vol
    stop_loss_pct = clamp(stop_loss_pct, MIN_STOP_LOSS_PCT, MAX_STOP_LOSS_PCT)

    return stop_loss_pct, recent_vol


def calculate_signal_score(row):
    """
    Score from 0 to 10-ish.
    This is deliberately simple and explainable.
    """

    score = 0.0

    # Main buy signals
    if row["early_local_min_warning_cd"]:
        score += 5.0

    if row["confirmed_local_min_cd"]:
        score += 3.0

    # Stronger curvature means stronger bend upward
    if row["curvature"] > MIN_ABS_CURVATURE:
        curvature_strength = min(1.5, row["curvature"] / MIN_ABS_CURVATURE * 0.25)
        score += curvature_strength

    # Less negative slope means the fall is closer to ending
    if row["slope"] < 0:
        slope_softening_bonus = max(0.0, 1.0 - abs(row["slope"]) / 0.10)
        score += slope_softening_bonus

    # Price above VWAP is a trend/context bonus.
    # For a falling knife, this may be false, so it is only a bonus.
    if row["close"] > row["vwap"]:
        score += 1.0

    # Avoid buying too late if there is no actual min-type signal
    if not row["early_local_min_warning_cd"] and not row["confirmed_local_min_cd"]:
        score = 0.0

    return min(score, 10.0)


def signal_strength_factor(signal_score):
    """
    Converts score into allocation factor.
    """
    if signal_score < 6:
        return 0.0
    elif signal_score < 7:
        return 0.60
    elif signal_score < 8:
        return 0.80
    else:
        return 1.00


def calculate_buy_allocation(
    equity,
    cash,
    signal_score,
    stop_loss_pct
):
    """
    Buy allocation:
        position_value = risk_dollars / stop_loss_pct
    then adjusted by signal strength and capped by cash/account exposure.
    """

    factor = signal_strength_factor(signal_score)

    if factor <= 0:
        return {
            "ok": False,
            "reason": "signal too weak",
            "signal_factor": factor
        }

    risk_dollars = equity * RISK_PER_TRADE_PCT

    base_position_value = risk_dollars / stop_loss_pct
    adjusted_position_value = base_position_value * factor

    max_position_value = equity * MAX_POSITION_FRACTION
    usable_cash = cash * (1 - CASH_BUFFER_FRACTION)

    final_position_value = min(
        adjusted_position_value,
        max_position_value,
        usable_cash
    )

    if final_position_value < MIN_POSITION_VALUE:
        return {
            "ok": False,
            "reason": "position too small",
            "signal_factor": factor,
            "risk_dollars": risk_dollars,
            "position_value": final_position_value
        }

    actual_risk_dollars = final_position_value * stop_loss_pct

    return {
        "ok": True,
        "position_value": final_position_value,
        "risk_dollars": risk_dollars,
        "actual_risk_dollars": actual_risk_dollars,
        "signal_factor": factor
    }


# ============================================================
# 8. BACKTEST ENGINE
# ============================================================

cash = STARTING_CASH
position = None
trade_log = []
equity_curve = []

def current_equity(price):
    if position is None:
        return cash
    return cash + position["qty"] * price


def buy_nvda(i, reason):
    global cash, position, trade_log

    row = df.loc[i]
    ts = row["timestamp"]
    raw_price = row["close"]

    if position is not None:
        return

    if ts > last_buy_time:
        return

    stop_loss_pct, recent_vol = calculate_stop_loss_pct(df, i)

    if stop_loss_pct is None:
        return

    equity = current_equity(raw_price)
    signal_score = calculate_signal_score(row)

    allocation = calculate_buy_allocation(
        equity=equity,
        cash=cash,
        signal_score=signal_score,
        stop_loss_pct=stop_loss_pct
    )

    if not allocation["ok"]:
        return

    buy_price = raw_price * (1 + SLIPPAGE_RATE)
    position_value = allocation["position_value"]
    qty = position_value / buy_price

    cash -= position_value

    position = {
        "symbol": SYMBOL,
        "qty": qty,
        "entry_price": buy_price,
        "entry_time": ts,
        "cost": position_value,
        "stop_loss_pct": stop_loss_pct,
        "stop_price": buy_price * (1 - stop_loss_pct),
        "risk_dollars": allocation["actual_risk_dollars"],
        "intended_risk_dollars": allocation["risk_dollars"],
        "signal_score": signal_score,
        "signal_factor": allocation["signal_factor"],
        "recent_vol": recent_vol,
        "partial_taken": False,
        "high_water": buy_price,
    }

    trade_log.append({
        "timestamp": ts,
        "side": "BUY",
        "price": buy_price,
        "qty": qty,
        "value": position_value,
        "cash_after": cash,
        "reason": reason,
        "signal_score": signal_score,
        "signal_factor": allocation["signal_factor"],
        "stop_loss_pct": stop_loss_pct,
        "stop_price": position["stop_price"],
        "risk_dollars": allocation["actual_risk_dollars"],
        "pnl": np.nan,
        "pnl_pct": np.nan
    })


def sell_nvda(i, fraction, reason):
    global cash, position, trade_log

    if position is None:
        return

    row = df.loc[i]
    ts = row["timestamp"]
    raw_price = row["close"]

    sell_price = raw_price * (1 - SLIPPAGE_RATE)

    fraction = clamp(fraction, 0.0, 1.0)
    qty_to_sell = position["qty"] * fraction

    if qty_to_sell <= 0:
        return

    proceeds = qty_to_sell * sell_price
    cost_basis_sold = position["entry_price"] * qty_to_sell

    pnl = proceeds - cost_basis_sold
    pnl_pct = sell_price / position["entry_price"] - 1

    cash += proceeds

    trade_log.append({
        "timestamp": ts,
        "side": "SELL",
        "price": sell_price,
        "qty": qty_to_sell,
        "value": proceeds,
        "cash_after": cash,
        "reason": reason,
        "signal_score": position.get("signal_score", np.nan),
        "signal_factor": position.get("signal_factor", np.nan),
        "stop_loss_pct": position.get("stop_loss_pct", np.nan),
        "stop_price": position.get("stop_price", np.nan),
        "risk_dollars": position.get("risk_dollars", np.nan),
        "pnl": pnl,
        "pnl_pct": pnl_pct
    })

    position["qty"] -= qty_to_sell

    if position["qty"] <= 1e-10 or fraction >= 0.999:
        position = None


def manage_position(i):
    global position

    if position is None:
        return

    row = df.loc[i]
    ts = row["timestamp"]
    price = row["close"]

    position["high_water"] = max(position["high_water"], price)

    unrealized_pnl = position["qty"] * (price - position["entry_price"])
    pnl_pct = price / position["entry_price"] - 1
    drawdown_from_high = price / position["high_water"] - 1

    # 1. Forced end-of-day liquidation
    if ts >= force_sell_time:
        sell_nvda(i, 1.0, "forced end-of-day sell")
        return

    # 2. Hard stop loss
    if price <= position["stop_price"]:
        sell_nvda(i, 1.0, "stop loss hit")
        return

    # 3. Trailing stop, only if trade is profitable
    if pnl_pct > 0 and drawdown_from_high <= -TRAILING_STOP_PCT:
        sell_nvda(i, 1.0, "trailing stop hit")
        return

    # 4. Partial profit at 1R
    if not position["partial_taken"] and unrealized_pnl >= position["risk_dollars"]:
        sell_nvda(i, PARTIAL_SELL_FRACTION, "1R reached, sell half")

        if position is not None:
            position["partial_taken"] = True
            position["stop_price"] = position["entry_price"]  # move stop to breakeven

        return

    # 5. Sell into local max signal
    if row["confirmed_local_max_cd"]:
        sell_nvda(i, 1.0, "confirmed local max, sell all")
        return

    # 6. Early local max warning:
    # If already took partial, sell remaining.
    # If not partial yet but profitable, sell half.
    if row["early_local_max_warning_cd"]:
        if position["partial_taken"]:
            sell_nvda(i, 1.0, "early local max after partial, sell rest")
        elif pnl_pct > 0:
            sell_nvda(i, 0.50, "early local max while profitable, sell half")

            if position is not None:
                position["partial_taken"] = True
                position["stop_price"] = position["entry_price"]

        return


print("\nRunning NVDA stationary-point allocation backtest...")

for i in range(len(df)):
    row = df.loc[i]
    ts = row["timestamp"]
    price = row["close"]

    # Manage existing position first
    manage_position(i)

    # Entry signal: buy on early or confirmed local min
    if position is None:
        if row["early_local_min_warning_cd"]:
            buy_nvda(i, "early local min warning")
        elif row["confirmed_local_min_cd"]:
            buy_nvda(i, "confirmed local min")

    equity_curve.append({
        "timestamp": ts,
        "cash": cash,
        "equity": current_equity(price),
        "close": price,
        "has_position": position is not None,
        "position_qty": 0 if position is None else position["qty"]
    })

# Final liquidation if somehow still open
if position is not None:
    sell_nvda(len(df) - 1, 1.0, "final liquidation")

equity_curve.append({
    "timestamp": df.loc[len(df) - 1, "timestamp"],
    "cash": cash,
    "equity": cash,
    "close": df.loc[len(df) - 1, "close"],
    "has_position": False,
    "position_qty": 0
})

equity_df = pd.DataFrame(equity_curve).drop_duplicates("timestamp").set_index("timestamp")
trades_df = pd.DataFrame(trade_log)

print("Backtest complete.")


# ============================================================
# 9. RESULTS
# ============================================================

final_equity = equity_df["equity"].iloc[-1]
total_return = final_equity / STARTING_CASH - 1

equity_df["running_max"] = equity_df["equity"].cummax()
equity_df["drawdown"] = equity_df["equity"] / equity_df["running_max"] - 1
max_drawdown = equity_df["drawdown"].min()

if not trades_df.empty:
    sell_trades = trades_df[trades_df["side"] == "SELL"].copy()
else:
    sell_trades = pd.DataFrame()

if not sell_trades.empty:
    total_pnl = sell_trades["pnl"].sum()
    win_rate = (sell_trades["pnl"] > 0).mean()
    avg_win = sell_trades.loc[sell_trades["pnl"] > 0, "pnl"].mean()
    avg_loss = sell_trades.loc[sell_trades["pnl"] <= 0, "pnl"].mean()

    gross_profit = sell_trades.loc[sell_trades["pnl"] > 0, "pnl"].sum()
    gross_loss = -sell_trades.loc[sell_trades["pnl"] <= 0, "pnl"].sum()
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
else:
    total_pnl = 0
    win_rate = np.nan
    avg_win = np.nan
    avg_loss = np.nan
    profit_factor = np.nan

# NVDA buy-and-hold benchmark for same day
first_close = df["close"].iloc[0]
last_close = df["close"].iloc[-1]
nvda_bh_return = last_close / first_close - 1
nvda_bh_final = STARTING_CASH * (1 + nvda_bh_return)

print("\n================ BACKTEST RESULTS ================")
print(f"Symbol:              {SYMBOL}")
print(f"Date:                {target_day}")
print(f"Starting cash:       ${STARTING_CASH:,.2f}")
print(f"Final equity:        ${final_equity:,.2f}")
print(f"Bot return:          {total_return:.2%}")
print(f"NVDA buy-hold final: ${nvda_bh_final:,.2f}")
print(f"NVDA buy-hold ret:   {nvda_bh_return:.2%}")
print(f"Max drawdown:        {max_drawdown:.2%}")
print(f"Total realised PnL:  ${total_pnl:.2f}")
print(f"Number of trades:    {len(trades_df)}")
print(f"Number of sells:     {len(sell_trades)}")

if not sell_trades.empty:
    print(f"Win rate:            {win_rate:.2%}")
    print(f"Average win:         ${avg_win:.2f}" if not pd.isna(avg_win) else "Average win:         N/A")
    print(f"Average loss:        ${avg_loss:.2f}" if not pd.isna(avg_loss) else "Average loss:        N/A")
    print(f"Profit factor:       {profit_factor:.2f}")


# ============================================================
# 10. PLOTS
# ============================================================

buy_trades = trades_df[trades_df["side"] == "BUY"].copy() if not trades_df.empty else pd.DataFrame()
sell_trades = trades_df[trades_df["side"] == "SELL"].copy() if not trades_df.empty else pd.DataFrame()

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
    label=f"EMA{EMA_PRICE_SPAN} smooth close",
    linewidth=1.8
)

if not buy_trades.empty:
    plt.scatter(
        buy_trades["timestamp"],
        buy_trades["price"],
        marker="^",
        s=110,
        label="Backtest BUY"
    )

if not sell_trades.empty:
    plt.scatter(
        sell_trades["timestamp"],
        sell_trades["price"],
        marker="v",
        s=110,
        label="Backtest SELL"
    )

plt.title("NVDA Stationary Strategy Backtest: Buy/Sell Points")
plt.xlabel("Time")
plt.ylabel("NVDA Price")
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(14, 5))

plt.plot(
    equity_df.index,
    equity_df["equity"],
    label="Bot equity",
    linewidth=1.8
)

nvda_bh_curve = STARTING_CASH * (
    df.set_index("timestamp")["close"].reindex(equity_df.index).ffill() / first_close
)

plt.plot(
    equity_df.index,
    nvda_bh_curve,
    label="NVDA buy-and-hold",
    linewidth=1.5,
    linestyle="--"
)

plt.title("Equity Curve: Bot vs NVDA Buy-and-Hold")
plt.xlabel("Time")
plt.ylabel("Portfolio Value ($)")
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(14, 4))

plt.plot(
    equity_df.index,
    equity_df["drawdown"],
    label="Bot drawdown",
    linewidth=1.5
)

plt.axhline(0, linestyle=":", linewidth=1)
plt.title("Bot Drawdown")
plt.xlabel("Time")
plt.ylabel("Drawdown")
plt.legend()
plt.grid(True)
plt.show()


# ============================================================
# 11. TRADE LOG + SAVE OUTPUT
# ============================================================

if trades_df.empty:
    print("\nNo trades were placed.")
else:
    print("\nTrade log:")
    print(trades_df.to_string(index=False))

    print("\nSell reason counts:")
    print(sell_trades["reason"].value_counts().to_string())

out_trade_file = f"nvda_stationary_risk_backtest_trades_{target_day}.csv"
out_equity_file = f"nvda_stationary_risk_backtest_equity_{target_day}.csv"
out_signal_file = f"nvda_stationary_risk_backtest_signals_{target_day}.csv"

trades_df.to_csv(out_trade_file, index=False)
equity_df.to_csv(out_equity_file)
df.to_csv(out_signal_file, index=False)

print(f"\nSaved trades to: {out_trade_file}")
print(f"Saved equity curve to: {out_equity_file}")
print(f"Saved signal file to: {out_signal_file}")
