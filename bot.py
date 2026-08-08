"""
╔══════════════════════════════════════════════════════════════════╗
║     SOLUSDT 5m — 9 EMA / 9 SMA Crossover Backtester             ║
║     Entry  : BUY when 9-EMA crosses ABOVE its 9-SMA             ║
║     SL     : 1.5× ATR below entry                               ║
║     TP     : 5× SL distance above entry                         ║
║     Fee    : 0.08% round-trip (Binance)                         ║
║     Account: $1 000 with compounding                            ║
╚══════════════════════════════════════════════════════════════════╝

All parameters are customisable at the top of the CONFIG section.
Run:  python solusdt_ema_sma_backtest.py
"""

import sys
import time
import math
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

# ─────────────────────────────────────────────────────────────────
#  CONFIG  ── change anything here
# ─────────────────────────────────────────────────────────────────
SYMBOL          = "SOLUSDT"          # Binance symbol
INTERVAL        = "5m"               # Kline interval
LOOKBACK_DAYS   = 365                # How many days of history to fetch

EMA_LEN         = 9                  # Fast EMA length
SMA_LEN         = 9                  # Smoothing SMA length (applied to the EMA)

ATR_LEN         = 14                 # ATR period for SL sizing
SL_ATR_MULT     = 1.5                # SL = entry − (SL_ATR_MULT × ATR)
TP_SL_MULT      = 3.0                # TP = entry + (TP_SL_MULT × SL_distance)

INITIAL_CAPITAL = 1_000.0            # Starting account size in USD
FEE_RT_PCT      = 0.08               # Round-trip fee in percent (0.08 = 0.08 %)
RISK_PCT        = 1.0                # % of current equity risked per trade (for position sizing)

# ─────────────────────────────────────────────────────────────────
#  BINANCE DATA FETCH
# ─────────────────────────────────────────────────────────────────
BINANCE_SPOT_URL    = "https://api.binance.com/api/v3/klines"
BINANCE_FUTURES_URL = "https://fapi.binance.com/fapi/v1/klines"
MAX_KLINES_PER_REQ  = 1_000          # Binance limit per request

def fetch_klines(symbol: str, interval: str, days: int) -> pd.DataFrame:
    """Fetch OHLCV klines from Binance (spot, with futures fallback)."""
    end_ms   = int(datetime.now(timezone.utc).timestamp() * 1_000)
    start_ms = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1_000)

    all_rows = []
    current_start = start_ms
    url = BINANCE_SPOT_URL

    print(f"\n📡  Fetching {symbol} {interval} from Binance …")
    req_count = 0

    while current_start < end_ms:
        params = {
            "symbol":    symbol,
            "interval":  interval,
            "startTime": current_start,
            "endTime":   end_ms,
            "limit":     MAX_KLINES_PER_REQ,
        }
        try:
            resp = requests.get(url, params=params, timeout=20)
        except Exception as e:
            sys.exit(f"❌  Network error: {e}")

        if resp.status_code == 451:
            # geo-blocked on spot → try futures
            url = BINANCE_FUTURES_URL
            continue
        if resp.status_code != 200:
            sys.exit(f"❌  Binance returned HTTP {resp.status_code}: {resp.text[:200]}")

        batch = resp.json()
        if not batch:
            break

        all_rows.extend(batch)
        last_open_ms = batch[-1][0]
        if last_open_ms == current_start:
            break
        current_start = last_open_ms + 1
        req_count += 1

        # polite rate-limiting
        if req_count % 5 == 0:
            time.sleep(0.5)

    if not all_rows:
        sys.exit("❌  No kline data returned — check symbol / interval.")

    df = pd.DataFrame(all_rows, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","quote_vol","trades","taker_buy_base",
        "taker_buy_quote","ignore"
    ])
    df = df[["open_time","open","high","low","close","volume"]].copy()
    for col in ["open","high","low","close","volume"]:
        df[col] = df[col].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df.drop_duplicates("open_time", inplace=True)
    df.sort_values("open_time", inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"✅  {len(df):,} candles loaded  "
          f"({df['open_time'].iloc[0].strftime('%Y-%m-%d')} → "
          f"{df['open_time'].iloc[-1].strftime('%Y-%m-%d')})")
    return df


# ─────────────────────────────────────────────────────────────────
#  INDICATORS
# ─────────────────────────────────────────────────────────────────
def compute_indicators(df: pd.DataFrame,
                       ema_len: int, sma_len: int, atr_len: int) -> pd.DataFrame:
    df = df.copy()

    # 9 EMA on close
    df["ema"] = df["close"].ewm(span=ema_len, adjust=False).mean()

    # 9 SMA of the EMA (smoothing layer)
    df["sma_of_ema"] = df["ema"].rolling(sma_len).mean()

    # ATR (Wilder's method = RMA of True Range)
    high_low   = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close  = (df["low"]  - df["close"].shift()).abs()
    tr         = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"]  = tr.ewm(alpha=1 / atr_len, adjust=False).mean()

    # Crossover signals: EMA crosses ABOVE SMA-of-EMA
    ema_above_prev = df["ema"].shift(1) <= df["sma_of_ema"].shift(1)
    ema_above_now  = df["ema"] > df["sma_of_ema"]
    df["signal"]   = ema_above_prev & ema_above_now   # True on cross bar

    return df


# ─────────────────────────────────────────────────────────────────
#  BACKTEST ENGINE
# ─────────────────────────────────────────────────────────────────
def run_backtest(df: pd.DataFrame,
                 initial_capital: float,
                 sl_atr_mult: float,
                 tp_sl_mult:  float,
                 fee_rt_pct:  float,
                 risk_pct:    float) -> dict:

    equity       = initial_capital
    peak_equity  = initial_capital
    max_drawdown = 0.0          # as a positive fraction, e.g. 0.15 = 15 %
    fee_rt       = fee_rt_pct / 100.0

    trades       = []           # list of completed trade dicts
    in_trade     = False
    entry_price  = 0.0
    sl_price     = 0.0
    tp_price     = 0.0
    qty          = 0.0          # in SOL units
    trade_risk   = 0.0          # USD at risk (before fee)
    entry_time   = None

    equity_curve = [initial_capital]

    for i in range(1, len(df)):
        row  = df.iloc[i]
        prev = df.iloc[i - 1]

        if in_trade:
            # Check SL / TP using this candle's high & low
            hit_sl = row["low"]  <= sl_price
            hit_tp = row["high"] >= tp_price

            if hit_sl or hit_tp:
                # Conservative: if both hit on same candle, SL wins
                if hit_sl:
                    exit_price = sl_price
                    outcome    = "LOSS"
                    pnl_pts    = exit_price - entry_price          # negative
                else:
                    exit_price = tp_price
                    outcome    = "WIN"
                    pnl_pts    = exit_price - entry_price          # positive

                gross_pnl = pnl_pts * qty
                fee_cost  = entry_price * qty * fee_rt             # round-trip fee
                net_pnl   = gross_pnl - fee_cost
                equity   += net_pnl

                trades.append({
                    "entry_time":  entry_time,
                    "exit_time":   row["open_time"],
                    "entry_price": entry_price,
                    "exit_price":  exit_price,
                    "sl":          sl_price,
                    "tp":          tp_price,
                    "qty":         qty,
                    "gross_pnl":   gross_pnl,
                    "fee":         fee_cost,
                    "net_pnl":     net_pnl,
                    "outcome":     outcome,
                    "equity":      equity,
                })

                # Update drawdown
                if equity > peak_equity:
                    peak_equity = equity
                dd = (peak_equity - equity) / peak_equity
                if dd > max_drawdown:
                    max_drawdown = dd

                in_trade = False
                equity_curve.append(equity)

        # New signal — only enter if NOT already in a trade
        if not in_trade and prev["signal"]:
            atr_val = prev["atr"]
            if math.isnan(atr_val) or atr_val <= 0:
                continue

            entry_price = row["open"]               # enter on next bar open
            sl_distance = sl_atr_mult * atr_val
            sl_price    = entry_price - sl_distance
            tp_price    = entry_price + tp_sl_mult * sl_distance

            # Position sizing: risk_pct % of current equity
            risk_usd  = equity * (risk_pct / 100.0)
            # qty such that (entry − sl) × qty = risk_usd
            qty       = risk_usd / sl_distance
            # Deduct entry fee immediately
            entry_fee = entry_price * qty * (fee_rt / 2)   # half of round-trip on entry
            equity   -= entry_fee

            in_trade   = True
            entry_time = row["open_time"]
            trade_risk = risk_usd

    # If still in trade at end, close at last close price
    if in_trade:
        last = df.iloc[-1]
        exit_price = last["close"]
        gross_pnl  = (exit_price - entry_price) * qty
        fee_cost   = entry_price * qty * fee_rt
        net_pnl    = gross_pnl - fee_cost
        equity    += net_pnl
        trades.append({
            "entry_time":  entry_time,
            "exit_time":   last["open_time"],
            "entry_price": entry_price,
            "exit_price":  exit_price,
            "sl":          sl_price,
            "tp":          tp_price,
            "qty":         qty,
            "gross_pnl":   gross_pnl,
            "fee":         fee_cost,
            "net_pnl":     net_pnl,
            "outcome":     "OPEN→CLOSED",
            "equity":      equity,
        })
        equity_curve.append(equity)

    return {
        "trades":       trades,
        "equity_curve": equity_curve,
        "final_equity": equity,
        "max_drawdown": max_drawdown,
    }


# ─────────────────────────────────────────────────────────────────
#  STATISTICS
# ─────────────────────────────────────────────────────────────────
def compute_stats(result: dict, initial_capital: float) -> dict:
    trades  = result["trades"]
    n       = len(trades)
    if n == 0:
        return {"total_trades": 0}

    wins    = [t for t in trades if t["outcome"] == "WIN"]
    losses  = [t for t in trades if t["outcome"] == "LOSS"]

    n_win   = len(wins)
    n_loss  = len(losses)
    win_rt  = n_win / n if n else 0.0

    avg_win  = np.mean([t["net_pnl"] for t in wins])   if wins   else 0.0
    avg_loss = np.mean([t["net_pnl"] for t in losses]) if losses else 0.0

    # Expected Value per trade (in USD)
    ev = win_rt * avg_win + (1 - win_rt) * avg_loss

    total_fees = sum(t["fee"] for t in trades)
    net_profit = result["final_equity"] - initial_capital
    roi_pct    = net_profit / initial_capital * 100.0

    # Max Drawdown
    mdd_pct    = result["max_drawdown"] * 100.0

    # Profit Factor
    gross_wins  = sum(t["net_pnl"] for t in wins)
    gross_losses = abs(sum(t["net_pnl"] for t in losses))
    pf = gross_wins / gross_losses if gross_losses > 0 else float("inf")

    # Consecutive stats
    outcomes = [1 if t["outcome"] == "WIN" else 0 for t in trades]
    max_cons_win = max_cons_loss = cur = 0
    cur_type = None
    for o in outcomes:
        if o == cur_type:
            cur += 1
        else:
            cur_type = o
            cur = 1
        if o == 1:
            max_cons_win  = max(max_cons_win,  cur)
        else:
            max_cons_loss = max(max_cons_loss, cur)

    return {
        "total_trades":     n,
        "wins":             n_win,
        "losses":           n_loss,
        "win_rate_pct":     win_rt * 100.0,
        "avg_win_usd":      avg_win,
        "avg_loss_usd":     avg_loss,
        "ev_per_trade_usd": ev,
        "profit_factor":    pf,
        "total_fees_usd":   total_fees,
        "net_profit_usd":   net_profit,
        "roi_pct":          roi_pct,
        "final_equity_usd": result["final_equity"],
        "max_drawdown_pct": mdd_pct,
        "max_cons_wins":    max_cons_win,
        "max_cons_losses":  max_cons_loss,
    }


# ─────────────────────────────────────────────────────────────────
#  PRETTY PRINT REPORT
# ─────────────────────────────────────────────────────────────────
def print_report(stats: dict, cfg: dict) -> None:
    sep = "─" * 54

    print("\n")
    print("╔══════════════════════════════════════════════════════╗")
    print("║        SOLUSDT 5m │ 9 EMA × 9 SMA Crossover         ║")
    print("╚══════════════════════════════════════════════════════╝")

    print(f"\n{'CONFIG':}")
    print(sep)
    print(f"  Symbol          : {cfg['symbol']}  ({cfg['interval']})")
    print(f"  EMA length      : {cfg['ema_len']}")
    print(f"  SMA length      : {cfg['sma_len']}  (smoothing on EMA)")
    print(f"  ATR length      : {cfg['atr_len']}")
    print(f"  SL multiplier   : {cfg['sl_atr_mult']}× ATR")
    print(f"  TP multiplier   : {cfg['tp_sl_mult']}× SL distance")
    print(f"  Round-trip fee  : {cfg['fee_rt_pct']}%")
    print(f"  Risk per trade  : {cfg['risk_pct']}% of equity")
    print(f"  Starting capital: ${cfg['initial_capital']:,.2f}")
    print(f"  Lookback        : {cfg['lookback_days']} days")

    if stats.get("total_trades", 0) == 0:
        print("\n⚠️   No trades generated — try adjusting parameters.")
        return

    print(f"\n{'TRADE SUMMARY':}")
    print(sep)
    print(f"  Total trades    : {stats['total_trades']}")
    print(f"  Wins            : {stats['wins']}")
    print(f"  Losses          : {stats['losses']}")
    print(f"  Win rate        : {stats['win_rate_pct']:.1f}%")
    print(f"  Max consec wins : {stats['max_cons_wins']}")
    print(f"  Max consec loss : {stats['max_cons_losses']}")

    print(f"\n{'PERFORMANCE':}")
    print(sep)
    print(f"  Avg win  (net)  : ${stats['avg_win_usd']:+.2f}")
    print(f"  Avg loss (net)  : ${stats['avg_loss_usd']:+.2f}")
    print(f"  EV / trade      : ${stats['ev_per_trade_usd']:+.2f}")
    print(f"  Profit factor   : {stats['profit_factor']:.2f}")
    print(f"  Total fees paid : ${stats['total_fees_usd']:.2f}")

    print(f"\n{'ACCOUNT':}")
    print(sep)
    print(f"  Starting equity : ${cfg['initial_capital']:,.2f}")
    print(f"  Final equity    : ${stats['final_equity_usd']:,.2f}")
    net = stats['net_profit_usd']
    roi = stats['roi_pct']
    sign = "+" if net >= 0 else ""
    print(f"  Net profit      : {sign}${net:,.2f}  ({sign}{roi:.1f}%)")
    print(f"  Max drawdown    : {stats['max_drawdown_pct']:.1f}%")

    ev  = stats['ev_per_trade_usd']
    ev_label = "✅ Positive EV" if ev > 0 else "❌ Negative EV"
    print(f"\n  {ev_label}  (EV = ${ev:+.2f}/trade)")
    print(sep)
    print()


# ─────────────────────────────────────────────────────────────────
#  OPTIONAL: SAVE TRADE LOG TO CSV
# ─────────────────────────────────────────────────────────────────
def save_trade_log(trades: list, filename: str = "trade_log.csv") -> None:
    if not trades:
        return
    df = pd.DataFrame(trades)
    df.to_csv(filename, index=False)
    print(f"📄  Trade log saved → {filename}")


# ─────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    cfg = {
        "symbol":          SYMBOL,
        "interval":        INTERVAL,
        "lookback_days":   LOOKBACK_DAYS,
        "ema_len":         EMA_LEN,
        "sma_len":         SMA_LEN,
        "atr_len":         ATR_LEN,
        "sl_atr_mult":     SL_ATR_MULT,
        "tp_sl_mult":      TP_SL_MULT,
        "initial_capital": INITIAL_CAPITAL,
        "fee_rt_pct":      FEE_RT_PCT,
        "risk_pct":        RISK_PCT,
    }

    # 1. Fetch data
    df = fetch_klines(cfg["symbol"], cfg["interval"], cfg["lookback_days"])

    # 2. Compute indicators
    print("📊  Computing indicators …")
    df = compute_indicators(df, cfg["ema_len"], cfg["sma_len"], cfg["atr_len"])

    signal_count = df["signal"].sum()
    print(f"🔔  Crossover signals found : {signal_count:,}")

    # 3. Run backtest
    print("🔁  Running backtest …")
    result = run_backtest(
        df,
        initial_capital = cfg["initial_capital"],
        sl_atr_mult     = cfg["sl_atr_mult"],
        tp_sl_mult      = cfg["tp_sl_mult"],
        fee_rt_pct      = cfg["fee_rt_pct"],
        risk_pct        = cfg["risk_pct"],
    )

    # 4. Stats & report
    stats = compute_stats(result, cfg["initial_capital"])
    print_report(stats, cfg)

    # 5. Save trade log
    save_trade_log(result["trades"])

    # 6. Mini equity curve (ASCII sparkline)
    curve = result["equity_curve"]
    if len(curve) > 1:
        lo, hi = min(curve), max(curve)
        rng    = hi - lo or 1
        bars   = "▁▂▃▄▅▆▇█"
        width  = min(60, len(curve))
        step   = max(1, len(curve) // width)
        spark  = ""
        for k in range(0, len(curve), step):
            idx   = min(int((curve[k] - lo) / rng * 7), 7)
            spark += bars[idx]
        print(f"  Equity curve  : {spark}")
        print(f"  (${lo:,.0f} → ${hi:,.0f})\n")


if __name__ == "__main__":
    main()
